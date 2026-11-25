"""代码/表格与相邻描述文本的关联。

检索语义上，紧贴代码/表格的一小段文字往往是它们的说明（“如下：”“上述代码”
“该表列出……”）。如果把它们拆成独立 chunk，代码/表格会失去上下文，召回
质量下降。本模块在切片前做一次轻量关联：

1. 直接前导：紧邻特殊块、同一标题路径、以冒号/“如下/例如/代码”等结尾；
2. 直接后续：紧邻特殊块、同一标题路径、以“上述/该/其中/注意”等开头；
3. 弱关联：没有明确提示词时，表格**之前**的连续题注/引用
   （“表2 生化检查”“见表1。”）整体吸附；表格之后的普通正文不靠弱关联
   猜测，只有 `注：/备注：` 等显式后续说明才进入第 2 步。

只有不超过 ``max_tokens`` 的短文本才会被吸附；长段落仍作为独立正文。
被吸附的块不再单独切片，其文本和 block_id 交给代码/表格切片器，
使特殊块 chunk 自带描述。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from uuid import UUID

from rag.shared.enums import BlockType
from rag.shared.models import ContentBlock
from utils.tokens import estimate_tokens

__all__ = [
    "FOLLOW_UP_HINTS",
    "LEAD_IN_HINTS",
    "SpecialContext",
    "associate_special_contexts",
]

# 可作为描述/说明的正文块类型。
_PROSE_TYPES = frozenset(
    {BlockType.PARAGRAPH, BlockType.LIST, BlockType.FOOTNOTE}
)
# 需要吸附描述的特殊块类型。
_SPECIAL_TYPES = frozenset({BlockType.TABLE, BlockType.CODE})

# 前导/后续描述提示词（内置词表）。
LEAD_IN_HINTS = frozenset(
    {
        "如下",
        "以下",
        "例如",
        "比如",
        "示例",
        "代码",
        "表格",
        "下表",
        "上图",
        "下图",
        "调用",
        "传入",
        "写法",
        "实现",
        "输出",
    }
)
FOLLOW_UP_HINTS = frozenset(
    {
        "上述",
        "这段",
        "该",
        "其中",
        "这里",
        "注意",
        "另外",
        "后续",
        "例如",
        "比如",
        "如图",
        "如表",
        "上表",
        "下表",
        "代码",
        "示例",
        "输出",
        "返回",
        "结果",
        "说明",
        # 表格后的注释行；要求带冒号，避免把“注射……”等正文当说明。
        "注：",
        "注:",
        "备注：",
        "备注:",
        "note:",
        "note：",
    }
)


@dataclass(slots=True)
class SpecialContext:
    """吸附到代码/表格上的描述文本。

    ``before_*`` / ``after_*`` 保留描述相对特殊块的位置，便于整节组装时
    按原始阅读顺序还原；``text`` / ``block_ids`` 是两者合并后的视图，
    供单独切片的代码/表格 chunk 使用。
    """

    text: str
    block_ids: list[UUID] = field(default_factory=list)
    before_text: str = ""
    before_block_ids: list[UUID] = field(default_factory=list)
    after_text: str = ""
    after_block_ids: list[UUID] = field(default_factory=list)
    token_count: int = 0


def _is_prose(block: ContentBlock) -> bool:
    """判断块是否可能作为描述文本。"""
    return block.block_type in _PROSE_TYPES


def _is_special(block: ContentBlock) -> bool:
    """判断块是否需要吸附描述。"""
    return block.block_type in _SPECIAL_TYPES


# 表题：表1/表2-1/表B.2/Table S1；图注不吸附到表格。
_TABLE_CAPTION_START_RE = re.compile(
    r"^\s*(?:附?表|续表|table)\s*[A-Za-z]?\s*[.\-·]?\s*\d",
    re.IGNORECASE,
)
# 表引用：“见表1。”“结果如表2所示”“按表B.3执行”。
_TABLE_REFERENCE_RE = re.compile(
    r"^\s*.{0,24}?(?:见|如|按|依据|参照|参考|详见).{0,8}?"
    r"(?:附?表|续表|table)\s*[A-Za-z]?\s*[.\-·]?\s*\d",
    re.IGNORECASE,
)


def _is_table_caption_or_reference(text: str) -> bool:
    """判断文本是否是表题或“见表X”型引用（可安全并入表格描述）。"""
    value = (text or "").strip()
    return bool(
        _TABLE_CAPTION_START_RE.match(value)
        or _TABLE_REFERENCE_RE.match(value)
    )


def _same_heading(left: ContentBlock, right: ContentBlock) -> bool:
    """只有同一标题路径下的文本才允许吸附，避免跨章节误关联。"""
    return left.heading_path == right.heading_path


def _lead_in_hint(text: str) -> bool:
    """判断文本是否像特殊块的前导说明。"""
    value = text.strip()
    if not value:
        return False
    if value.endswith(("：", ":")):
        return True
    tail = value[-20:].lower()
    return any(hint in tail for hint in LEAD_IN_HINTS)


def _follow_up_hint(text: str) -> bool:
    """判断文本是否像特殊块的后续说明。"""
    value = text.lstrip().lower()
    return any(
        value.startswith(hint) for hint in FOLLOW_UP_HINTS
    )


def associate_special_contexts(
    blocks: list[ContentBlock],
    *,
    max_tokens: int = 80,
    short_tokens: int = 40,
) -> tuple[list[ContentBlock], dict[UUID, SpecialContext]]:
    """把紧贴代码/表格的短文本吸附到特殊块，并返回过滤后的块列表。

    返回值：
    - ``filtered_blocks``：移除了已吸附描述块的块列表（标题/特殊块保留）；
    - ``contexts``：特殊块 id → ``SpecialContext``。

    前导/后续提示词使用本模块内置词表。
    """
    if not blocks or max_tokens <= 0:
        return blocks, {}

    consumed: set[UUID] = set()
    attachments: dict[UUID, list[tuple[int, str, UUID, bool]]] = {}
    attached_tokens: dict[UUID, int] = {}

    def attach(special_index: int, prose_index: int) -> None:
        """尝试把 ``prose_index`` 处的正文吸附到 ``special_index`` 处的特殊块。"""
        if prose_index < 0 or prose_index >= len(blocks):
            return
        special = blocks[special_index]
        prose = blocks[prose_index]
        if not _is_special(special) or not _is_prose(prose):
            return
        if prose.block_id in consumed:
            return
        if not _same_heading(special, prose):
            return
        tokens = estimate_tokens(prose.text)
        if tokens <= 0 or tokens > max_tokens:
            return
        if attached_tokens.get(special.block_id, 0) + tokens > max_tokens:
            return
        consumed.add(prose.block_id)
        attachments.setdefault(special.block_id, []).append(
            (
                prose_index,
                prose.text.strip(),
                prose.block_id,
                prose_index < special_index,
            )
        )
        attached_tokens[special.block_id] = (
            attached_tokens.get(special.block_id, 0) + tokens
        )

    special_indexes = [
        index for index, block in enumerate(blocks) if _is_special(block)
    ]

    # 1) 优先吸附“前导说明”，避免代码/表格与说明被拆开。
    for index in special_indexes:
        prose_index = index - 1
        if prose_index < 0:
            continue
        prose = blocks[prose_index]
        if _is_prose(prose) and _lead_in_hint(prose.text):
            attach(index, prose_index)

    # 2) 再吸附“后续说明”。
    for index in special_indexes:
        prose_index = index + 1
        if prose_index >= len(blocks):
            continue
        prose = blocks[prose_index]
        if _is_prose(prose) and _follow_up_hint(prose.text):
            attach(index, prose_index)

    # 3) 没有明确提示词时，紧邻的表题/表引用整体跟随特殊块。
    #    “表2 生化检查”“见表1。”这类题注常被 PDF 拆成多段；只把连续的
    #    题注/引用作为一个整体判断预算，避免把一句话拆成“正文 chunk +
    #    半个表格描述”，也避免把表前整节正文都吸进描述。
    #    只向前吸附表题/引用；表格之后的普通正文不靠弱关联猜，避免把
    #    浮动表格后面的整段正文（甚至下一小节）吸成表格描述。
    def attach_run(special_index: int, run: list[int]) -> None:
        """按整段预算吸附一组连续正文；放不下则整段放弃。"""
        if not run:
            return
        tokens = sum(estimate_tokens(blocks[item].text) for item in run)
        attached = attached_tokens.get(blocks[special_index].block_id, 0)
        if tokens <= 0 or attached + tokens > max_tokens:
            return
        # 弱关联只处理短文本：一段长正文属于章节内容，不是表题/说明。
        if any(
            estimate_tokens(blocks[item].text) > short_tokens for item in run
        ):
            return
        for prose_index in run:
            attach(special_index, prose_index)

    for index in special_indexes:
        special = blocks[index]
        # 向前收集连续的表题/表引用（如“表2 生化检查”“见表1。”），
        # 普通正文即使很短也不吸附，避免为一句题注把整节正文拆走。
        run: list[int] = []
        cursor = index - 1
        while cursor >= 0:
            prose = blocks[cursor]
            if not _is_prose(prose) or not _same_heading(special, prose):
                break
            if prose.block_id in consumed:
                cursor -= 1
                continue
            if not _is_table_caption_or_reference(prose.text):
                break
            run.append(cursor)
            cursor -= 1
        attach_run(index, list(reversed(run)))

    contexts: dict[UUID, SpecialContext] = {}
    for special_id, items in attachments.items():
        items.sort(key=lambda item: item[0])
        text = "\n".join(item[1] for item in items if item[1])
        if not text:
            continue
        before_items = [item for item in items if item[3]]
        after_items = [item for item in items if not item[3]]
        contexts[special_id] = SpecialContext(
            text=text,
            block_ids=[item[2] for item in items],
            before_text="\n".join(item[1] for item in before_items if item[1]),
            before_block_ids=[item[2] for item in before_items],
            after_text="\n".join(item[1] for item in after_items if item[1]),
            after_block_ids=[item[2] for item in after_items],
            token_count=estimate_tokens(text),
        )

    filtered = [
        block
        for block in blocks
        if block.block_type not in _PROSE_TYPES or block.block_id not in consumed
    ]
    return filtered, contexts
