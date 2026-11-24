"""按标题层级组织的章节切片策略。

核心思路（由粗到细、按需下钻）：
1. 用解析器产出的 ``HEADING`` 块构建章节树，每个节点带完整标题路径；
2. 取文档中“第二浅”的标题层级作为默认切片层级（通常就是二级标题；
   如果文档只有一级标题，则用一级标题）；
3. 默认层级节点的整棵子树不超过 ``max_tokens`` 时，在该标题下打包正文，
   不继续下钻到三级/四级标题；**整节组装只适用于纯正文/列表**；
4. 只有超过 ``max_tokens`` 才进入下一级标题，递归处理；
5. 表格/代码保持独立切片（不变量：不跨 block_type），并携带其所在的最深
   标题路径；章节里只要出现特殊块，该章节不做整节组装；
6. 紧贴表格/代码的短描述会被吸附到该特殊块，避免代码/表格脱离语义。

这样既不会一上来就按最细标题切成一堆小 chunk，也不会让单个 chunk 超过
硬上限；标题路径只进入 embedding，不写入正文。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from uuid import UUID

from config.rag import IndexerConfig
from rag.shared.enums import (
    BlockType,
    DocumentType,
    QualityFlag,
    SectionType,
)
from rag.shared.models import ChunkDraft, ContentBlock, ProcessedBlock
from rag.shared.registry import ChunkerRegistry
from utils.text import smart_join, split_sentences
from utils.tokens import estimate_tokens

from .association import SpecialContext, associate_special_contexts
from .base import (
    BaseChunker,
    coalesce_text_chunks,
    merge_small_sibling_chunks,
    section_type_for,
)
from .packing import Unit, build_group_chunk, pack_units

__all__ = ["SectionNode", "SectionAwareChunker", "build_section_tree"]

# 参与“章节内正文打包”的块类型。
_TEXT_BLOCK_TYPES = frozenset(
    {BlockType.PARAGRAPH, BlockType.LIST, BlockType.FOOTNOTE}
)
# 独立切片、不与其他正文合并的块类型。
_SPECIAL_BLOCK_TYPES = frozenset({BlockType.TABLE, BlockType.CODE})

# Q&A 识别：优先显式 “问/答”“Q/A”“Question/Answer” 标记；
# 没有标记时只在 FAQ 类章节标题下用“问句结尾”兜底。
_QA_QUESTION_RE = re.compile(
    r"(?:^|(?<=[。！？!?；;\n]))\s*(?:问|问题|Q|Question)\s*\d*\s*[：:]",
    re.IGNORECASE | re.MULTILINE,
)
_QA_ANSWER_RE = re.compile(
    r"(?:^|(?<=[。！？!?；;\n]))\s*(?:答|回答|A|Answer)\s*\d*\s*[：:]",
    re.IGNORECASE | re.MULTILINE,
)
_QA_SECTION_HINTS = (
    "问题",
    "问答",
    "常见问题",
    "faq",
    "q&a",
    "frequently asked",
)


@dataclass(slots=True)
class _QaPair:
    """一组问答：问句、答案与原文片段。"""

    question: str
    answer: str
    text: str

    @property
    def token_count(self) -> int:
        """估算整组问答 token 数，用于决定是否拆分。"""
        return estimate_tokens(self.text)


def _is_qa_section(path: list[str]) -> bool:
    """章节标题是否明确是 FAQ/问答；无显式标记时只在这种情况下兜底。"""
    title = (path[-1] if path else "").lower()
    return any(hint in title for hint in _QA_SECTION_HINTS)


def _clean_question(value: str) -> str:
    """去掉问句前的“问：/Q:”标记，保留问题正文。"""
    match = _QA_QUESTION_RE.search(value)
    if match:
        return value[match.end() :].strip()
    return value.strip()


def _extract_qa_pairs(
    text: str,
    *,
    path: list[str],
) -> tuple[str, list[_QaPair]]:
    """从正文中抽取问答组，返回 ``(前缀, pairs)``。

    - 有显式问/答标记：按问句起始切分，问句到答句标记之间为问题；
    - 没有显式标记且章节是 FAQ：按“以问号结尾的句子”开新问答组；
    - 其他情况返回空 pairs，调用方继续走普通切片，避免影响正常正文。
    """
    explicit = list(_QA_QUESTION_RE.finditer(text or ""))
    if explicit and _QA_ANSWER_RE.search(text or ""):
        prefix = text[: explicit[0].start()].strip()
        pairs: list[_QaPair] = []
        for index, marker in enumerate(explicit):
            end = (
                explicit[index + 1].start()
                if index + 1 < len(explicit)
                else len(text)
            )
            raw = text[marker.start() : end].strip()
            answer_match = _QA_ANSWER_RE.search(raw)
            if answer_match:
                question_raw = _clean_question(raw[: answer_match.start()])
                # “问：…？补充说明。答：…” 中问号后的补充说明属于答案，
                # 不应进入 heading_path；问句只取到第一个问号。
                qmark = re.search(r"[？?]", question_raw)
                answer_prefix = ""
                if qmark:
                    answer_prefix = question_raw[qmark.end() :].strip()
                    question_raw = question_raw[: qmark.end()].strip()
                question = question_raw
                answer_text = _QA_ANSWER_RE.sub("", raw[answer_match.start() :], count=1)
                answer_text = answer_text.strip()
                if answer_prefix:
                    answer_text = f"{answer_prefix} {answer_text}".strip()
            else:
                question = _clean_question(raw)
                answer_text = ""
            if question:
                pairs.append(
                    _QaPair(
                        question=question,
                        answer=answer_text,
                        text=raw,
                    )
                )
        if pairs:
            return prefix, pairs

    if not _is_qa_section(path):
        return "", []

    # 无显式标记：FAQ 章节中，以问号结尾的句子视为新问题；后续句子为答案。
    sentences = split_sentences(text or "")
    pairs = []
    question = ""
    answer: list[str] = []
    for sentence in sentences:
        value = sentence.strip()
        if not value:
            continue
        if value.endswith(("？", "?")):
            if question:
                pairs.append(
                    _QaPair(
                        question=question,
                        answer=smart_join(answer).strip(),
                        text=f"{question}{smart_join(answer)}".strip(),
                    )
                )
            question = value
            answer = []
        elif question:
            answer.append(value)
    if question:
        pairs.append(
            _QaPair(
                question=question,
                answer="".join(answer).strip(),
                text=f"{question}{''.join(answer)}".strip(),
            )
        )
    return "", pairs


@dataclass(slots=True)
class SectionNode:
    """章节树节点：标题、完整路径、按文档顺序排列的内容/子节点。"""

    level: int
    title: str | None
    path: list[str]
    items: list[object] = field(default_factory=list)
    total_tokens: int = 0


def build_section_tree(blocks: list[ContentBlock]) -> SectionNode:
    """根据 HEADING 块构建章节树；内容块挂到当前最深标题下。"""
    root = SectionNode(level=0, title=None, path=[])
    stack: list[SectionNode] = [root]
    for block in blocks:
        if block.block_type is BlockType.HEADING:
            level = max(1, int(block.heading_level or 1))
            while len(stack) > 1 and stack[-1].level >= level:
                stack.pop()
            parent = stack[-1]
            node = SectionNode(
                level=level,
                title=block.text,
                path=[*parent.path, block.text],
            )
            parent.items.append(node)
            stack.append(node)
            continue
        stack[-1].items.append(block)
    return root


def _bbox_tuple(block: ContentBlock) -> tuple[float, float, float, float] | None:
    """返回块的 PDF 版面坐标；缺失/非法时返回 None。"""
    bbox = block.bbox
    if not bbox or len(bbox) != 4:
        return None
    try:
        return tuple(float(value) for value in bbox)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def _vertical_gap(
    previous: ContentBlock,
    current: ContentBlock,
) -> float | None:
    """计算同页、同栏相邻块的纵向间距；跨页/跨栏返回 None。"""
    if previous.page_no != current.page_no:
        return None
    prev_box = _bbox_tuple(previous)
    curr_box = _bbox_tuple(current)
    if prev_box is None or curr_box is None:
        return None
    overlap = min(prev_box[2], curr_box[2]) - max(prev_box[0], curr_box[0])
    if overlap <= 0:
        return None
    return curr_box[1] - prev_box[3]


def _visual_groups(
    blocks: list[ContentBlock],
) -> list[list[ContentBlock]] | None:
    """按“组内间距小、组间间距大”把块分成视觉组。

    这是问答之外更通用的分组兜底：术语/定义、步骤/说明、问答等都可能是
    “短起始块 + 紧跟内容 + 与下一组拉开间距”的结构。

    为避免影响普通正文，只有同时满足以下条件才启用：

    - 所有块都有 PDF bbox；
    - 至少 3 组、2 个明显的大间距；
    - 大多数组的首块是短块（≤80 token），普通段落通常不满足。
    """
    if len(blocks) < 4:
        return None
    gaps: list[float] = []
    for previous, current in zip(blocks, blocks[1:], strict=False):
        gap = _vertical_gap(previous, current)
        if gap is not None and gap >= 0:
            gaps.append(gap)
    if len(gaps) < 3:
        return None
    ordered = sorted(gaps)
    median = ordered[len(ordered) // 2]
    threshold = max(8.0, median * 1.8)
    large_gaps = [gap for gap in gaps if gap >= threshold]
    if len(large_gaps) < 2:
        return None

    groups: list[list[ContentBlock]] = [[blocks[0]]]
    for previous, current in zip(blocks, blocks[1:], strict=False):
        gap = _vertical_gap(previous, current)
        if gap is None or gap >= threshold:
            groups.append([current])
        else:
            groups[-1].append(current)
    if len(groups) < 3:
        return None
    short_starts = sum(
        1 for group in groups if estimate_tokens(group[0].text) <= 80
    )
    if short_starts / len(groups) < 0.7:
        return None
    return groups


def _is_chunkable(block: ContentBlock) -> bool:
    """判断块是否会产生 chunk；图/图注/目录等不参与 token 统计。"""
    return (
        block.block_type in _TEXT_BLOCK_TYPES
        or block.block_type in _SPECIAL_BLOCK_TYPES
    )


def _compute_totals(
    node: SectionNode,
    contexts: dict[UUID, SpecialContext],
) -> int:
    """后序计算每个节点子树的可切片 token 总量（含吸附描述）。"""
    total = 0
    for item in node.items:
        if isinstance(item, SectionNode):
            total += _compute_totals(item, contexts)
        elif isinstance(item, ContentBlock) and _is_chunkable(item):
            total += estimate_tokens(item.text)
            context = contexts.get(item.block_id)
            if context is not None:
                total += context.token_count
    node.total_tokens = total
    return total


def _collect_content_blocks(node: SectionNode) -> list[ContentBlock]:
    """按阅读顺序收集节点子树中的可切片内容块。"""
    blocks: list[ContentBlock] = []
    for item in node.items:
        if isinstance(item, SectionNode):
            blocks.extend(_collect_content_blocks(item))
        elif isinstance(item, ContentBlock) and _is_chunkable(item):
            blocks.append(item)
    return blocks


def _text_starts_with_title(
    segments: list[tuple[str, BlockType]],
    title: str,
) -> bool:
    """小节标题是否已出现在紧随其后的正文开头，避免重复写标签。"""
    compact_title = re.sub(r"\s+", "", title)
    if not compact_title:
        return False
    for value, block_type in segments:
        if block_type is BlockType.HEADING:
            return False
        return re.sub(r"\s+", "", value).startswith(compact_title)
    return False


def _assembly_segments(node: SectionNode) -> list[tuple[str, BlockType]]:
    """按阅读顺序收集整节组装用的文本段。

    小节标题以 ``HEADING`` 段插入，便于父节点吞下 2.1/2.2 等子节后仍保留
    最小语义标签；标题若已在首段正文开头出现则不重复插入。
    """
    segments: list[tuple[str, BlockType]] = []
    for item in node.items:
        if isinstance(item, SectionNode):
            if item.total_tokens <= 0:
                continue
            child_segments = _assembly_segments(item)
            title = (item.title or "").strip()
            if title and not _text_starts_with_title(child_segments, title):
                segments.append((title, BlockType.HEADING))
            segments.extend(child_segments)
        elif isinstance(item, ContentBlock) and item.block_type in _TEXT_BLOCK_TYPES:
            value = (item.text or "").strip()
            if value:
                segments.append((value, item.block_type))
    return segments


def _join_assembly_segments(
    segments: list[tuple[str, BlockType]],
) -> str:
    """拼接组装文本段：标题独立成行，连续正文按块类型拼接。"""
    lines: list[str] = []
    buffer: list[str] = []
    buffer_types: set[BlockType] = set()

    def flush() -> None:
        """输出当前连续正文段。"""
        if not buffer:
            return
        joined = (
            smart_join(buffer)
            if buffer_types <= {BlockType.PARAGRAPH, BlockType.FOOTNOTE}
            else "\n".join(buffer)
        ).strip()
        if joined:
            lines.append(joined)
        buffer.clear()
        buffer_types.clear()

    for value, block_type in segments:
        if block_type is BlockType.HEADING:
            flush()
            lines.append(value)
            continue
        buffer.append(value)
        buffer_types.add(block_type)
    flush()
    return "\n".join(lines).strip()


def _subtree_has_qa_section(node: SectionNode) -> bool:
    """子树中是否存在 FAQ/问答小节。

    父节点整节组装会把多个问答压成普通文本，因此只要子树里有 FAQ 小节，
    就回退到逐节处理，由 ``flush_text`` 保留问答原子性。
    """
    for item in node.items:
        if not isinstance(item, SectionNode):
            continue
        if _is_qa_section(item.path) or _subtree_has_qa_section(item):
            return True
    return False


def _base_heading_level(blocks: list[ContentBlock]) -> int:
    """选择默认切片层级：文档中“第二浅”的标题级别。

    例如 ``[1,2,3]`` → 2，``[1,5]`` → 5，``[1]`` → 1。默认从二级标题
    （或等价层级）开始，只有内容超限时才继续下钻。
    """
    levels = sorted(
        {
            max(1, int(block.heading_level or 1))
            for block in blocks
            if block.block_type is BlockType.HEADING
        }
    )
    if not levels:
        return 1
    return levels[1] if len(levels) > 1 else levels[0]


class SectionAwareChunker:
    """按章节层级进行由粗到细的切片。"""

    def __init__(
        self,
        config: IndexerConfig,
        chunkers: ChunkerRegistry,
    ) -> None:
        """保存配置与特殊块切片器注册表。"""
        self.config = config
        self.chunkers = chunkers
        # 唯一硬上限来源：IndexerConfig.max_chunk_tokens。
        self.max_tokens = config.max_chunk_tokens
        self.min_tokens = config.min_chunk_tokens
        self.target_tokens = min(config.chunk_size_tokens, self.max_tokens)
        self.packer = BaseChunker(
            target_tokens=self.target_tokens,
            max_tokens=self.max_tokens,
            min_tokens=config.min_chunk_tokens,
            overlap_tokens=config.chunk_overlap_tokens,
            chunking_version=config.chunking_version,
            chunk_id_namespace=config.chunk_id_namespace,
        )

    async def chunk(
        self,
        blocks: list[ContentBlock],
        *,
        kb_id: UUID,
        doc_type: DocumentType,
        document_description: str | None = None,
    ) -> list[ChunkDraft]:
        """把整篇文档的块切成 chunk 列表。"""
        contexts = {}
        if self.config.special_context_enabled:
            # 描述最多占 chunk 预算的一半，避免代码/表格本体没有空间。
            context_max = min(
                self.config.special_context_max_tokens,
                max(1, self.max_tokens // 2),
            )
            context_short = min(
                self.config.special_context_short_tokens,
                context_max,
            )
            blocks, contexts = associate_special_contexts(
                blocks,
                max_tokens=context_max,
                short_tokens=context_short,
            )
        root = build_section_tree(blocks)
        _compute_totals(root, contexts)
        base_level = _base_heading_level(blocks)

        chunks: list[ChunkDraft] = []
        index = 0

        def flush_text(buffer: list[ContentBlock], path: list[str]) -> None:
            """把缓存中的同类正文块打包成 chunk。"""
            nonlocal index
            if not buffer:
                return
            doc_id = buffer[0].doc_id
            block_type = buffer[0].block_type
            section_type = section_type_for(block_type)
            # 低置信度 OCR 文本保留，但打标供检索降权/过滤。
            low_ocr = any(
                block.metadata.get("ocr")
                and block.confidence is not None
                and block.confidence < self.config.quality.min_ocr_confidence
                for block in buffer
            )
            unit_flags = {QualityFlag.LOW_OCR} if low_ocr else set()
            joiner = (
                (lambda parts: "\n".join(parts))
                if section_type is SectionType.LIST
                else smart_join
            )
            block_ids = [block.block_id for block in buffer]
            page_no = next(
                (block.page_no or None for block in buffer if block.page_no),
                None,
            )
            language = next(
                (block.language for block in buffer if block.language),
                None,
            )
            heading_text = " > ".join(path) if path else None

            # FAQ/问答章节：按“一组问答”为原子单位打包，避免 Q/A 被 token
            # 边界拆开；普通正文没有显式问答标记，不会进入这个分支。
            joined_text = joiner(
                [block.text for block in buffer if block.text]
            )
            qa_prefix, qa_pairs = _extract_qa_pairs(joined_text, path=path)
            if qa_pairs:
                buffer.clear()

                def emit_qa_chunk(
                    pair_list: list[_QaPair],
                    *,
                    heading: str | None,
                    heading_path: list[str],
                    metadata: dict,
                ) -> None:
                    """输出一个不拆分的问答组 chunk。"""
                    nonlocal index
                    text = "\n\n".join(pair.text for pair in pair_list).strip()
                    if not text:
                        return
                    embedding = (
                        f"{heading}\n{text}" if heading else text
                    )
                    chunks.append(
                        self.packer.build_chunk_from_text(
                            doc_id=doc_id,
                            kb_id=kb_id,
                            block_type=block_type,
                            section_type=section_type,
                            index=index,
                            text=text,
                            embedding_text=embedding,
                            block_ids=block_ids,
                            page_no=page_no,
                            heading=heading,
                            heading_path=heading_path,
                            language=language,
                            quality_flags=set(unit_flags),
                            metadata=metadata,
                        )
                    )
                    index += 1

                # FAQ 章节开头的非问答正文（如导语）按普通文本输出。
                if qa_prefix:
                    prefix_units = [
                        Unit(text=piece)
                        for piece in (split_sentences(qa_prefix) or [qa_prefix])
                        if piece.strip()
                    ]
                    prefix_groups = pack_units(
                        prefix_units,
                        target_tokens=self.target_tokens,
                        max_tokens=self.max_tokens,
                        overlap_tokens=self.config.chunk_overlap_tokens,
                    )
                    for group in prefix_groups:
                        text = smart_join([unit.text for unit in group])
                        chunks.append(
                            self.packer.build_chunk_from_text(
                                doc_id=doc_id,
                                kb_id=kb_id,
                                block_type=block_type,
                                section_type=section_type,
                                index=index,
                                text=text,
                                embedding_text=(
                                    f"{heading_text}\n{text}"
                                    if heading_text
                                    else text
                                ),
                                block_ids=block_ids,
                                page_no=page_no,
                                heading=heading_text,
                                heading_path=path,
                                language=language,
                                quality_flags=set(unit_flags),
                                metadata={"qa_prefix": True},
                            )
                        )
                        index += 1

                bucket: list[_QaPair] = []
                bucket_tokens = 0

                def flush_bucket() -> None:
                    """输出当前短问答组；多个短 QA 共用一个 chunk。"""
                    nonlocal bucket, bucket_tokens
                    if not bucket:
                        return
                    emit_qa_chunk(
                        bucket,
                        heading=heading_text,
                        heading_path=path,
                        metadata={"qa": True, "qa_pairs": len(bucket)},
                    )
                    bucket = []
                    bucket_tokens = 0

                for pair in qa_pairs:
                    tokens = pair.token_count
                    if tokens > self.max_tokens:
                        # 单个问答超过硬上限：不拆问答对，但把问题放进 heading，
                        # 答案按句子/token 切分，所有子 chunk 共享同一问题。
                        flush_bucket()
                        units = [
                            Unit(text=piece)
                            for piece in (split_sentences(pair.text) or [pair.text])
                            if piece.strip()
                        ]
                        groups = pack_units(
                            units,
                            target_tokens=self.target_tokens,
                            max_tokens=self.max_tokens,
                            overlap_tokens=0,
                        )
                        for group in groups:
                            text = smart_join([unit.text for unit in group])
                            question_heading = " > ".join(
                                [*path, pair.question]
                            )
                            chunks.append(
                                self.packer.build_chunk_from_text(
                                    doc_id=doc_id,
                                    kb_id=kb_id,
                                    block_type=block_type,
                                    section_type=section_type,
                                    index=index,
                                    text=text,
                                    embedding_text=(
                                        f"{question_heading}\n{text}"
                                        if question_heading
                                        else text
                                    ),
                                    block_ids=block_ids,
                                    page_no=page_no,
                                    heading=question_heading,
                                    heading_path=[*path, pair.question],
                                    language=language,
                                    quality_flags=set(unit_flags),
                                    metadata={"qa": True, "qa_long": True},
                                )
                            )
                            index += 1
                        continue

                    if (
                        bucket
                        and bucket_tokens + tokens > self.target_tokens
                    ):
                        flush_bucket()
                    if tokens > self.target_tokens:
                        # 较长但不能拆：单独成 chunk。问题追加到原章节标题后，
                        # 保留完整层级（标题 > 问题），不覆盖原 heading。
                        flush_bucket()
                        emit_qa_chunk(
                            [pair],
                            heading=" > ".join([*path, pair.question]),
                            heading_path=[*path, pair.question],
                            metadata={"qa": True, "qa_long": True},
                        )
                        continue
                    bucket.append(pair)
                    bucket_tokens += tokens
                    if bucket_tokens >= self.target_tokens:
                        flush_bucket()
                flush_bucket()
                return

            # 通用“视觉分组”兜底：即使没有“问/答”等关键词，只要同标题下
            # 呈现“组内紧、组间松”的重复结构（术语/定义、步骤/说明等），
            # 也以组为单位打包，避免一组内容被 token 边界拆开。
            visual_groups = _visual_groups(buffer)
            if visual_groups:
                buffer.clear()

                def emit_visual_group(
                    group: list[ContentBlock],
                ) -> None:
                    """输出单个视觉组；组内不拆，问题/标题沿用章节标题。"""
                    nonlocal index
                    if not group:
                        return
                    text = smart_join(
                        [block.text for block in group if block.text]
                    ).strip()
                    if not text:
                        return
                    group_ids = [block.block_id for block in group]
                    group_page = next(
                        (block.page_no or None for block in group if block.page_no),
                        None,
                    )
                    group_language = next(
                        (block.language for block in group if block.language),
                        None,
                    )
                    chunks.append(
                        self.packer.build_chunk_from_text(
                            doc_id=doc_id,
                            kb_id=kb_id,
                            block_type=block_type,
                            section_type=section_type,
                            index=index,
                            text=text,
                            embedding_text=(
                                f"{heading_text}\n{text}"
                                if heading_text
                                else text
                            ),
                            block_ids=group_ids,
                            page_no=group_page,
                            heading=heading_text,
                            heading_path=path,
                            language=group_language,
                            quality_flags=set(unit_flags),
                            metadata={"visual_group": True},
                        )
                    )
                    index += 1

                visual_bucket: list[list[ContentBlock]] = []
                visual_tokens = 0

                def flush_visual_bucket() -> None:
                    """输出若干短视觉组；多组共用一个 chunk。"""
                    nonlocal visual_bucket, visual_tokens, index
                    if not visual_bucket:
                        return
                    text = "\n\n".join(
                        smart_join(
                            [block.text for block in group if block.text]
                        ).strip()
                        for group in visual_bucket
                    ).strip()
                    if not text:
                        visual_bucket = []
                        visual_tokens = 0
                        return
                    group_ids = [
                        block.block_id
                        for group in visual_bucket
                        for block in group
                    ]
                    group_page = next(
                        (
                            block.page_no or None
                            for group in visual_bucket
                            for block in group
                            if block.page_no
                        ),
                        None,
                    )
                    group_language = next(
                        (
                            block.language
                            for group in visual_bucket
                            for block in group
                            if block.language
                        ),
                        None,
                    )
                    chunks.append(
                        self.packer.build_chunk_from_text(
                            doc_id=doc_id,
                            kb_id=kb_id,
                            block_type=block_type,
                            section_type=section_type,
                            index=index,
                            text=text,
                            embedding_text=(
                                f"{heading_text}\n{text}"
                                if heading_text
                                else text
                            ),
                            block_ids=group_ids,
                            page_no=group_page,
                            heading=heading_text,
                            heading_path=path,
                            language=group_language,
                            quality_flags=set(unit_flags),
                            metadata={
                                "visual_group": True,
                                "visual_groups": len(visual_bucket),
                            },
                        )
                    )
                    index += 1
                    visual_bucket = []
                    visual_tokens = 0

                for group in visual_groups:
                    group_text = smart_join(
                        [block.text for block in group if block.text]
                    ).strip()
                    if not group_text:
                        continue
                    tokens = estimate_tokens(group_text)
                    if tokens > self.max_tokens:
                        flush_visual_bucket()
                        units = [
                            Unit(text=piece)
                            for piece in (split_sentences(group_text) or [group_text])
                            if piece.strip()
                        ]
                        groups = pack_units(
                            units,
                            target_tokens=self.target_tokens,
                            max_tokens=self.max_tokens,
                            overlap_tokens=0,
                        )
                        group_page = next(
                            (
                                block.page_no or None
                                for block in group
                                if block.page_no
                            ),
                            page_no,
                        )
                        group_language = next(
                            (
                                block.language
                                for block in group
                                if block.language
                            ),
                            language,
                        )
                        for packed in groups:
                            piece_text = smart_join(
                                [unit.text for unit in packed]
                            )
                            chunks.append(
                                self.packer.build_chunk_from_text(
                                    doc_id=doc_id,
                                    kb_id=kb_id,
                                    block_type=block_type,
                                    section_type=section_type,
                                    index=index,
                                    text=piece_text,
                                    embedding_text=(
                                        f"{heading_text}\n{piece_text}"
                                        if heading_text
                                        else piece_text
                                    ),
                                    block_ids=[
                                        block.block_id for block in group
                                    ],
                                    page_no=group_page,
                                    heading=heading_text,
                                    heading_path=path,
                                    language=group_language,
                                    quality_flags=set(unit_flags),
                                    metadata={
                                        "visual_group": True,
                                        "visual_group_split": True,
                                    },
                                )
                            )
                            index += 1
                        continue
                    if (
                        visual_bucket
                        and visual_tokens + tokens > self.target_tokens
                    ):
                        flush_visual_bucket()
                    if tokens > self.target_tokens:
                        flush_visual_bucket()
                        emit_visual_group(group)
                        continue
                    visual_bucket.append(group)
                    visual_tokens += tokens
                    if visual_tokens >= self.target_tokens:
                        flush_visual_bucket()
                flush_visual_bucket()
                return

            units: list[Unit] = []
            for block in buffer:
                text = (block.text or "").strip()
                if not text:
                    continue
                pieces = (
                    [line.strip() for line in text.split("\n") if line.strip()]
                    if section_type is SectionType.LIST
                    else split_sentences(text) or [text]
                )
                units.extend(
                    Unit(
                        text=piece,
                        block_id=block.block_id,
                        page_no=block.page_no or None,
                        language=block.language,
                        quality_flags=set(unit_flags),
                    )
                    for piece in pieces
                    if piece
                )
            buffer.clear()
            if not units:
                return
            groups = pack_units(
                units,
                target_tokens=self.target_tokens,
                max_tokens=self.max_tokens,
                overlap_tokens=self.config.chunk_overlap_tokens,
            )
            for group in groups:
                chunks.append(
                    build_group_chunk(
                        self.packer,
                        group,
                        doc_id=doc_id,
                        kb_id=kb_id,
                        index=index,
                        block_type=block_type,
                        section_type=section_type,
                        heading_text=heading_text,
                        heading_path=path,
                        joiner=joiner,
                        document_description=document_description,
                    )
                )
                index += 1

        async def emit_special(block: ContentBlock) -> None:
            """表格/代码走各自切片器，保留最深标题路径与吸附的描述。"""
            nonlocal index
            resolved = self.chunkers.resolve(doc_type, block.block_type)
            if resolved is None:
                return
            _spec, chunker = resolved
            context = contexts.get(block.block_id)
            metadata = {}
            if context is not None:
                metadata = {
                    "context_text": context.text,
                    "context_block_ids": context.block_ids,
                }
            if document_description:
                metadata["document_description"] = document_description
            produced = await chunker.chunk(
                block,
                ProcessedBlock(block=block, text=block.text, metadata=metadata),
                kb_id=kb_id,
                doc_type=doc_type,
                chunking_version=self.config.chunking_version,
                start_index=index,
            )
            chunks.extend(produced)
            index += len(produced)

        async def assemble(
            node: SectionNode,
            path: list[str],
            *,
            allow_visual_groups: bool = True,
        ) -> bool:
            """把“纯正文/列表”章节整体组装成一个语义完整的 chunk。

            优先让父节点吞下子树：只要整棵子树放得进 ``max_tokens``，就把
            2.1/2.2 等子节连标题一起组装；放不下时返回 ``False``，由调用方
            回退到逐节/按块处理（父级无法吞下才划分子集）。

            硬规则（《RAG 模块数据结构设计》§5.4 / §15）：表格、代码等特殊块
            必须各自独立切片，不能与正文混进同一个 chunk。因此本函数只处理
            ``_TEXT_BLOCK_TYPES``；章节里只要出现特殊块，调用方会回退到
            ``process_items``，由 TableChunker / CodeChunker 保留结构、表头
            和语义描述。
            """
            nonlocal index
            content = _collect_content_blocks(node)
            if not content or any(
                block.block_type not in _TEXT_BLOCK_TYPES
                for block in content
            ):
                return False

            block_ids: list[UUID] = []
            flags: set[QualityFlag] = set()
            block_types: set[BlockType] = set()
            page_no: int | None = None
            language: str | None = None

            def add_ids(ids: list[UUID]) -> None:
                """按首次出现顺序合并来源 block id。"""
                for block_id in ids:
                    if block_id not in block_ids:
                        block_ids.append(block_id)

            for block in content:
                block_types.add(block.block_type)
                add_ids([block.block_id])
                if (
                    block.metadata.get("ocr")
                    and block.confidence is not None
                    and block.confidence < self.config.quality.min_ocr_confidence
                ):
                    flags.add(QualityFlag.LOW_OCR)
                if page_no is None and block.page_no:
                    page_no = block.page_no
                if language is None and block.language:
                    language = block.language

            # 子节标题作为独立文本段保留，父节点吞下子树时仍能看到 2.1/2.2。
            segments = _assembly_segments(node)
            text = _join_assembly_segments(segments)
            if not text or estimate_tokens(text) > self.max_tokens:
                return False
            # FAQ/问答或视觉分组必须交给 process_items -> flush_text 处理：
            # 整节组装会把多个问答压成一个普通 chunk，丢失问答原子性和
            # 长问答按问题拆分的逻辑。返回 False 让调用方按块处理。
            if _extract_qa_pairs(text, path=path)[1] or _subtree_has_qa_section(
                node
            ):
                return False
            if allow_visual_groups and _visual_groups(content):
                return False

            if block_types == {BlockType.LIST}:
                section_type, block_type = SectionType.LIST, BlockType.LIST
            else:
                section_type, block_type = (
                    SectionType.PARAGRAPH,
                    BlockType.PARAGRAPH,
                )

            heading_text = " > ".join(path) if path else None
            # 文档描述只作为 chunk metadata 供展示/审计，不进入 embedding：
            # 同一文档的所有 chunk 共享描述会让向量趋同，稀释检索信号（D-103）。
            embedding_text = (
                f"{heading_text}\n{text}" if heading_text else text
            )
            metadata = {
                "mixed": len(block_types) > 1,
                "block_types": sorted(item.value for item in block_types),
            }
            # 记录被父节点吞下的子节标题，便于审计“哪些小节合并成了一个 chunk”。
            subsection_titles = [
                value
                for value, kind in segments
                if kind is BlockType.HEADING
            ]
            if subsection_titles:
                metadata["subsections"] = subsection_titles
            if document_description:
                metadata["document_description"] = document_description
            chunks.append(
                self.packer.build_chunk_from_text(
                    doc_id=content[0].doc_id,
                    kb_id=kb_id,
                    block_type=block_type,
                    section_type=section_type,
                    index=index,
                    text=text,
                    embedding_text=embedding_text,
                    block_ids=block_ids,
                    page_no=page_no,
                    heading=heading_text,
                    heading_path=path,
                    language=language,
                    quality_flags=flags,
                    metadata=metadata,
                )
            )
            index += 1
            return True

        async def process_items(node: SectionNode) -> None:
            """按文档顺序处理节点自身内容，并递归子标题。"""
            buffer: list[ContentBlock] = []
            for item in node.items:
                if isinstance(item, SectionNode):
                    flush_text(buffer, node.path)
                    await process(item)
                    continue
                if not isinstance(item, ContentBlock):
                    continue
                if item.block_type in _TEXT_BLOCK_TYPES:
                    if buffer and buffer[-1].block_type is not item.block_type:
                        flush_text(buffer, node.path)
                    buffer.append(item)
                elif item.block_type in _SPECIAL_BLOCK_TYPES:
                    flush_text(buffer, node.path)
                    await emit_special(item)
            flush_text(buffer, node.path)

        async def process(node: SectionNode) -> None:
            """按“默认层级优先、超限再下钻”的规则处理章节节点。

            - 比默认层级更浅的节点（如一级标题）：按文档顺序处理自身正文和子标题；
            - 默认层级及更深的节点：整棵子树不超过 max_tokens 时，只有“纯正文/
              列表”章节才整体组装成一个 chunk（子节标题一并写入正文）；
              含表格/代码的章节必须回退到按块处理，保证特殊块独立切片
              （§5.4 不变量）；
            - 超限时按文档顺序处理自身正文，并逐个递归子标题。
            """
            # 有内容的子小节：父节点放得下时一并吞下；放不下时逐个递归，
            # 子节内部继续按同样的“父级优先”规则尝试组装。
            child_sections = [
                item
                for item in node.items
                if isinstance(item, SectionNode) and item.total_tokens > 0
            ]
            # 父节点优先吞下整棵子树：只要 total_tokens 放得下，就把
            # 2.1/2.2 等子节连标题一起组装成一个 chunk；放不下才逐节处理，
            # 由下一层继续尝试“父级吞子树”，直到叶子或达到 max_tokens。
            if (
                node.level >= base_level
                and node.total_tokens <= self.max_tokens
            ):
                content = _collect_content_blocks(node)
                # 只有一个特殊块时仍走专门切片器，保留表格行组/代码元数据。
                if (
                    len(content) == 1
                    and content[0].block_type in _SPECIAL_BLOCK_TYPES
                ):
                    await emit_special(content[0])
                    return
                # 只有纯正文/列表才能整节组装；混入表格/代码时必须保持结构边界。
                # 有子小节时子节标题已经是最小语义边界，不再用 PDF 视觉间距
                # 把父节点拆开；视觉分组只用于没有子节的叶子章节。
                if await assemble(
                    node,
                    node.path,
                    allow_visual_groups=not child_sections,
                ):
                    return
            await process_items(node)

        await process(root)
        # 同一标题下被表格/代码隔开的正文片段再合并一次，减少碎片 chunk。
        chunks = coalesce_text_chunks(
            chunks,
            max_tokens=self.max_tokens,
            chunking_version=self.config.chunking_version,
            chunk_id_namespace=self.config.chunk_id_namespace,
        )
        # 药典/标准里 2.1/2.2 这种小节经常只有一两句话；在公共父标题下
        # 合并过小 chunk，避免产生大量极短片段。
        return merge_small_sibling_chunks(
            chunks,
            min_tokens=self.min_tokens,
            max_tokens=self.max_tokens,
            # 默认层级（病例报告的 1./2. 等）保持独立，只合并更深的小点。
            min_path_depth=base_level + 1,
            chunking_version=self.config.chunking_version,
            chunk_id_namespace=self.config.chunk_id_namespace,
        )
