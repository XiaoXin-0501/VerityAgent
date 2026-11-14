"""文档级描述提取。

优先级：

1. ``DocumentSource.metadata["description"]``：调用方显式传入，直接可信；
2. 工作簿属性描述（Excel）：``TableBlock.summary``，可信；
3. 文章大标题 + 摘要/前言类章节：
   - 识别 level 1~3 的大标题，跳过栏目/作者/目录/章节编号，标题替代文件名；
   - 摘要/执行摘要/ABSTRACT/SUMMARY > 前言/序言/引言 > Introduction > 候选标题；
   - 支持标题块和同段标签（``前　言：`` / ``摘要：`` / ``译者按：``）；
   - 组合“标题 + 正文”，总长 ≤ ``description_max_chars`` 直接采用；
   - 超长或只有候选标题时，在 ``description_llm_enabled=true`` 且注入
     ``DescriptionRefiner`` 时交给 LLM 压缩/精炼；LLM 不可用按句截断兜底；
4. 可读文件名：只在没有标题/摘要/内容候选时兜底；UUID、哈希、乱码、
   通用文件名被过滤；**仅基于文件名的描述不调用 LLM**。

LLM 只通过 ``DescriptionRefiner`` 协议注入，主链路默认不导入任何模型客户端；
``bootstrap`` 可在配置了 ``description_model_id`` + ``description_chat_model``
时装配 ``ChatDescriptionRefiner``。
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol
from uuid import UUID

from config.rag import IndexerConfig
from prompt import PromptId, PromptRegistry, get_default_prompt_registry
from rag.shared.enums import BlockType, DocumentType
from rag.shared.models import (
    ContentBlock,
    DocumentDescriptionCandidate,
    DocumentSource,
    ParsedDocument,
)
from utils.text import (
    filename_to_title,
    is_meaningful_filename,
    normalize_whitespace,
    truncate,
)

from .caching import ChunkCache, description_cache_key
from .policy import (
    IndexerPolicy,
    get_indexer_policy,
    normalize_heading_key,
)

__all__ = [
    "DescriptionExtraction",
    "DescriptionRefiner",
    "CallableDescriptionRefiner",
    "ChatDescriptionRefiner",
    "DocumentDescriptionExtractor",
]

logger = logging.getLogger(__name__)

# 可靠描述标题词表由索引策略管理（config/rag_policy.json），
# 这里只保留“像描述但不够可靠”的标题正则。

# 可能的描述标题（如“企业简介”“课程简介”）：只作为 LLM 候选，不直接采信。
_CANDIDATE_HEADING_RE = re.compile(
    r"^[一-龥A-Za-z0-9 ]{1,12}(?:简介|说明|概述|描述|summary|overview)$",
    re.IGNORECASE,
)

# 摘要类标题优先级最高；前言/序言其次；Introduction 再次；候选标题最后。
# 归一化 key 与 policy.DescriptionPolicy.reliable_headings 同一口径。
_ABSTRACT_KEYS = frozenset(
    normalize_heading_key(item)
    for item in (
        "文档简介",
        "文档说明",
        "文档描述",
        "内容简介",
        "内容说明",
        "内容概述",
        "内容摘要",
        "内容提要",
        "简介",
        "概述",
        "摘要",
        "执行摘要",
        "提要",
        "abstract",
        "overview",
        "summary",
        "executive summary",
        "description",
    )
)
_PREFACE_KEYS = frozenset(
    normalize_heading_key(item)
    for item in ("前言", "序言", "引言", "导言", "preface", "foreword")
)
_INTRODUCTION_KEYS = frozenset({"introduction", "介绍"})

# 文献首页常见的栏目/元信息标题，不能当作文章大标题。
_TITLE_REJECT_KEYS = frozenset(
    normalize_heading_key(item)
    for item in (
        "case report",
        "case report open access",
        "open access",
        "review article",
        "original article",
        "research article",
        "作者",
        "author",
        "authors",
        "目录",
        "contents",
        "关键词",
        "keywords",
        "中文版翻译",
        "中文版审核",
        "译者按",
        "摘要",
        "abstract",
    )
)
# 描述正文遇到这些“标签型”段落应停止，避免把关键词/基金/收稿信息混进描述。
_STOP_SECTION_KEYS = frozenset(
    normalize_heading_key(item)
    for item in (
        "关键词",
        "关键字",
        "keywords",
        "key words",
        "基金项目",
        "基金资助",
        "作者简介",
        "通讯作者",
        "通信作者",
        "收稿日期",
        "修回日期",
        "中图分类号",
        "文献标识码",
        "文章编号",
        "doi",
        "correspondence",
        "funding",
        "conflict of interest",
    )
)
# 无可靠标题但明显在描述文档的标签，交给 LLM/截断兜底。
_CANDIDATE_INLINE_KEYS = frozenset(
    normalize_heading_key(item) for item in ("译者按", "内容提要")
)

# 章节编号标题（1. / 1.1 / 2 | ...）不是文章大标题。
_NUMBERED_HEADING_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*|[一二三四五六七八九十]+)\s*[.、|｜]\s*"
)
_TITLE_TOO_LONG = 300


@dataclass(slots=True)
class _RuleExtraction:
    """规则提取的中间结果；直接文本和待精炼候选分开。"""

    text: str | None = None
    source: str = "none"
    reliable: bool = False
    candidate: str | None = None
    candidate_source: str | None = None
    title: str | None = None


@dataclass(slots=True)
class _DescriptionSection:
    """开头附近找到的摘要/前言类章节。"""

    priority: int
    index: int
    body: str
    reliable: bool


_CJK_RE = re.compile(r"[一-鿿]")
_INLINE_LABEL_RE = re.compile(
    r"^\s*([^：:\n]{1,20})\s*[：:]\s*(.*)$",
    re.DOTALL,
)
_SENTENCE_END = "。！？.!?；;"


@dataclass(slots=True)
class DescriptionExtraction:
    """一次文档描述提取的结果（内部）。"""

    text: str | None = None
    source: str = "none"
    reliable: bool = False
    confidence: float = 0.0
    candidate: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_candidate(self, doc_id: UUID) -> DocumentDescriptionCandidate:
        """转成 `ParsedDocument.description` 字段。"""
        return DocumentDescriptionCandidate(
            doc_id=doc_id,
            description=self.text,
            source=self.source,
            confidence=self.confidence,
            candidate=self.candidate,
            warnings=list(self.warnings),
        )


class DescriptionRefiner(Protocol):
    """LLM 描述精炼器协议。"""

    async def refine(
        self,
        *,
        candidate: str,
        context: str,
        source: str,
        max_chars: int,
    ) -> str | None:
        """判断/生成文档描述；失败返回 None。"""
        ...


class CallableDescriptionRefiner:
    """把异步 callable 适配成 DescriptionRefiner。

    调用方可以在这里接入 ChatService / Provider；默认链路不会实例化。
    """

    def __init__(
        self,
        call: Callable[..., Awaitable[str | None]],
    ) -> None:
        """保存外部异步 callable；调用签名需与 ``refine`` 一致。"""
        self._call = call

    async def refine(
        self,
        *,
        candidate: str,
        context: str,
        source: str,
        max_chars: int,
    ) -> str | None:
        """执行外部 LLM 调用。"""
        return await self._call(
            candidate=candidate,
            context=context,
            source=source,
            max_chars=max_chars,
        )


class ChatDescriptionRefiner:
    """用现有 ChatModel 句柄实现 DescriptionRefiner。

    建议使用小型/中等规模的 chat 模型，purpose 设为
    ``ModelPurpose.DOCUMENT_DESCRIPTION``；默认参数温度低、输出短。
    System/User Prompt 统一由 :class:`PromptRegistry` 管理，不在本类写死。
    """

    def __init__(
        self,
        chat_model,
        *,
        model_id: str,
        temperature: float = 0.1,
        max_output_tokens: int = 256,
        prompts: PromptRegistry | None = None,
        cache: ChunkCache | None = None,
        cache_enabled: bool = True,
        cache_ttl_s: int = 2_592_000,
    ) -> None:
        """保存 ChatModel 句柄、生成参数、Prompt 注册表与可选描述缓存。"""
        self.chat_model = chat_model
        self.model_id = model_id
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.prompts = prompts or get_default_prompt_registry()
        self.cache = cache
        self.cache_enabled = bool(cache_enabled and cache is not None)
        self.cache_ttl_s = int(cache_ttl_s)

    async def refine(
        self,
        *,
        candidate: str,
        context: str,
        source: str,
        max_chars: int,
    ) -> str | None:
        """调用 ChatModel.complete，返回清理后的描述。

        文档描述同样按“模型 + Prompt 版本 + 候选/上下文摘要”缓存；重试时
        不再重复调用模型。
        """
        system = self.prompts.get(PromptId.RAG_DESCRIPTION_SYSTEM)
        user = self.prompts.get(PromptId.RAG_DESCRIPTION_USER)
        key = description_cache_key(
            model_id=self.model_id,
            prompt_id=str(PromptId.RAG_DESCRIPTION_USER),
            prompt_version=user.version,
            prompt_checksum=f"{system.checksum}:{user.checksum}",
            max_chars=max_chars,
            payload={
                "candidate": candidate,
                "context": context,
                "source": source,
            },
        )
        if self.cache_enabled and self.cache is not None:
            try:
                cached = await self.cache.get(key)
            except Exception as exc:  # noqa: BLE001 - 缓存失败只降级
                logger.warning("读取文档描述缓存失败: %s", exc)
                cached = None
            if isinstance(cached, dict):
                cached = cached.get("text")
            if isinstance(cached, str) and cached.strip():
                return cached.strip()

        # 局部导入，避免 RAG 模块在未启用 LLM 时加载 provider wire 类型。
        from provider.enums import ModelRole, ResponseFormatType
        from provider.llm.types import (
            ChatModelRequest,
            ModelGenerationOptions,
            ModelMessage,
            ModelResponseFormat,
            ModelTextPart,
        )

        request = ChatModelRequest(
            model=self.model_id,
            messages=[
                ModelMessage(
                    role=ModelRole.SYSTEM,
                    content=[
                        ModelTextPart(
                            text=self.prompts.render_text(
                                PromptId.RAG_DESCRIPTION_SYSTEM,
                                max_chars=max_chars,
                            )
                        )
                    ],
                ),
                ModelMessage(
                    role=ModelRole.USER,
                    content=[
                        ModelTextPart(
                            text=self.prompts.render_text(
                                PromptId.RAG_DESCRIPTION_USER,
                                source=source,
                                candidate=candidate,
                                context=context,
                            )
                        )
                    ],
                ),
            ],
            options=ModelGenerationOptions(
                temperature=self.temperature,
                top_p=1.0,
                max_output_tokens=self.max_output_tokens,
            ),
            response_format=ModelResponseFormat(
                type=ResponseFormatType.TEXT
            ),
            stream=False,
        )
        response = await self.chat_model.complete(request)
        text = "".join(
            part.text
            for part in response.message.content
            if isinstance(part, ModelTextPart)
        ).strip()
        result = truncate(text, max_chars) if text else None
        if result and self.cache_enabled and self.cache is not None:
            try:
                await self.cache.set(
                    key,
                    {"text": result},
                    ttl_s=self.cache_ttl_s,
                )
            except Exception as exc:  # noqa: BLE001 - 缓存失败只降级
                logger.warning("写入文档描述缓存失败: %s", exc)
        return result


def _is_reliable_heading(text: str, policy: IndexerPolicy) -> bool:
    """是否是可作为文档描述的可靠标题（词表来自索引策略）。"""
    key = normalize_heading_key(text)
    return key in policy.description.reliable_headings


def _is_candidate_heading(text: str, policy: IndexerPolicy) -> bool:
    """是否是“像描述但不够可靠”的标题，例如企业简介/课程简介。"""
    value = (text or "").strip()
    if not value or len(value) > 24:
        return False
    if _is_reliable_heading(value, policy):
        return True
    return bool(_CANDIDATE_HEADING_RE.match(value))


def _clean_description(text: str) -> str:
    """清理描述文本。"""
    return normalize_whitespace(text, keep_newlines=False).strip()


def _find_workbook_description(parsed: ParsedDocument) -> str | None:
    """从 Excel parser 写入的 TableBlock.summary 中取工作簿描述。"""
    if parsed.profile.doc_type is not DocumentType.SPREADSHEET:
        return None
    for block in parsed.blocks:
        summary = (block.table.summary if block.table else None) or ""
        if summary.strip():
            return _clean_description(summary)
    return None


def _description_heading_priority(
    text: str,
    policy: IndexerPolicy,
) -> int | None:
    """描述章节标题优先级：摘要 0、前言 1、Introduction 2、候选 3。"""
    key = normalize_heading_key(text)
    if not key:
        return None
    if key in _ABSTRACT_KEYS:
        return 0
    if key in _PREFACE_KEYS:
        return 1
    if key in _INTRODUCTION_KEYS:
        return 2
    if key in policy.description.reliable_headings:
        return 1
    if key in _CANDIDATE_INLINE_KEYS or _is_candidate_heading(text, policy):
        return 3
    return None


def _is_stop_section_text(text: str) -> bool:
    """关键词/基金/作者信息等标签型段落是描述章节的结束边界。"""
    match = _INLINE_LABEL_RE.match(text or "")
    label = match.group(1) if match else (text or "")[:24]
    return normalize_heading_key(label) in _STOP_SECTION_KEYS


def _split_inline_description(
    text: str,
    policy: IndexerPolicy,
) -> tuple[str, str] | None:
    """拆出 ``前　言：正文`` / ``摘要: 正文`` 这类同段标签。

    只有标签属于可靠描述标题/候选描述标签才拆分，避免把普通句子里的冒号
    误判成描述标签。
    """
    match = _INLINE_LABEL_RE.match(text or "")
    if match is None:
        return None
    label = match.group(1).strip()
    if _description_heading_priority(label, policy) is None:
        return None
    return label, match.group(2)


def _find_document_title(
    early: list[ContentBlock],
    policy: IndexerPolicy,
) -> str | None:
    """找文章大标题：第一个非栏目标题、非章节编号、非描述标题的标题块。

    PDF 首页常有 ``CASE REPORT OPEN ACCESS``（level 5）、``作者：``、
    ``中文版翻译`` 等栏目/元信息标题；这些不能替代文件名。真正的文章
    标题通常是 level 1~3，且带完整主题。
    """
    for block in early:
        if block.block_type is not BlockType.HEADING:
            continue
        text = _clean_description(block.text or "")
        if not text or len(text) < 4 or len(text) > _TITLE_TOO_LONG:
            continue
        key = normalize_heading_key(text)
        if key in _TITLE_REJECT_KEYS:
            continue
        if _description_heading_priority(text, policy) is not None:
            continue
        if _NUMBERED_HEADING_RE.match(text):
            continue
        if block.heading_level is not None and block.heading_level > 3:
            continue
        return text
    return None


def _compose_description(title: str | None, body: str) -> str:
    """组合“标题 + 摘要/前言正文”，并避免标题在正文开头重复。"""
    title = _clean_description(title or "")
    body = _clean_description(body or "")
    if not title:
        return body
    if not body:
        return title
    title_key = normalize_heading_key(title)
    body_key = normalize_heading_key(body)
    if title_key and (
        body_key.startswith(title_key)
        or title_key in body_key[: max(80, len(title_key) + 8)]
    ):
        return body
    separator = "：" if _CJK_RE.search(title) else ": "
    return f"{title}{separator}{body}"


def _truncate_description(text: str, max_chars: int) -> str:
    """按句末标点截断描述；找不到句末时退回词边界/硬截断。"""
    value = _clean_description(text)
    if len(value) <= max_chars:
        return value
    head = value[:max_chars]
    cut = max((head.rfind(mark) for mark in _SENTENCE_END), default=-1)
    if cut >= int(max_chars * 0.5):
        return head[: cut + 1].strip()
    # 英文摘要常见按空格分词；硬截断前尽量不切在单词中间。
    space = head.rfind(" ")
    if space >= int(max_chars * 0.6):
        return head[:space].rstrip()
    return head.rstrip()


def _truncated_source(source: str | None) -> str:
    """候选来源 → 截断兜底来源，便于审计是否真正经过 LLM。"""
    if not source:
        return "candidate_truncated"
    if source.endswith("_candidate"):
        return source[: -len("_candidate")] + "_truncated"
    if source.endswith("candidate"):
        return source[: -len("candidate")] + "truncated"
    return f"{source}_truncated"


def _collect_section_body(
    blocks: list[ContentBlock],
    start_index: int,
    max_chars: int,
    *,
    prefix: str = "",
) -> str:
    """收集标题/标签后的正文，直到下一个标题或关键词/基金等元信息。

    收集上限取 ``max(max_chars*4, 2000)``：既要判断是否超过描述上限，
    也要给 LLM 足够上下文；不是最终落库长度。
    """
    limit = max(2000, max_chars * 4)
    parts: list[str] = []
    if prefix.strip():
        parts.append(_clean_description(prefix))
    for block in blocks[start_index:]:
        if block.block_type in {
            BlockType.HEADING,
            BlockType.TABLE,
            BlockType.CODE,
            BlockType.FIGURE,
            BlockType.FORMULA,
        }:
            break
        if block.block_type not in {
            BlockType.PARAGRAPH,
            BlockType.LIST,
            BlockType.FOOTNOTE,
        }:
            continue
        text = _clean_description(block.text or "")
        if not text:
            continue
        if _is_stop_section_text(text):
            break
        parts.append(text)
        if sum(len(item) for item in parts) >= limit:
            break
    return _clean_description("\n".join(parts))


def _find_description_section(
    early: list[ContentBlock],
    policy: IndexerPolicy,
    max_chars: int,
) -> _DescriptionSection | None:
    """在开头附近选择最合适的摘要/前言章节。

    同一文档可能同时出现“译者按”“目录”“执行摘要”“Introduction”；
    不能按最先出现直接取。按“摘要 > 前言 > Introduction > 候选标题”的
    优先级选择，优先级相同取最早出现。
    """
    candidates: list[_DescriptionSection] = []
    for index, block in enumerate(early):
        text = _clean_description(block.text or "")
        if not text:
            continue
        if block.block_type is BlockType.HEADING:
            priority = _description_heading_priority(text, policy)
            if priority is None:
                continue
            body = _collect_section_body(
                early,
                index + 1,
                max_chars,
            )
            if not body:
                continue
            candidates.append(
                _DescriptionSection(
                    priority=priority,
                    index=index,
                    body=body,
                    reliable=priority < 3,
                )
            )
            continue
        if block.block_type not in {BlockType.PARAGRAPH, BlockType.LIST}:
            continue
        split = _split_inline_description(text, policy)
        if split is None:
            # 某些解析器把“摘要”“前言”单独识别成普通段落而不是 heading；
            # 只有整段就是描述标题时才向下收集，避免普通句子误触发。
            priority = _description_heading_priority(text, policy)
            if priority is None or len(text) > 12:
                continue
            label = text
            prefix = ""
        else:
            label, prefix = split
            priority = _description_heading_priority(label, policy)
        if priority is None:
            continue
        body = _collect_section_body(
            early,
            index + 1,
            max_chars,
            prefix=prefix,
        )
        if not body:
            continue
        candidates.append(
            _DescriptionSection(
                priority=priority,
                index=index,
                body=body,
                reliable=priority < 3,
            )
        )
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item.priority, item.index))


def _extract_from_blocks(
    blocks: list[ContentBlock],
    config: IndexerConfig,
    policy: IndexerPolicy,
) -> _RuleExtraction:
    """从开头附近组合“大标题 + 摘要/前言”或提取候选。

    规则：

    - 有文章大标题但无摘要/前言：标题直接替代文件名；
    - 标题 + 可靠摘要/前言且总长 ≤ ``description_max_chars``：直接采用；
    - 标题 + 摘要/前言过长：作为候选交给 LLM，LLM 不可用时按句截断兜底；
    - 只有候选标题（如“企业简介”）或“译者按”：作为弱候选，不直接冒充
      可靠描述。
    """
    early = [
        block
        for block in blocks
        if block.block_type
        in {
            BlockType.HEADING,
            BlockType.PARAGRAPH,
            BlockType.LIST,
            BlockType.FOOTNOTE,
        }
    ][: config.description_max_early_blocks]
    if not early:
        return _RuleExtraction()

    title = _find_document_title(early, policy)
    section = _find_description_section(
        early,
        policy,
        config.description_max_chars,
    )
    candidate_cap = max(2000, config.description_max_chars * 4)

    if section is not None:
        combined = _compose_description(title, section.body)
        if not combined:
            return _RuleExtraction(title=title)
        if section.reliable and len(combined) <= config.description_max_chars:
            # 有标题时标题已提供主题上下文，即使正文很短也可直接采用；
            # 无标题的短正文仍按 min_chars 判断，避免把一句普通开头当描述。
            if title or len(section.body) >= config.description_min_chars:
                return _RuleExtraction(
                    text=combined,
                    source=(
                        "title_heading_rule"
                        if title
                        else "heading_rule"
                    ),
                    reliable=True,
                    title=title,
                )
        candidate_source = (
            "title_heading_candidate" if title else "heading_candidate"
        )
        return _RuleExtraction(
            candidate=_truncate_description(combined, candidate_cap),
            candidate_source=candidate_source,
            title=title,
        )

    if title:
        # 没有摘要/前言时，大标题本身比文件名更能表达文档主题。
        return _RuleExtraction(
            text=_truncate_description(title, config.description_max_chars),
            source="title_rule",
            reliable=True,
            title=title,
        )
    return _RuleExtraction()


def _build_context(
    parsed: ParsedDocument,
    source: DocumentSource,
    *,
    title: str | None = None,
    section_text: str | None = None,
    max_chars: int = 2000,
) -> str:
    """给 LLM 的上下文：文件名 + 标题 + 候选段落；无候选时退回开头文本。"""
    parts = [f"文件名: {source.filename}"]
    if title:
        parts.append(f"文档标题: {title}")
    if section_text:
        parts.append(f"候选描述段落:\n{section_text}")
    else:
        for block in parsed.blocks[:20]:
            text = (block.text or "").strip()
            if text:
                parts.append(text)
            if sum(len(item) for item in parts) >= max_chars:
                break
    return "\n".join(parts)[:max_chars]


class DocumentDescriptionExtractor:
    """文档描述提取器。"""

    def __init__(
        self,
        config: IndexerConfig,
        refiner: DescriptionRefiner | None = None,
        *,
        policy: IndexerPolicy | None = None,
    ) -> None:
        """保存配置、可选的 LLM refiner 与索引策略。"""
        self.config = config
        self.refiner = refiner
        self.policy = policy or get_indexer_policy()

    async def extract(
        self,
        parsed: ParsedDocument,
        source: DocumentSource,
    ) -> DescriptionExtraction:
        """按优先级提取描述；返回内部结果。"""
        if not self.config.description_enabled:
            return DescriptionExtraction()

        # 1) 调用方显式描述
        manual = str(source.metadata.get("description") or "").strip()
        if manual:
            text = truncate(
                _clean_description(manual),
                self.config.description_max_chars,
            )
            return DescriptionExtraction(
                text=text,
                source="manual",
                reliable=True,
                confidence=0.95,
            )

        # 2) 工作簿属性描述
        workbook = _find_workbook_description(parsed)
        if workbook:
            text = truncate(workbook, self.config.description_max_chars)
            return DescriptionExtraction(
                text=text,
                source="workbook",
                reliable=True,
                confidence=0.9,
            )

        # 3) 内容规则：大标题 + 摘要/前言组合优先，标题本身可替代文件名。
        rule = _extract_from_blocks(
            parsed.blocks,
            self.config,
            self.policy,
        )
        if rule.text and rule.reliable:
            return DescriptionExtraction(
                text=rule.text,
                source=rule.source,
                reliable=True,
                confidence=0.9 if rule.source.startswith("title_") else 0.85,
            )

        # 4) 文件名候选：只接受可读、且长度足够的信息性文件名；
        # 文件名本身绝不交给 LLM。
        filename_text = filename_to_title(source.filename)
        filename_candidate = None
        if (
            self.config.description_use_filename
            and is_meaningful_filename(source.filename)
            and len(filename_text) >= 4
        ):
            filename_candidate = filename_text

        weak_candidate = rule.candidate
        candidate_source = rule.candidate_source
        warnings: list[str] = []
        if weak_candidate:
            if len(weak_candidate) > self.config.description_max_chars:
                warnings.append(
                    "标题+摘要/前言超过描述上限，未直接采用"
                )
            else:
                warnings.append(
                    "描述候选来源不够可靠，未直接采用"
                )

        # 5) LLM 生成/压缩：只处理内容候选（标题+摘要/前言、候选标题）。
        # 强规则直接命中或只有文件名时不调用 LLM。
        if weak_candidate and self.config.description_llm_enabled:
            if self.refiner is None:
                warnings.append(
                    "description_llm_enabled=True 但未注入 DescriptionRefiner"
                )
            else:
                llm_candidate = _truncate_description(
                    weak_candidate,
                    max(2000, self.config.description_max_chars * 4),
                )
                # 标题会在 LLM 输出前统一补回描述，因此给模型的字符预算要
                # 扣除标题和分隔符，避免最终组合后再次被截断。
                title_budget = (
                    len(rule.title) + 2 if rule.title else 0
                )
                generated_max_chars = max(
                    40,
                    self.config.description_max_chars - title_budget,
                )
                try:
                    refined = await self.refiner.refine(
                        candidate=llm_candidate,
                        context=_build_context(
                            parsed,
                            source,
                            title=rule.title,
                            section_text=llm_candidate,
                        ),
                        source=candidate_source or "heading_candidate",
                        max_chars=generated_max_chars,
                    )
                except Exception as exc:
                    warnings.append(f"LLM 描述生成失败: {exc}")
                    refined = None
                if refined and refined.strip():
                    refined_text = _clean_description(refined)
                    # LLM 可能改写标题；把文章标题补回描述，保证“标题+摘要”
                    # 的主题锚点不丢失（标题已在正文开头时不重复）。
                    if rule.title:
                        refined_text = _compose_description(
                            rule.title,
                            refined_text,
                        )
                    return DescriptionExtraction(
                        text=_truncate_description(
                            refined_text,
                            self.config.description_max_chars,
                        ),
                        source="llm",
                        reliable=True,
                        confidence=0.75,
                        candidate=weak_candidate,
                        warnings=warnings,
                    )
                warnings.append("LLM 未返回有效描述")

        # 6) 兜底：有“标题+摘要/前言”等候选时，按句截断直接采用，
        # 不再退回文件名；只有完全没有内容候选时才用文件名。
        if weak_candidate:
            fallback = _truncate_description(
                weak_candidate,
                self.config.description_max_chars,
            )
            if fallback:
                warnings.append(
                    "未获得 LLM 压缩，已按句截断采用候选描述"
                )
                return DescriptionExtraction(
                    text=fallback,
                    source=_truncated_source(candidate_source),
                    reliable=False,
                    confidence=0.55,
                    candidate=weak_candidate,
                    warnings=warnings,
                )

        if filename_candidate:
            return DescriptionExtraction(
                text=filename_candidate,
                source="filename",
                reliable=False,
                confidence=0.4,
                candidate=filename_candidate,
                warnings=warnings,
            )

        return DescriptionExtraction(
            text=None,
            source="none",
            reliable=False,
            confidence=0.0,
            candidate=None,
            warnings=warnings,
        )
