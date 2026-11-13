"""文档级描述提取。

优先级：

1. ``DocumentSource.metadata["description"]``：调用方显式传入，直接可信；
2. 工作簿属性描述（Excel）：``TableBlock.summary``，可信；
3. 内容规则：开头附近出现 ``文档简介 / 文档说明 / 内容简介 / 简介 / 概述 /
   摘要`` 等可靠标题，且正文长度在 ``description_min_chars`` ~
   ``description_max_chars`` 之间；
4. 可读文件名：只作为候选/兜底，UUID、哈希、乱码、通用文件名会被过滤；
5. 非可靠候选：交给可注入的 LLM refiner 判断/生成；默认不启用。

LLM 只通过 ``DescriptionRefiner`` 协议注入，主链路默认不导入任何模型客户端。
"""

from __future__ import annotations

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

# 可靠描述标题词表由索引策略管理（config/rag_policy.json），
# 这里只保留“像描述但不够可靠”的标题正则。

# 可能的描述标题（如“企业简介”“课程简介”）：只作为 LLM 候选，不直接采信。
_CANDIDATE_HEADING_RE = re.compile(
    r"^[一-龥A-Za-z0-9 ]{1,12}(?:简介|说明|概述|描述|summary|overview)$",
    re.IGNORECASE,
)
_DIRECT_DESCRIPTION_RE = re.compile(
    r"^(文档简介|文档说明|文档描述|内容简介|内容说明|内容概述|简介|概述|摘要)"
    r"\s*[：:]\s*(.+)$",
    re.IGNORECASE | re.DOTALL,
)


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
    ) -> None:
        """保存 ChatModel 句柄、生成参数与可选的 Prompt 注册表。"""
        self.chat_model = chat_model
        self.model_id = model_id
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.prompts = prompts or get_default_prompt_registry()

    async def refine(
        self,
        *,
        candidate: str,
        context: str,
        source: str,
        max_chars: int,
    ) -> str | None:
        """调用 ChatModel.complete，返回清理后的描述。"""
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
        return truncate(text, max_chars) if text else None


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


def _is_length_ok(text: str, config: IndexerConfig) -> bool:
    """可靠规则的字符数必须落在配置范围。"""
    return config.description_min_chars <= len(text) <= config.description_max_chars


def _find_workbook_description(parsed: ParsedDocument) -> str | None:
    """从 Excel parser 写入的 TableBlock.summary 中取工作簿描述。"""
    if parsed.profile.doc_type is not DocumentType.SPREADSHEET:
        return None
    for block in parsed.blocks:
        summary = (block.table.summary if block.table else None) or ""
        if summary.strip():
            return _clean_description(summary)
    return None


def _collect_section_body(
    blocks: list[ContentBlock],
    start_index: int,
    max_chars: int,
) -> str:
    """收集标题后的正文，直到下一个标题或非正文块。"""
    parts: list[str] = []
    for block in blocks[start_index : start_index + 4]:
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
        text = (block.text or "").strip()
        if text:
            parts.append(text)
        if sum(len(item) for item in parts) >= max_chars * 2:
            break
    return _clean_description("\n".join(parts))


def _extract_from_blocks(
    blocks: list[ContentBlock],
    config: IndexerConfig,
    policy: IndexerPolicy,
) -> tuple[str | None, bool, str | None]:
    """从开头附近的标题/段落中提取描述。

    返回 ``(text, reliable, candidate_source)``。
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

    for index, block in enumerate(early):
        text = (block.text or "").strip()
        if not text:
            continue

        # 1) 段落直接写成“文档简介：xxx”
        if block.block_type in {BlockType.PARAGRAPH, BlockType.LIST}:
            match = _DIRECT_DESCRIPTION_RE.match(text)
            if match:
                body = _clean_description(match.group(2))
                if _is_length_ok(body, config):
                    return body, True, "heading_rule"
                if body:
                    return (
                        truncate(body, config.description_max_chars),
                        False,
                        "heading_candidate",
                    )

        # 2) 标题 + 后续正文
        if block.block_type is not BlockType.HEADING:
            continue
        if not _is_candidate_heading(text, policy):
            continue
        body = _collect_section_body(early, index + 1, config.description_max_chars)
        if not body:
            continue
        if _is_reliable_heading(text, policy) and _is_length_ok(body, config):
            return body, True, "heading_rule"
        return (
            truncate(body, config.description_max_chars),
            False,
            "heading_candidate",
        )
    return None, False, None


def _build_context(parsed: ParsedDocument, source: DocumentSource) -> str:
    """给 LLM 的上下文：文件名 + 开头文本，限制长度。"""
    parts = [f"文件名: {source.filename}"]
    for block in parsed.blocks[:20]:
        text = (block.text or "").strip()
        if text:
            parts.append(text)
        if sum(len(item) for item in parts) >= 2000:
            break
    return "\n".join(parts)[:2000]


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

        # 3) 内容规则
        text, reliable, rule_source = _extract_from_blocks(
            parsed.blocks,
            self.config,
            self.policy,
        )
        if text and reliable:
            return DescriptionExtraction(
                text=text,
                source=rule_source or "heading_rule",
                reliable=True,
                confidence=0.85,
            )

        # 4) 文件名候选：只接受可读、且长度足够的信息性文件名。
        # 注意：文件名候选只用于兜底，绝不交给 LLM。
        filename_text = filename_to_title(source.filename)
        filename_candidate = None
        if (
            self.config.description_use_filename
            and is_meaningful_filename(source.filename)
            and len(filename_text) >= 4
        ):
            filename_candidate = filename_text

        # 只有“从内容中提取到、但不够可靠”的候选才允许交给 LLM。
        weak_candidate = text if text and not reliable else None
        candidate = weak_candidate or filename_candidate
        candidate_source = (
            rule_source if weak_candidate else (
                "filename" if filename_candidate else None
            )
        )
        warnings: list[str] = []
        if weak_candidate:
            warnings.append("描述候选长度/来源不够可靠，未直接采用")

        # 5) LLM 判断/生成：仅处理 weak_candidate，默认关闭。
        if weak_candidate and self.config.description_llm_enabled:
            if self.refiner is None:
                warnings.append(
                    "description_llm_enabled=True 但未注入 DescriptionRefiner"
                )
            else:
                try:
                    refined = await self.refiner.refine(
                        candidate=weak_candidate,
                        context=_build_context(parsed, source),
                        source=candidate_source or "heading_candidate",
                        max_chars=self.config.description_max_chars,
                    )
                except Exception as exc:
                    warnings.append(f"LLM 描述生成失败: {exc}")
                    refined = None
                if refined and refined.strip():
                    return DescriptionExtraction(
                        text=truncate(
                            _clean_description(refined),
                            self.config.description_max_chars,
                        ),
                        source="llm",
                        reliable=True,
                        confidence=0.7,
                        candidate=weak_candidate,
                        warnings=warnings,
                    )
                warnings.append("LLM 未返回有效描述")

        # 6) 兜底：
        # - 没有 weak_candidate：只使用可读文件名，不调用 LLM；
        # - weak_candidate 但 LLM 关闭/失败：也回退文件名，不直接采用弱候选。
        if filename_candidate:
            return DescriptionExtraction(
                text=filename_candidate,
                source="filename",
                reliable=False,
                confidence=0.4,
                candidate=filename_candidate,
                warnings=warnings,
            )

        if weak_candidate:
            warnings.append("存在描述候选，但未启用/未成功调用 LLM，暂不采用")
        return DescriptionExtraction(
            text=None,
            source="candidate" if weak_candidate else "none",
            reliable=False,
            confidence=0.0,
            candidate=weak_candidate,
            warnings=warnings,
        )
