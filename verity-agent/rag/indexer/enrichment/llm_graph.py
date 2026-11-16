"""LLM 图抽取器：领域提示词驱动，失败自动降级。

设计原则：

- **Prompt 由注册表管理**：system/user 模板在 ``prompt/templates/rag``，
  本类只负责组装变量（领域类型/关系/规则/数量上限）；
- **领域知识来自画像**：实体类型、关系白名单、角色说明都在
  ``DomainProfile.prompt`` 里，换领域不改代码；
- **词典做归一**：LLM 输出的别名/缩写先经领域词典归一为标准名，
  词典外的实体保留规范化后的原词；
- **不阻断入库**：模型不可用、超时、JSON 解析失败都只记 warning，
  降级到领域词典抽取（``fallback``）；
- **弱关系不写图**：统一走 ``filter_relations`` 的白名单/置信度/上限。

提示词要求模型返回纯 JSON；OpenAI 系走 ``json_object``，Anthropic
不接收 response_format，由提示词约束并由本地解析兜底。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from config.rag import IndexerConfig
from prompt import PromptId, PromptRegistry, get_default_prompt_registry
from rag.shared.models import Chunk, ChunkGraphExtraction, GraphRelation
from utils.text import truncate

from .domain.extractor import (
    DomainGraphExtractor,
    merge_keywords,
    merged_allowed_relations,
)
from .domain.models import DomainProfile
from .graph import (
    GENERIC_STOPWORDS,
    GraphExtractor,
    _is_valid_keyword,
    _normalize_keyword,
    filter_relations,
    graph_nearest_heading,
    graph_source_text,
)
__all__ = ["HybridGraphExtractor", "LLMGraphExtractor"]

logger = logging.getLogger(__name__)


class _LLMEntity(BaseModel):
    """模型返回的实体项；多余字段忽略，避免厂商扩展导致解析失败。"""

    model_config = ConfigDict(extra="ignore")

    name: str
    type: str | None = None


class _LLMRelation(BaseModel):
    """模型返回的关系项；confidence 缺省时由配置补默认值。"""

    model_config = ConfigDict(extra="ignore")

    subject: str
    relation: str
    object: str
    confidence: float | None = None


class _LLMResponse(BaseModel):
    """模型返回的完整 JSON 结构。"""

    model_config = ConfigDict(extra="ignore")

    entities: list[_LLMEntity] = Field(default_factory=list)
    relations: list[_LLMRelation] = Field(default_factory=list)


class LLMGraphExtractor:
    """用 ChatModel 抽取领域实体/关系；一个实例绑定一个领域画像。"""

    def __init__(
        self,
        config: IndexerConfig,
        profile: DomainProfile,
        chat_model: Any,
        *,
        model_id: str | None = None,
        fallback: GraphExtractor | None = None,
        prompts: PromptRegistry | None = None,
    ) -> None:
        """保存模型句柄、领域画像与 Prompt 注册表。"""
        self.config = config
        self.profile = profile
        self.chat_model = chat_model
        self.model_id = (
            model_id
            or config.graph_model_id
            or getattr(getattr(chat_model, "spec", None), "id", None)
            or "graph-extractor"
        )
        self.fallback = fallback
        self.prompts = prompts or get_default_prompt_registry()

    async def extract(self, chunk: Chunk) -> ChunkGraphExtraction:
        """抽取单个 chunk；失败时使用 fallback，没有 fallback 返回空结果。"""
        result = await self._extract_or_none(chunk)
        if result is not None:
            return result
        if self.fallback is not None:
            return await self.fallback.extract(chunk)
        return ChunkGraphExtraction(chunk_id=chunk.chunk_id)

    async def _extract_or_none(
        self, chunk: Chunk
    ) -> ChunkGraphExtraction | None:
        """执行一次 LLM 抽取；失败返回 None，让调用方决定降级策略。

        与 ``extract`` 分开是为了让 ``HybridGraphExtractor`` 能区分
        “模型失败”和“模型成功但没有抽到实体”。
        """
        text = (graph_source_text(chunk) or "").strip()
        if not text:
            return ChunkGraphExtraction(chunk_id=chunk.chunk_id)
        # 过短的 chunk（如单个标题）语义不足，直接交给词典/规则。
        if len(text) < self.config.graph_llm_min_chunk_chars:
            return None

        try:
            response = await self.chat_model.complete(self._build_request(chunk))
            payload = _response_text(response)
            parsed = _LLMResponse.model_validate(_loads_json(payload))
        except Exception as exc:  # noqa: BLE001 - 第三方调用/解析都不能阻断入库
            logger.warning(
                "LLM 图抽取失败，降级到词典/规则: chunk_id=%s err=%s",
                chunk.chunk_id,
                exc,
            )
            return None
        return self._to_extraction(chunk, parsed)

    # ——————————————————— Prompt 构建 ———————————————————

    def _build_request(self, chunk: Chunk):
        """组装 ChatModelRequest：模板来自 PromptRegistry，变量来自画像/配置。"""
        from provider.enums import ModelRole, ResponseFormatType
        from provider.llm.types import (
            ChatModelRequest,
            ModelGenerationOptions,
            ModelMessage,
            ModelResponseFormat,
            ModelTextPart,
        )

        text = truncate(
            chunk.content or "",
            self.config.graph_llm_max_input_chars,
        )
        heading = graph_nearest_heading(chunk)
        user_values: dict[str, Any] = {
            "block_type": chunk.block_type.value,
            "text": text,
        }
        # 标题为空时不传，使用 manifest 里的默认值“（无）”。
        if heading:
            user_values["heading"] = heading
        user_text = self.prompts.render_text(
            PromptId.RAG_GRAPH_USER, **user_values
        )
        return ChatModelRequest(
            model=self.model_id,
            messages=[
                ModelMessage(
                    role=ModelRole.SYSTEM,
                    content=[
                        ModelTextPart(text=self._render_system_prompt())
                    ],
                ),
                ModelMessage(
                    role=ModelRole.USER,
                    content=[ModelTextPart(text=user_text)],
                ),
            ],
            options=ModelGenerationOptions(
                temperature=self.config.graph_llm_temperature,
                top_p=1.0,
                max_output_tokens=self.config.graph_llm_max_output_tokens,
            ),
            response_format=ModelResponseFormat(
                type=ResponseFormatType.JSON_OBJECT
            ),
            stream=False,
        )

    def _render_system_prompt(self) -> str:
        """把领域画像/配置映射为模板变量，模板本身由 PromptRegistry 管理。

        画像未提供的角色/规则/关系清单不传，使用 manifest 的默认值；
        这样“通用兜底文案”也集中在一个可管理的位置，而不是散落在代码里。
        """
        prompt = self.profile.prompt
        values: dict[str, Any] = {
            "entity_types": "\n".join(
                self.profile.entity_type_prompt_lines()
            ),
            "max_entities": self.config.graph_max_keywords_per_chunk,
            "max_relations": self.config.graph_max_relations_per_chunk,
        }
        if prompt.role:
            values["role"] = prompt.role
        if self.profile.relations:
            values["relation_types"] = "\n".join(
                self.profile.relation_prompt_lines()
            )
        if prompt.entity_guidance:
            values["entity_guidance"] = prompt.entity_guidance
        if prompt.relation_guidance:
            values["relation_guidance"] = prompt.relation_guidance
        if prompt.extra_instructions:
            values["extra_instructions"] = "\n".join(
                prompt.extra_instructions
            )
        return self.prompts.render_text(
            PromptId.RAG_GRAPH_SYSTEM, **values
        )

    # ——————————————————— 结果归一 ———————————————————

    def _to_extraction(
        self,
        chunk: Chunk,
        parsed: _LLMResponse,
    ) -> ChunkGraphExtraction:
        """把 LLM 结果归一为标准实体名，并通过关系白名单过滤。"""
        max_keywords = self.config.graph_max_keywords_per_chunk
        keywords: list[str] = []
        seen: set[str] = set()
        for entity in parsed.entities:
            name = (
                self.profile.canonicalize(entity.name)
                or _normalize_keyword(entity.name)
            )
            if not _is_valid_entity_name(
                name, GENERIC_STOPWORDS
            ) or self.profile.is_stopword(name):
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            keywords.append(name)
            if len(keywords) >= max_keywords:
                break

        relations: list[GraphRelation] = []
        for item in parsed.relations:
            subject = (
                self.profile.canonicalize(item.subject)
                or _normalize_keyword(item.subject)
            )
            obj = (
                self.profile.canonicalize(item.object)
                or _normalize_keyword(item.object)
            )
            relation = (item.relation or "").strip().lower()
            if not _is_valid_entity_name(
                subject, GENERIC_STOPWORDS
            ) or not _is_valid_entity_name(
                obj, GENERIC_STOPWORDS
            ):
                continue
            if self.profile.is_stopword(subject) or self.profile.is_stopword(obj):
                continue
            if not relation:
                continue
            if subject.lower() == obj.lower():
                continue
            confidence = (
                item.confidence
                if item.confidence is not None
                else self.config.graph_llm_default_confidence
            )
            relations.append(
                GraphRelation(
                    subject=subject,
                    relation=relation,
                    object=obj,
                    # 模型偶尔给出越界分数；入库前夹紧，避免污染阈值判断。
                    confidence=max(0.0, min(1.0, float(confidence))),
                )
            )

        filtered = filter_relations(
            relations,
            self.config,
            allowed_relations=merged_allowed_relations(
                self.config, self.profile
            ),
        )
        return ChunkGraphExtraction(
            chunk_id=chunk.chunk_id,
            keywords=keywords,
            relations=filtered,
            model=self.model_id,
            # 有实体/关系即视为模型完成了一次有效抽取。
            confidence=0.9 if (keywords or filtered) else None,
        )


class HybridGraphExtractor:
    """词典与 LLM 合并：兼顾领域精度与语义泛化。

    - 关键词：词典命中优先，LLM 实体补充；
    - 关系：只用 LLM（词典模式默认不抽关系）；
    - LLM 失败：退回词典结果。
    """

    def __init__(
        self,
        config: IndexerConfig,
        profile: DomainProfile,
        llm: LLMGraphExtractor,
        lexicon: DomainGraphExtractor,
    ) -> None:
        """保存两个子抽取器与配置。"""
        self.config = config
        self.profile = profile
        self.llm = llm
        self.lexicon = lexicon

    async def extract(self, chunk: Chunk) -> ChunkGraphExtraction:
        """先跑词典，再合并 LLM 结果；任一失败都不影响入库。"""
        lexicon_result = await self.lexicon.extract(chunk)
        llm_result = await self.llm._extract_or_none(chunk)
        if llm_result is None:
            return lexicon_result
        keywords = merge_keywords(
            lexicon_result.keywords,
            llm_result.keywords,
            top_k=self.config.graph_max_keywords_per_chunk,
        )
        return ChunkGraphExtraction(
            chunk_id=chunk.chunk_id,
            keywords=keywords,
            # 词典模式默认不产生关系；LLM 没有关系时才保留词典结果。
            relations=llm_result.relations or lexicon_result.relations,
            model=llm_result.model,
            confidence=llm_result.confidence,
        )


def _is_valid_entity_name(
    value: str,
    stopwords: Iterable[str] = (),
) -> bool:
    """实体/关系端点的长度与内容校验；防止模型输出整句或超长噪声。"""
    return (
        bool(value)
        and len(value) <= 64
        and _is_valid_keyword(value, stopwords=stopwords)
    )


def _response_text(response: Any) -> str:
    """从 ChatModelResponse 中取文本 part；忽略 thinking/tool parts。"""
    from provider.llm.types import ModelTextPart

    return "".join(
        part.text
        for part in response.message.content
        if isinstance(part, ModelTextPart)
    ).strip()


def _loads_json(text: str) -> dict[str, Any]:
    """解析模型输出：去掉 Markdown 代码块，截取最外层 JSON 对象。"""
    value = (text or "").strip()
    if value.startswith("```"):
        # 允许 ```json ... ``` 或 ``` ... ```。
        lines = value.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    start = value.find("{")
    end = value.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("模型输出不包含 JSON 对象")
    payload = json.loads(value[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("模型输出 JSON 必须是对象")
    return payload
