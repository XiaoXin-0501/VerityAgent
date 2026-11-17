"""LLM 图抽取器：领域画像只做提示与归一，实体/关系必须由模型产出。

设计原则：

- **LLM 是唯一抽取器**：不再保留规则/词典/混合模式，也不把词典命中直接
  写成图节点；词典只用于提示词参考、别名归一和查询期 seed 匹配；
- **Prompt 由注册表管理**：system/user 模板在 ``prompt/templates/rag``，
  本类只负责组装变量（类型、补充词 reference_terms、黑名单、关系白名单、上限）；
- **按 chunk 批量抽取**：一次请求携带多个 chunk，按 ``chunk_id`` 返回结果，
  降低调用次数；解析类错误拆分重试，API 故障返回空结果不阻断入库；
- **并发受控**：同一文档的多个批次通过信号量限制并发；
- **chunk 级缓存**：key = 模型 + Prompt 版本/checksum + 画像内容签名 + 正文，
  重试/REINDEX/重复上传直接复用；
- **类型必须落库**：每个实体都带已声明 ``type``，写入 Neo4j ``Keyword.type``；
  模型给出未声明类型时该实体不写入图，只记录聚合日志；
- **弱关系不写图**：关系类型只能取画像声明，再走置信度/数量上限过滤。

提示词要求模型返回纯 JSON；OpenAI 系走 ``json_object``，Anthropic
不接收 response_format，由提示词约束并由本地解析兜底。
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from config.rag import IndexerConfig
from provider.enums import ChatFinishReason
from prompt import PromptId, PromptRegistry, get_default_prompt_registry
from rag.shared.models import (
    Chunk,
    ChunkGraphExtraction,
    GraphKeyword,
    GraphRelation,
)
from utils.hashing import short_digest
from utils.text import truncate

from ...caching import ChunkCache, graph_extraction_cache_key
from ..domain.models import DomainProfile
from .base import (
    GENERIC_KEYWORD_BLACKLIST,
    filter_relations,
    graph_nearest_heading,
    graph_source_text,
    is_valid_keyword,
    normalize_blocked_terms,
    normalize_keyword,
)

__all__ = ["LLMGraphExtractor"]

logger = logging.getLogger(__name__)

# 文本中实际命中的补充词（标准名/类型）；随 chunk JSON 注入，仅辅助归一。
_ReferenceTerms = list[dict[str, str]]
# (chunk, 截断后的正文, 补充词, 缓存 key)
_PendingChunk = tuple[Chunk, str, _ReferenceTerms, Any | None]


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


class _LLMResultItem(BaseModel):
    """单个 chunk 的 LLM 抽取结果。"""

    model_config = ConfigDict(extra="ignore")

    chunk_id: str = ""
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
        prompts: PromptRegistry | None = None,
        cache: ChunkCache | None = None,
    ) -> None:
        """保存模型句柄、领域画像、Prompt 注册表与可选缓存。"""
        self.config = config
        self.profile = profile
        self.chat_model = chat_model
        self.model_id = (
            model_id
            or config.graph_model_id
            or getattr(getattr(chat_model, "spec", None), "id", None)
            or "graph-extractor"
        )
        self.prompts = prompts or get_default_prompt_registry()
        self.cache = cache
        # 通用黑名单 + 运维黑名单 + 领域画像黑名单；命中即丢弃。
        self.prompt_blacklist = (
            normalize_blocked_terms(config.graph_keyword_blacklist)
            | profile.blacklist_keys
        )
        self.blacklist = GENERIC_KEYWORD_BLACKLIST | self.prompt_blacklist
        self.allowed_relations = frozenset(
            item.lower() for item in profile.relation_names
        )

    async def extract(self, chunk: Chunk) -> ChunkGraphExtraction:
        """抽取单个 chunk；内部复用批量接口。"""
        result = await self.extract_batch([chunk])
        return result[chunk.chunk_id]

    async def extract_batch(
        self,
        chunks: list[Chunk],
        *,
        max_concurrency: int | None = None,
    ) -> dict[str, ChunkGraphExtraction]:
        """批量抽取：缓存命中优先，未命中按 batch 分组并发调用模型。"""
        if not chunks:
            return {}
        results: dict[str, ChunkGraphExtraction] = {}
        pending: list[_PendingChunk] = []
        # 与 _make_batches 使用同一截断上限，确保 reference_terms 和缓存 key
        # 都基于实际送入模型的正文。
        prompt_limit = min(
            max(200, int(self.config.graph_llm_max_input_chars)),
            max(500, int(self.config.graph_llm_batch_max_chars)),
        )
        for chunk in chunks:
            text = (graph_source_text(chunk) or "").strip()
            if not text or len(text) < self.config.graph_llm_min_chunk_chars:
                results[chunk.chunk_id] = ChunkGraphExtraction(
                    chunk_id=chunk.chunk_id
                )
                continue
            prompt_text = truncate(text, prompt_limit)
            # 只把当前 chunk 文本里实际出现的补充词带给模型；白名单不是
            # 抽取边界，也不要求穷举领域。
            reference_terms = self.profile.reference_terms(
                prompt_text,
                limit=self.config.graph_llm_reference_terms_per_chunk,
            )
            key = self._cache_key(prompt_text, reference_terms)
            cached = await self._cache_get(key)
            if cached is not None:
                results[chunk.chunk_id] = cached
                continue
            pending.append((chunk, prompt_text, reference_terms, key))
        if not pending:
            return results

        batches = list(self._make_batches(pending))
        concurrency = max(
            1,
            int(max_concurrency or self.config.graph_llm_concurrency),
        )
        semaphore = asyncio.Semaphore(concurrency)

        async def run(
            batch: list[_PendingChunk],
        ) -> dict[str, ChunkGraphExtraction]:
            """在并发信号量内执行一个批次。"""
            async with semaphore:
                return await self._extract_batch_items(batch)

        outcomes = await asyncio.gather(
            *(run(batch) for batch in batches),
            return_exceptions=True,
        )
        for batch, outcome in zip(batches, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                logger.warning(
                    "LLM 图抽取批次失败: chunks=%d err=%s",
                    len(batch),
                    outcome,
                )
                outcome = {}
            for chunk, _text, _reference_terms, key in batch:
                extraction = outcome.get(chunk.chunk_id)
                if extraction is None:
                    # 图构建是入库增强项：模型故障时返回空结果，不阻断入库。
                    extraction = ChunkGraphExtraction(chunk_id=chunk.chunk_id)
                elif key is not None:
                    await self._cache_set(key, extraction)
                results[chunk.chunk_id] = extraction
        return results

    # ——————————————————— 批次与缓存 ———————————————————

    def _make_batches(
        self,
        pending: list[_PendingChunk],
    ) -> Iterable[list[_PendingChunk]]:
        """按 batch_size 与总字符上限切批；正文已在入批前按上限截断。"""
        batch_size = max(1, int(self.config.graph_llm_batch_size))
        max_chars = max(500, int(self.config.graph_llm_batch_max_chars))
        batch: list[_PendingChunk] = []
        chars = 0
        for chunk, prompt_text, reference_terms, key in pending:
            if batch and (
                len(batch) >= batch_size
                or chars + len(prompt_text) > max_chars
            ):
                yield batch
                batch = []
                chars = 0
            batch.append((chunk, prompt_text, reference_terms, key))
            chars += len(prompt_text)
            if len(batch) >= batch_size:
                yield batch
                batch = []
                chars = 0
        if batch:
            yield batch

    def _cache_key(self, text: str, reference_terms: _ReferenceTerms):
        """构造 chunk 级图抽取缓存 key；未启用/未注入缓存时返回 None。"""
        if (
            not self.config.graph_llm_cache_enabled
            or self.config.graph_llm_cache_ttl_s <= 0
            or self.cache is None
        ):
            return None
        system = self.prompts.get(PromptId.RAG_GRAPH_SYSTEM)
        user = self.prompts.get(PromptId.RAG_GRAPH_USER)
        # 影响输出的运行配置也进入 key：改实体/关系上限、置信度阈值、
        # 补充词数量或运维黑名单后，旧结果不再复用。
        reference_digest = short_digest(
            json.dumps(reference_terms, ensure_ascii=False, sort_keys=True)
        )
        extraction_config = "|".join(
            [
                f"reference_terms={reference_digest}",
                f"max_entities={self.config.graph_max_keywords_per_chunk}",
                f"max_relations={self.config.graph_max_relations_per_chunk}",
                f"min_confidence={self.config.graph_min_confidence}",
                f"default_confidence={self.config.graph_llm_default_confidence}",
                f"temperature={self.config.graph_llm_temperature}",
                f"max_output_tokens={self.config.graph_llm_max_output_tokens}",
                f"blacklist={short_digest('|'.join(sorted(self.blacklist)))}",
            ]
        )
        return graph_extraction_cache_key(
            model_id=self.model_id,
            prompt_version=user.version,
            prompt_checksum=f"{system.checksum}:{user.checksum}",
            # 画像签名覆盖类型/白名单/黑名单/关系/提示词；换文档或改词典
            # 后旧缓存自然失效，不依赖人工 bump version。
            domain_profile=(
                f"{self.profile.name}:{self.profile.version}:"
                f"{self.profile.signature}:{extraction_config}"
            ),
            text=text,
        )

    async def _cache_get(self, key: Any) -> ChunkGraphExtraction | None:
        """读取缓存；非法值按 miss 处理。"""
        if key is None or self.cache is None:
            return None
        try:
            value = await self.cache.get(key)
        except Exception as exc:  # noqa: BLE001 - 缓存失败不能影响入库
            logger.debug("读取图抽取缓存失败: %s", exc)
            return None
        if value is None:
            return None
        try:
            return ChunkGraphExtraction.model_validate(value)
        except Exception:  # noqa: BLE001 - 旧格式/损坏值按 miss
            return None

    async def _cache_set(self, key: Any, extraction: ChunkGraphExtraction) -> None:
        """写入缓存；失败只记录，不影响本次结果。"""
        if key is None or self.cache is None:
            return
        ttl = int(self.config.graph_llm_cache_ttl_s)
        if ttl <= 0:
            return
        try:
            await self.cache.set(
                key,
                extraction.model_dump(mode="json"),
                ttl_s=ttl,
            )
        except Exception as exc:  # noqa: BLE001 - 缓存失败不能影响入库
            logger.debug("写入图抽取缓存失败: %s", exc)

    # ——————————————————— 模型调用 ———————————————————

    async def _extract_batch_items(
        self,
        batch: list[_PendingChunk],
    ) -> dict[str, ChunkGraphExtraction]:
        """执行一个批次；截断/解析类错误拆分重试，API 故障返回空结果。"""
        chunks = [item[0] for item in batch]
        texts = [item[1] for item in batch]
        reference_terms = [item[2] for item in batch]
        try:
            response = await self.chat_model.complete(
                self._build_request(chunks, texts, reference_terms)
            )
            if response.finish_reason is ChatFinishReason.LENGTH:
                logger.warning(
                    "LLM 图抽取输出达到 max_tokens 被截断: chunks=%d "
                    "output_tokens=%s",
                    len(batch),
                    getattr(response.usage, "output_tokens", None),
                )
                if len(batch) > 1:
                    return await self._split_retry(batch)
                return {}
            payload = _loads_json(_response_text(response))
            parsed = _parse_batch(payload, chunks)
            if len(parsed) < len(chunks):
                # 协议要求每个 chunk 返回一条 results（可为空）；漏项说明
                # 批次输出不完整，拆小重试，避免静默丢图节点。
                logger.warning(
                    "LLM 图抽取结果缺少 chunk: expected=%d got=%d",
                    len(chunks),
                    len(parsed),
                )
                if len(batch) > 1:
                    return await self._split_retry(batch)
                return {}
        except ValueError as exc:
            # JSON/结构不合法：批量响应可能被截断或模型漏了 chunk_id，
            # 拆分后通常能恢复；单 chunk 仍失败则返回空结果。
            logger.warning(
                "LLM 图抽取批次解析失败: chunks=%d err=%s",
                len(batch),
                exc,
            )
            if len(batch) > 1:
                return await self._split_retry(batch)
            return {}
        except Exception as exc:  # noqa: BLE001 - 第三方调用异常统一降级
            logger.warning(
                "LLM 图抽取失败，返回空结果: chunks=%d err=%s",
                len(batch),
                exc,
            )
            return {}

        result: dict[str, ChunkGraphExtraction] = {}
        for chunk in chunks:
            item = parsed.get(chunk.chunk_id)
            if item is None:
                result[chunk.chunk_id] = ChunkGraphExtraction(
                    chunk_id=chunk.chunk_id
                )
            else:
                result[chunk.chunk_id] = self._to_extraction(chunk, item)
        return result

    async def _split_retry(
        self,
        batch: list[_PendingChunk],
    ) -> dict[str, ChunkGraphExtraction]:
        """把一个失败批次对半拆分重试；递归深度受 batch_size 限制。"""
        middle = len(batch) // 2
        left = await self._extract_batch_items(batch[:middle])
        right = await self._extract_batch_items(batch[middle:])
        return {**left, **right}

    # ——————————————————— Prompt 构建 ———————————————————

    def _build_request(
        self,
        chunks: list[Chunk],
        texts: list[str],
        reference_terms: list[_ReferenceTerms],
    ):
        """组装批量 ChatModelRequest：模板来自 PromptRegistry。"""
        from provider.enums import ModelRole, ResponseFormatType
        from provider.llm.types import (
            ChatModelRequest,
            ModelGenerationOptions,
            ModelMessage,
            ModelResponseFormat,
            ModelTextPart,
        )

        entries = [
            {
                "chunk_id": chunk.chunk_id,
                "heading": graph_nearest_heading(chunk),
                "block_type": chunk.block_type.value,
                "text": text,
                # 只包含该 chunk 文本中出现的补充词；空列表也保留字段，
                # 让模型明确“没有补充词”而不是漏看协议。
                "reference_terms": terms,
            }
            for chunk, text, terms in zip(
                chunks, texts, reference_terms, strict=True
            )
        ]
        user_text = self.prompts.render_text(
            PromptId.RAG_GRAPH_USER,
            chunks_json=json.dumps(entries, ensure_ascii=False),
        )
        return ChatModelRequest(
            model=self.model_id,
            messages=[
                ModelMessage(
                    role=ModelRole.SYSTEM,
                    content=[ModelTextPart(text=self._render_system_prompt())],
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
        """把领域画像/配置映射为模板变量，模板本身由 PromptRegistry 管理。"""
        prompt = self.profile.prompt
        values: dict[str, Any] = {
            "entity_types": "\n".join(self.profile.entity_type_prompt_lines()),
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
        if prompt.entity_examples:
            values["entity_examples"] = "\n".join(prompt.entity_examples)
        if prompt.noise_examples:
            values["noise_examples"] = "\n".join(prompt.noise_examples)
        values["blacklist"] = (
            # 通用黑名单仍参与结果过滤，但不全部塞进 Prompt，避免过度限制
            # 模型对正文概念的抽取；Prompt 只展示领域/运维明确禁止的词。
            "\n".join(f"- {item}" for item in sorted(self.prompt_blacklist))
            or "（无）"
        )
        return self.prompts.render_text(PromptId.RAG_GRAPH_SYSTEM, **values)

    # ——————————————————— 结果归一 ———————————————————

    def _to_extraction(
        self,
        chunk: Chunk,
        parsed: _LLMResultItem,
    ) -> ChunkGraphExtraction:
        """把单个 chunk 的 LLM 结果归一为标准名、补类型并过滤关系。"""
        max_entities = self.config.graph_max_keywords_per_chunk
        all_entities: list[GraphKeyword] = []
        types: dict[str, str] = {}
        seen: set[str] = set()
        unknown_types: list[tuple[str, str]] = []
        for entity in parsed.entities:
            raw = normalize_keyword(entity.name)
            if not raw:
                continue
            canonical = self.profile.canonicalize_entity(raw)
            name = canonical[0] if canonical else raw
            if not _is_valid_entity_name(name, blacklist=self.blacklist):
                continue
            if self.profile.is_blacklisted(name):
                continue
            if canonical is not None:
                entity_type = canonical[1]
            else:
                # type 不限制 LLM 自由给出，但只有画像已声明的类型才能入图；
                # 未声明类型只记录日志，便于后续补画像，不写入 Neo4j。
                resolved = self.profile.resolve_entity_type(entity.type or "")
                if resolved is None:
                    unknown_types.append(
                        (name, (entity.type or "").strip() or "<empty>")
                    )
                    continue
                entity_type = resolved
            key = name.casefold()
            if key in seen:
                continue
            seen.add(key)
            types[key] = entity_type
            all_entities.append(GraphKeyword(name=name, type=entity_type))

        # 先按全部实体构造候选关系，再决定实体截断：关系端点优先保留，
        # 避免第 11 个实体恰好是关系宾语时把关系一起丢掉。
        candidates: list[GraphRelation] = []
        required: set[str] = set()
        unknown_relations: list[tuple[str, str, str]] = []
        for item in parsed.relations:
            subject = self._canonical_name(item.subject)
            obj = self._canonical_name(item.object)
            if not subject or not obj:
                continue
            subject_key = subject.casefold()
            object_key = obj.casefold()
            if subject_key not in types or object_key not in types:
                continue
            if self.profile.is_blacklisted(subject) or self.profile.is_blacklisted(
                obj
            ):
                continue
            relation = (item.relation or "").strip().lower()
            if not relation or subject_key == object_key:
                continue
            if relation not in self.allowed_relations:
                # 关系类型同理：未声明关系暂不入图，只记录日志。
                unknown_relations.append((subject, relation, obj))
                continue
            confidence = (
                item.confidence
                if item.confidence is not None
                else self.config.graph_llm_default_confidence
            )
            required.update((subject_key, object_key))
            candidates.append(
                GraphRelation(
                    subject=subject,
                    relation=relation,
                    object=obj,
                    confidence=max(0.0, min(1.0, float(confidence))),
                    subject_type=types[subject_key],
                    object_type=types[object_key],
                )
            )

        # 未声明类型/关系只做聚合日志：避免每个实体/关系一行把日志刷乱。
        if unknown_types:
            logger.info(
                "图抽取跳过未声明类型: chunk=%s count=%d types=%s samples=%s",
                chunk.chunk_id,
                len(unknown_types),
                sorted({item[1] for item in unknown_types}),
                [item[0] for item in unknown_types[:5]],
            )
        if unknown_relations:
            logger.info(
                "图抽取跳过未声明关系: chunk=%s count=%d relations=%s "
                "samples=%s",
                chunk.chunk_id,
                len(unknown_relations),
                sorted({item[1] for item in unknown_relations}),
                [
                    f"{item[0]}-{item[1]}->{item[2]}"
                    for item in unknown_relations[:5]
                ],
            )

        # 关系端点优先，其余实体按 LLM 输出顺序补足到上限。
        entities = [
            item for item in all_entities if item.name.casefold() in required
        ]
        if len(entities) < max_entities:
            entities.extend(
                item
                for item in all_entities
                if item.name.casefold() not in required
            )
        entities = entities[:max_entities]
        kept = {item.name.casefold() for item in entities}
        relations = [
            item
            for item in candidates
            if item.subject.casefold() in kept and item.object.casefold() in kept
        ]

        filtered = filter_relations(
            relations,
            self.config,
            allowed_relations=self.allowed_relations,
        )
        return ChunkGraphExtraction(
            chunk_id=chunk.chunk_id,
            keywords=entities,
            relations=filtered,
            model=self.model_id,
            confidence=0.9 if (entities or filtered) else None,
        )

    def _canonical_name(self, value: str) -> str:
        """把 LLM 返回的实体名归一为标准名；无效名返回空串。"""
        raw = normalize_keyword(value)
        if not raw:
            return ""
        canonical = self.profile.canonicalize(raw)
        name = canonical or raw
        if not _is_valid_entity_name(name, blacklist=self.blacklist):
            return ""
        return name


def _parse_batch(
    payload: dict[str, Any],
    chunks: list[Chunk],
) -> dict[str, _LLMResultItem]:
    """解析批量响应；兼容模型返回单对象格式的旧行为。"""
    results = payload.get("results")
    if isinstance(results, list):
        parsed: dict[str, _LLMResultItem] = {}
        for index, item in enumerate(results):
            if not isinstance(item, dict):
                continue
            try:
                result = _LLMResultItem.model_validate(item)
            except Exception:  # noqa: BLE001 - 单个坏项不拖垮整批
                continue
            chunk_id = result.chunk_id
            if not chunk_id and index < len(chunks) and len(results) == len(chunks):
                chunk_id = chunks[index].chunk_id
            if chunk_id:
                parsed[chunk_id] = result
        return parsed
    if "entities" in payload or "relations" in payload:
        if len(chunks) != 1:
            raise ValueError("批量请求返回了单对象格式")
        result = _LLMResultItem.model_validate(
            {**payload, "chunk_id": chunks[0].chunk_id}
        )
        return {chunks[0].chunk_id: result}
    raise ValueError("模型输出缺少 results")


def _is_valid_entity_name(
    value: str,
    *,
    blacklist: Iterable[str] = (),
) -> bool:
    """实体/关系端点的长度与内容校验；防止模型输出整句或超长噪声。"""
    return (
        bool(value)
        and len(value) <= 64
        and is_valid_keyword(value, blacklist=blacklist)
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
