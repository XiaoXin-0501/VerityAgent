"""LLM 图抽取器：领域画像只做提示与归一，实体/关系必须由模型产出。

设计原则：

- **LLM 是唯一抽取器**：不再保留规则/词典/混合模式，也不把词典命中直接
  写成图节点；词典只用于提示词参考、别名归一和查询期 seed 匹配；
- **Prompt 由注册表管理**：system/user 模板在 ``prompt/templates/rag``，
  本类只负责组装变量（有效类型/关系、别名标准名、上限）；类型/关系黑名单
  与别名值只在抽取后过滤/归一阶段使用，不进入 Prompt；
- **按 chunk 批量抽取**：一次请求携带多个 chunk，按 ``chunk_id`` 返回结果，
  降低调用次数；解析类错误拆分重试，API 故障返回空结果不阻断入库；
- **并发受控**：同一文档的多个批次通过信号量限制并发；
- **chunk 级缓存**：key = 模型 + Prompt 版本/checksum + 画像内容签名 + 正文，
  重试/REINDEX/重复上传直接复用；
- **类型尽量落库**：每个实体都带类型，写入 Neo4j ``Keyword.type``；
  模型给出未声明类型时不直接丢弃，而是保留最贴近的合法类型并记录指标，
  避免画像枚举不完整导致实体丢失；
- **关系声明辅助**：默认关系 + 关系白名单 - 关系黑名单作为抽取提示；
  未声明关系记录指标，按配置保留证据；图下一跳遍历所有入库合法关系，
  不再按画像声明集合限制。

提示词要求模型返回纯 JSON；OpenAI 系走 ``json_object``，Anthropic
不接收 response_format，由提示词约束并由本地解析兜底。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from config.domain import DomainProfile
from config.rag import IndexerConfig
from prompt import PromptId, PromptRegistry, get_default_prompt_registry
from provider.enums import ChatFinishReason
from provider.llm.helpers import response_text
from rag.shared.events import (
    RagEvent,
    RagEventName,
    RagEventSink,
    emit_rag_event,
)
from rag.shared.models import (
    Chunk,
    ChunkGraphExtraction,
    GraphKeyword,
    GraphRelation,
)
from utils.hashing import short_digest
from utils.serialization import parse_json_object
from utils.text import truncate
from utils.tokens import estimate_tokens
from .base import (
    GENERIC_KEYWORD_BLACKLIST,
    filter_relations,
    graph_nearest_heading,
    graph_source_text,
    is_valid_keyword,
    normalize_blocked_terms,
    normalize_keyword,
)
from ..caching import ChunkCache, graph_extraction_cache_key

__all__ = ["LLMGraphExtractor"]

logger = logging.getLogger(__name__)

# (chunk, 截断后的正文, 缓存 key)
_PendingChunk = tuple[Chunk, str, Any | None]


def _new_metrics(chunks_total: int) -> dict[str, Any]:
    """初始化一次批量抽取的指标；只作为事件 payload，不承担日志格式化。"""
    return {
        "chunks_total": chunks_total,
        "chunks_skipped": 0,
        "cache_hits": 0,
        "batches": 0,
        "llm_calls": 0,
        "truncations": 0,
        "retries": 0,
        "failed_batches": 0,
        "input_truncated_chunks": 0,
        "entities": 0,
        "relations": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "unknown_types": 0,
        "unknown_types_kept": 0,
        "unknown_relations": 0,
        "unknown_relations_kept": 0,
        "unknown_type_samples": [],
        "unknown_relation_samples": [],
    }


def _add_unknown(
        metrics: dict[str, Any],
        *,
        count_key: str,
        sample_key: str,
        sample: str,
) -> None:
    """累计未声明类型/关系；只保留前 5 个样例行。"""
    metrics[count_key] += 1
    samples = metrics[sample_key]
    if len(samples) < 5 and sample not in samples:
        samples.append(sample)


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
            event_sink: RagEventSink | None = None,
    ) -> None:
        """保存模型句柄、领域画像、Prompt 注册表、缓存与可选事件接收端。"""
        self.config = config
        self.profile = profile
        self.chat_model = chat_model
        self.event_sink = event_sink
        self.model_id = (
                model_id
                or config.graph_model_id
                or getattr(getattr(chat_model, "spec", None), "id", None)
                or "graph-extractor"
        )
        self.prompts = prompts or get_default_prompt_registry()
        self.cache = cache
        # 通用噪声词 + 运维噪声词 + 画像 noise_terms；命中即丢弃。
        self.domain_noise_terms = (
                normalize_blocked_terms(config.graph_noise_terms)
                | profile.noise_keys
        )
        self.blocked_terms = GENERIC_KEYWORD_BLACKLIST | self.domain_noise_terms
        self.declared_relations = frozenset(
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
        started = time.perf_counter()
        metrics = _new_metrics(len(chunks))
        results: dict[str, ChunkGraphExtraction] = {}
        pending: list[_PendingChunk] = []
        # 单 chunk 只做字符级截断；批次大小由 _make_batches 按 token 预算决定。
        prompt_limit = max(200, int(self.config.graph_llm_max_input_chars))
        for chunk in chunks:
            text = (graph_source_text(chunk) or "").strip()
            if not text or len(text) < self.config.graph_llm_min_chunk_chars:
                metrics["chunks_skipped"] += 1
                results[chunk.chunk_id] = ChunkGraphExtraction(
                    chunk_id=chunk.chunk_id
                )
                continue
            if len(text) > prompt_limit:
                metrics["input_truncated_chunks"] += 1
            prompt_text = truncate(text, prompt_limit)
            key = self._cache_key(prompt_text)
            cached = await self._cache_get(key)
            if cached is not None:
                metrics["cache_hits"] += 1
                metrics["entities"] += len(cached.keywords)
                metrics["relations"] += len(cached.relations)
                # 缓存 key 只含正文/画像/Prompt；同一正文出现在不同 chunk 时
                # 必须把结果 id 归到当前 chunk，避免下游拿到别的 chunk_id。
                results[chunk.chunk_id] = cached.model_copy(
                    update={"chunk_id": chunk.chunk_id}
                )
                continue
            pending.append((chunk, prompt_text, key))
        if not pending:
            await self._emit_summary(metrics, started)
            return results

        batches = list(self._make_batches(pending))
        metrics["batches"] = len(batches)
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
                return await self._extract_batch_items(batch, metrics)

        outcomes = await asyncio.gather(
            *(run(batch) for batch in batches),
            return_exceptions=True,
        )
        for batch, outcome in zip(batches, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                metrics["failed_batches"] += 1
                logger.warning(
                    "LLM 图抽取批次失败: chunks=%d err=%s",
                    len(batch),
                    outcome,
                )
                outcome = {}
            for chunk, _text, key in batch:
                extraction = outcome.get(chunk.chunk_id)
                if extraction is None:
                    # 图构建是入库增强项：模型故障时返回空结果，不阻断入库。
                    extraction = ChunkGraphExtraction(chunk_id=chunk.chunk_id)
                elif key is not None:
                    await self._cache_set(key, extraction)
                results[chunk.chunk_id] = extraction
        await self._emit_summary(metrics, started)
        return results

    async def _emit_summary(
            self,
            metrics: dict[str, Any],
            started: float,
    ) -> None:
        """把运行指标作为结构化事件交给上层，不在 RAG 内拼日志。"""
        duration_ms = int((time.perf_counter() - started) * 1000)
        # 未声明类型/关系必须可追溯：既写结构化事件，也写一条聚合 warning。
        # 开关开启时这些条目会入库；关闭时只记录被丢弃的候选。
        if metrics["unknown_types"] or metrics["unknown_relations"]:
            logger.warning(
                "图抽取发现未声明类型/关系: types=%d(kept=%d) "
                "relations=%d(kept=%d) type_samples=%s relation_samples=%s",
                metrics["unknown_types"],
                metrics["unknown_types_kept"],
                metrics["unknown_relations"],
                metrics["unknown_relations_kept"],
                metrics["unknown_type_samples"],
                metrics["unknown_relation_samples"],
            )
        payload = {
            **metrics,
            "model_id": self.model_id,
            "duration_ms": duration_ms,
        }
        await emit_rag_event(
            self.event_sink,
            RagEvent(
                name=RagEventName.GRAPH_EXTRACTION_END,
                payload=payload,
                duration_ms=duration_ms,
            ),
        )

    # ——————————————————— 批次与缓存 ———————————————————

    def _make_batches(
            self,
            pending: list[_PendingChunk],
    ) -> Iterable[list[_PendingChunk]]:
        """按输入 token 预算装批；chunk 是最小单位，不拆 chunk 填缝。

        单 chunk 超过预算时仍单独成批（必要时由单 chunk 字符上限截断），
        保证不会因为预算把 chunk 丢掉。
        """
        max_tokens = max(500, int(self.config.graph_llm_batch_max_input_tokens))
        batch: list[_PendingChunk] = []
        batch_tokens = 0
        for item in pending:
            item_tokens = estimate_tokens(item[1])
            if batch and batch_tokens + item_tokens > max_tokens:
                yield batch
                batch = []
                batch_tokens = 0
            batch.append(item)
            batch_tokens += item_tokens
        if batch:
            yield batch

    def _cache_key(self, text: str):
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
        # 别名提示上限或未知类型/关系策略后，旧结果不再复用。
        extraction_config = "|".join(
            [
                f"alias_limit={self.config.graph_llm_alias_prompt_limit}",
                f"max_entities={self.config.graph_max_keywords_per_chunk}",
                f"max_relations={self.config.graph_max_relations_per_chunk}",
                f"min_confidence={self.config.graph_min_confidence}",
                f"default_confidence={self.config.graph_llm_default_confidence}",
                f"temperature={self.config.graph_llm_temperature}",
                f"max_output_tokens={self.config.graph_llm_max_output_tokens}",
                f"unknown_type={self.config.graph_unknown_entity_type}",
                f"allow_unknown_types={self.config.graph_allow_unknown_types}",
                f"unknown_relations={self.config.graph_allow_unknown_relations}",
                f"noise_terms={short_digest('|'.join(sorted(self.blocked_terms)))}",
            ]
        )
        return graph_extraction_cache_key(
            model_id=self.model_id,
            prompt_version=user.version,
            prompt_checksum=f"{system.checksum}:{user.checksum}",
            # 画像签名覆盖类型/关系/白名单/黑名单/别名/提示词；换文档或改画像
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
            metrics: dict[str, Any],
    ) -> dict[str, ChunkGraphExtraction]:
        """执行一个批次；截断/解析类错误拆分重试，API 故障返回空结果。"""
        metrics["llm_calls"] += 1
        chunks = [item[0] for item in batch]
        texts = [item[1] for item in batch]
        try:
            response = await self.chat_model.complete(
                self._build_request(chunks, texts)
            )
            metrics["input_tokens"] += int(
                getattr(response.usage, "input_tokens", 0) or 0
            )
            metrics["output_tokens"] += int(
                getattr(response.usage, "output_tokens", 0) or 0
            )
            if response.finish_reason is ChatFinishReason.LENGTH:
                metrics["truncations"] += 1
                logger.warning(
                    "LLM 图抽取输出达到 max_tokens 被截断: chunks=%d "
                    "output_tokens=%s",
                    len(batch),
                    getattr(response.usage, "output_tokens", None),
                )
                if len(batch) > 1:
                    return await self._split_retry(batch, metrics)
                metrics["failed_batches"] += 1
                return {}
            payload = parse_json_object(response_text(response))
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
                    return await self._split_retry(batch, metrics)
                metrics["failed_batches"] += 1
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
                return await self._split_retry(batch, metrics)
            metrics["failed_batches"] += 1
            return {}
        except Exception as exc:  # noqa: BLE001 - 第三方调用异常统一降级
            metrics["failed_batches"] += 1
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
                extraction = self._to_extraction(chunk, item, metrics)
                metrics["entities"] += len(extraction.keywords)
                metrics["relations"] += len(extraction.relations)
                result[chunk.chunk_id] = extraction
        return result

    async def _split_retry(
            self,
            batch: list[_PendingChunk],
            metrics: dict[str, Any],
    ) -> dict[str, ChunkGraphExtraction]:
        """把一个失败批次对半拆分重试；递归深度受 batch_size 限制。"""
        metrics["retries"] += 1
        middle = len(batch) // 2
        left = await self._extract_batch_items(batch[:middle], metrics)
        right = await self._extract_batch_items(batch[middle:], metrics)
        return {**left, **right}

    # ——————————————————— Prompt 构建 ———————————————————

    def _build_request(
            self,
            chunks: list[Chunk],
            texts: list[str],
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
            }
            for chunk, text in zip(chunks, texts, strict=True)
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
        # aliases 只给标准名，不带别名值；它帮助模型优先输出领域标准名，
        # 但不是抽取边界。类型/关系黑名单只在抽取后过滤，不进入 Prompt。
        values["alias_names"] = (
                "\n".join(
                    self.profile.alias_prompt_lines(
                        limit=self.config.graph_llm_alias_prompt_limit
                    )
                )
                or "（无）"
        )
        return self.prompts.render_text(PromptId.RAG_GRAPH_SYSTEM, **values)

    # ——————————————————— 结果归一 ———————————————————

    def _to_extraction(
            self,
            chunk: Chunk,
            parsed: _LLMResultItem,
            metrics: dict[str, Any],
    ) -> ChunkGraphExtraction:
        """把单个 chunk 的 LLM 结果归一为标准名、补类型并过滤关系。"""
        max_entities = self.config.graph_max_keywords_per_chunk
        all_entities: list[GraphKeyword] = []
        types: dict[str, str] = {}
        seen: set[str] = set()
        for entity in parsed.entities:
            raw = normalize_keyword(entity.name)
            if not raw:
                continue
            canonical = self.profile.canonicalize_entity(raw)
            name = canonical[0] if canonical else raw
            if not _is_valid_entity_name(name, blacklist=self.blocked_terms):
                continue
            if self.profile.is_noise_term(name):
                continue
            if canonical is not None:
                entity_type = canonical[1]
            else:
                # 画像类型是辅助而不是硬类型集合：模型给出清单外类型时保留
                # 最贴近的合法类型（或配置的兜底类型），只记录指标，避免
                # 因画像枚举不完整而丢掉实体。
                resolved = self.profile.resolve_entity_type(entity.type or "")
                if resolved is None:
                    _add_unknown(
                        metrics,
                        count_key="unknown_types",
                        sample_key="unknown_type_samples",
                        sample=(
                            f"{name}("
                            f"{(entity.type or '').strip() or '<empty>'})"
                        ),
                    )
                    if not self.config.graph_allow_unknown_types:
                        # 配置关闭时保持旧行为：未知类型不入图，仅记指标。
                        continue
                    metrics["unknown_types_kept"] += 1
                    entity_type = self._fallback_entity_type(entity.type)
                else:
                    entity_type = resolved
                # type_blacklist 按类型限制：类型命中黑名单的实体不入图。
                if (
                    entity_type.casefold()
                    in self.profile.type_blacklist_names
                ):
                    continue
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
        for item in parsed.relations:
            subject = self._canonical_name(item.subject)
            obj = self._canonical_name(item.object)
            if not subject or not obj:
                continue
            subject_key = subject.casefold()
            object_key = obj.casefold()
            if subject_key not in types or object_key not in types:
                continue
            if self.profile.is_noise_term(subject) or self.profile.is_noise_term(
                    obj
            ):
                continue
            relation = _normalize_relation(item.relation)
            if not relation or subject_key == object_key:
                continue
            if relation in self.profile.relation_blacklist_names:
                continue
            if relation not in self.declared_relations:
                # 未声明关系记录指标；是否保留由配置决定。检索期图扩展仍只
                # 使用画像声明关系，未知关系只作为证据数据。
                _add_unknown(
                    metrics,
                    count_key="unknown_relations",
                    sample_key="unknown_relation_samples",
                    sample=f"{subject}-{relation}->{obj}",
                )
                if not self.config.graph_allow_unknown_relations:
                    continue
                metrics["unknown_relations_kept"] += 1
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
            declared_relations=self.declared_relations,
            allow_unknown=self.config.graph_allow_unknown_relations,
        )
        return ChunkGraphExtraction(
            chunk_id=chunk.chunk_id,
            keywords=entities,
            relations=filtered,
            model=self.model_id,
            confidence=0.9 if (entities or filtered) else None,
        )

    def _fallback_entity_type(self, raw: str | None) -> str:
        """把未声明类型归一为可落库的类型名。

        优先保留模型给出的、像类型标识的 ASCII 名称（如 ``vaccine_type``）；
        缺失或明显不是类型标识时使用配置的兜底类型（默认 ``other``）。
        """
        value = normalize_keyword(raw or "").lower()
        if value and re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", value):
            return value
        return self.config.graph_unknown_entity_type

    def _canonical_name(self, value: str) -> str:
        """把 LLM 返回的实体名归一为标准名；无效名返回空串。"""
        raw = normalize_keyword(value)
        if not raw:
            return ""
        canonical = self.profile.canonicalize(raw)
        name = canonical or raw
        if not _is_valid_entity_name(name, blacklist=self.blocked_terms):
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


def _normalize_relation(value: str | None) -> str:
    """归一关系名：只接受小写 ASCII 标识（如 used_for）；其他返回空串。"""
    text = normalize_keyword(value or "").lower()
    return text if re.fullmatch(r"[a-z][a-z0-9_-]{0,47}", text) else ""


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
