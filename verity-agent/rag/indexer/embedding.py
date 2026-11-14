"""向量化服务：把 chunk 的 embedding_text 批量送入 embedding provider。

- 只依赖 provider 的 ``EmbeddingModel.embed`` 协议，不绑定具体厂商；
- 负责批大小、max_input_tokens 截断、维度/数量校验、usage 汇总；
- 失败统一转成 ``RagError``，由 IndexingPipeline 记为 EMBEDDING_FAILED。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from exception import RagError, RagErrorCode
from provider.embedding.types import EmbeddingRequest, EmbeddingUsage
from provider.enums import UsageSource
from rag.shared.enums import BlockType
from rag.shared.models import Chunk, ChunkDraft, EmbeddingBatch, EmbeddingInput
from utils.tokens import truncate_to_tokens

from .caching import ChunkCache, embedding_cache_key
from .chunkers.base import semantic_embedding_text

__all__ = ["EmbeddingResult", "EmbeddingService"]

logger = logging.getLogger(__name__)

# 需要“heading + 语义描述”向量化的特殊块；与切片阶段 §5.5 口径一致。
_SEMANTIC_BLOCK_TYPES = frozenset({BlockType.TABLE, BlockType.CODE})


def _chunk_embedding_text(
    chunk: Chunk,
    *,
    semantic_embedding_enabled: bool = True,
) -> str:
    """重建 chunk 的 embedding 文本，必须与切片阶段口径一致。

    - 文本/列表：完整标题路径 + 正文；
    - 表格/代码：``semantic_embedding_enabled`` 且有语义描述时，只用
      heading + 描述（正文语义密度低，整段向量化会稀释信号）；
    - 没有语义描述或关闭语义向量化时，回退到 heading + 正文，避免空向量。
    """
    heading = chunk.heading or (
        " > ".join(chunk.heading_path) if chunk.heading_path else ""
    )
    if (
        semantic_embedding_enabled
        and chunk.block_type in _SEMANTIC_BLOCK_TYPES
    ):
        semantic = semantic_embedding_text(
            heading,
            chunk.semantic_description,
        )
        if semantic:
            return semantic
    return f"{heading}\n{chunk.content}" if heading else chunk.content


@dataclass(slots=True)
class EmbeddingResult:
    """一次批量向量化结果。"""

    vectors: list[list[float]] = field(default_factory=list)
    dimensions: int = 0
    usage: EmbeddingUsage = field(default_factory=EmbeddingUsage)
    batches: list[EmbeddingBatch] = field(default_factory=list)
    # 被 max_input_tokens 截断的文本数
    truncated_count: int = 0
    # 命中缓存、未调用 embedding 模型的文本数
    cache_hits: int = 0
    # 未命中缓存、实际调用模型的文本数
    cache_misses: int = 0


class EmbeddingService:
    """EmbeddingModel 的业务封装。"""

    def __init__(
        self,
        model,
        *,
        model_id: str | None = None,
        batch_size: int = 32,
        semantic_embedding_enabled: bool = True,
        cache: ChunkCache | None = None,
        cache_enabled: bool = True,
        cache_ttl_s: int = 2_592_000,
        cache_version: str = "v1",
        configured_dimensions: int | None = None,
    ) -> None:
        """保存模型句柄、批大小、特殊块向量化策略与向量缓存。

        ``cache`` 为空时不做缓存；``cache_version`` 参与 key，换 embedding
        模型版本时旧向量自然失效。``configured_dimensions`` 来自
        ``IndexerConfig.embedding_dimensions``，用于让缓存 key 在进程重启后
        仍保持一致。
        """
        if not hasattr(model, "embed"):
            raise RagError(
                RagErrorCode.INVALID_CONFIG,
                "embedding_model 必须提供 embed()",
                details={"model_id": model_id},
            )
        self.model = model
        spec = getattr(model, "spec", None)
        self.model_id = model_id or getattr(spec, "id", None) or "embedding"
        self.batch_size = max(1, int(batch_size))
        self.semantic_embedding_enabled = semantic_embedding_enabled
        self.cache = cache
        self.cache_enabled = bool(
            cache_enabled and cache is not None and int(cache_ttl_s) > 0
        )
        self.cache_ttl_s = int(cache_ttl_s)
        self.cache_version = cache_version
        self._caps = getattr(spec, "embedding_capabilities", None)
        if self._caps is not None:
            self.batch_size = min(self.batch_size, int(self._caps.max_batch_size))
        # 缓存 key 必须使用“配置维度”而不是某次响应维度，否则首次运行
        # 写入的 key 与重启后读取的 key 不一致，缓存永远不会命中。
        self._cache_dimensions = int(
            configured_dimensions or self.dimensions or 0
        )

    @property
    def dimensions(self) -> int | None:
        """模型声明的维度（若 provider spec 提供）。"""
        return int(self._caps.dimensions) if self._caps is not None else None

    async def embed_chunks(
        self,
        chunks: list[ChunkDraft],
        *,
        dimensions: int | None = None,
    ) -> EmbeddingResult:
        """向量化 ChunkDraft；使用 ``embedding_text``，不是正文。"""
        return await self.embed_texts(
            [chunk.embedding_text for chunk in chunks],
            dimensions=dimensions,
            input_type="document",
            input_ids=[chunk.chunk_id for chunk in chunks],
        )

    async def embed_chunk_models(
        self,
        chunks: list[Chunk],
        *,
        dimensions: int | None = None,
    ) -> EmbeddingResult:
        """向量化已转换的 ``Chunk``；用于重建/补偿任务。

        ``Chunk`` 不落 embedding_text，这里按 heading_path + content 重建，
        与切片阶段的“标题 + 正文”口径保持一致；表格/代码优先使用
        heading + ``semantic_description``。
        """
        return await self.embed_texts(
            [
                _chunk_embedding_text(
                    chunk,
                    semantic_embedding_enabled=(
                        self.semantic_embedding_enabled
                    ),
                )
                for chunk in chunks
            ],
            dimensions=dimensions,
            input_type="document",
            input_ids=[chunk.chunk_id for chunk in chunks],
        )

    async def embed_query(
        self,
        query: str,
        *,
        dimensions: int | None = None,
    ) -> EmbeddingResult:
        """向量化 query；Retriever 侧使用。"""
        return await self.embed_texts(
            [query],
            dimensions=dimensions,
            input_type="query",
        )

    async def embed_texts(
        self,
        texts: list[str],
        *,
        dimensions: int | None = None,
        input_type: str = "document",
        input_ids: list[str] | None = None,
    ) -> EmbeddingResult:
        """批量向量化文本，返回与输入顺序一致的向量。

        缓存策略：逐条按“模型 + 维度 + input_type + 文本摘要”查缓存；命中
        的文本不进入模型批次，只对 miss 的文本调用 provider，再按原顺序
        合并。这样入库在后续阶段失败重试时，已经算过的 chunk 不会重复消耗
        embedding token。
        """
        if not texts:
            return EmbeddingResult(
                dimensions=dimensions or self.dimensions or 0,
            )

        max_tokens = (
            int(self._caps.max_input_tokens)
            if self._caps is not None
            else 8192
        )
        prepared: list[str] = []
        truncated = 0
        for text in texts:
            value = (text or "").strip()
            clipped = truncate_to_tokens(value, max_tokens)
            if clipped != value:
                truncated += 1
            prepared.append(clipped or " ")

        expected_dimensions = (
            dimensions or self.dimensions or self._cache_dimensions or None
        )
        vectors: list[list[float] | None] = [None] * len(prepared)
        miss_indices: list[int] = []
        cache_hits = 0

        # 1) 先查缓存；命中项直接填回原位置，不再进入模型批次。
        for index, text in enumerate(prepared):
            cached = await self._cached_vector(
                text=text,
                dimensions=expected_dimensions,
                input_type=input_type,
            )
            if cached is not None:
                vectors[index] = cached
                cache_hits += 1
                if expected_dimensions is None:
                    expected_dimensions = len(cached)
                continue
            miss_indices.append(index)

        # 2) 只对 miss 文本调用模型；批次编号只反映真实调用。
        total_batches = (
            (len(miss_indices) + self.batch_size - 1) // self.batch_size
            if miss_indices
            else 0
        )
        batches: list[EmbeddingBatch] = []
        usage_total = 0
        usage_input = 0
        resolved_dimensions: int | None = expected_dimensions

        for batch_no, start in enumerate(
            range(0, len(miss_indices), self.batch_size),
            start=1,
        ):
            batch_indices = miss_indices[start : start + self.batch_size]
            batch_texts = [prepared[index] for index in batch_indices]
            batch_ids = (
                [input_ids[index] for index in batch_indices]
                if input_ids is not None
                else ["" for _ in batch_texts]
            )
            response = await self._embed_batch(
                batch_texts,
                dimensions=dimensions,
                input_type=input_type,
            )
            if len(response.vectors) != len(batch_texts):
                raise RagError(
                    RagErrorCode.DEPENDENCY_UNAVAILABLE,
                    "Embedding 返回向量数量与输入不一致",
                    details={
                        "model_id": self.model_id,
                        "expected": len(batch_texts),
                        "actual": len(response.vectors),
                    },
                    retryable=True,
                )
            batch_dimensions = int(response.dimensions)
            if resolved_dimensions is None:
                resolved_dimensions = batch_dimensions
            if batch_dimensions != resolved_dimensions:
                raise RagError(
                    RagErrorCode.DEPENDENCY_UNAVAILABLE,
                    "Embedding 向量维度不一致",
                    details={
                        "model_id": self.model_id,
                        "expected": resolved_dimensions,
                        "actual": batch_dimensions,
                    },
                    retryable=True,
                )
            for index, vector in zip(
                batch_indices,
                response.vectors,
                strict=True,
            ):
                vectors[index] = list(vector)
                # 写回缓存；写失败只降级，不影响本次入库。
                await self._cache_vector(
                    text=prepared[index],
                    vector=vector,
                    dimensions=batch_dimensions,
                    input_type=input_type,
                )
            usage_input += int(response.usage.input_tokens)
            usage_total += int(response.usage.total_tokens)
            batches.append(
                EmbeddingBatch(
                    model_id=self.model_id,
                    dimensions=batch_dimensions,
                    inputs=[
                        EmbeddingInput(chunk_id=chunk_id, text=text)
                        for chunk_id, text in zip(
                            batch_ids,
                            batch_texts,
                            strict=True,
                        )
                    ],
                    vectors=response.vectors,
                    usage=response.usage,
                    batch_no=batch_no,
                    batch_total=total_batches,
                )
            )

        if any(vector is None for vector in vectors):
            raise RagError(
                RagErrorCode.INTERNAL,
                "Embedding 结果缺失，无法与输入对齐",
                details={"model_id": self.model_id},
            )

        return EmbeddingResult(
            vectors=[vector for vector in vectors if vector is not None],
            dimensions=resolved_dimensions or 0,
            usage=EmbeddingUsage(
                input_tokens=usage_input,
                total_tokens=usage_total or usage_input,
                source=UsageSource.PROVIDER,
            ),
            batches=batches,
            truncated_count=truncated,
            cache_hits=cache_hits,
            cache_misses=len(miss_indices),
        )

    async def _cached_vector(
        self,
        *,
        text: str,
        dimensions: int | None,
        input_type: str,
    ) -> list[float] | None:
        """读取并校验缓存向量；格式/维度不符按 miss 处理。"""
        if not self.cache_enabled or self.cache is None:
            return None
        key = embedding_cache_key(
            model_id=self.model_id,
            dimensions=self._cache_dimensions,
            input_type=input_type,
            text=text,
            version=self.cache_version,
        )
        try:
            payload = await self.cache.get(key)
        except Exception as exc:  # noqa: BLE001 - 缓存失败只降级
            logger.warning("读取 embedding 缓存失败: %s", exc)
            return None
        raw = payload.get("vector") if isinstance(payload, dict) else payload
        if not isinstance(raw, list) or not raw:
            return None
        try:
            vector = [float(value) for value in raw]
        except (TypeError, ValueError):
            return None
        if dimensions is not None and len(vector) != dimensions:
            # 维度变化说明模型/collection 已换，旧向量不可复用。
            return None
        return vector

    async def _cache_vector(
        self,
        *,
        text: str,
        vector: list[float],
        dimensions: int,
        input_type: str,
    ) -> None:
        """写回向量缓存；失败不抛，避免影响入库主流程。"""
        if not self.cache_enabled or self.cache is None:
            return
        key = embedding_cache_key(
            model_id=self.model_id,
            dimensions=self._cache_dimensions,
            input_type=input_type,
            text=text,
            version=self.cache_version,
        )
        try:
            await self.cache.set(
                key,
                {
                    "vector": list(vector),
                    "model_id": self.model_id,
                    "dimensions": int(dimensions),
                    "input_type": input_type,
                    "version": self.cache_version,
                },
                ttl_s=self.cache_ttl_s,
            )
        except Exception as exc:  # noqa: BLE001 - 缓存失败只降级
            logger.warning("写入 embedding 缓存失败: %s", exc)

    async def _embed_batch(
        self,
        texts: list[str],
        *,
        dimensions: int | None,
        input_type: str,
    ):
        """调用 provider；异常统一收敛到 RagError。"""
        try:
            return await self.model.embed(
                EmbeddingRequest(
                    model=self.model_id,
                    inputs=texts,
                    dimensions=dimensions,
                    normalize=True,
                    batch_size=len(texts),
                    input_type=input_type,
                )
            )
        except RagError:
            raise
        except Exception as exc:
            raise RagError(
                RagErrorCode.DEPENDENCY_UNAVAILABLE,
                "Embedding 调用失败",
                details={"model_id": self.model_id, "cause": str(exc)},
                retryable=True,
            ) from exc
