"""向量化服务：把 chunk 的 embedding_text 批量送入 embedding provider。

- 只依赖 provider 的 ``EmbeddingModel.embed`` 协议，不绑定具体厂商；
- 负责批大小、max_input_tokens 截断、维度/数量校验、usage 汇总；
- 失败统一转成 ``RagError``，由 IndexingPipeline 记为 EMBEDDING_FAILED。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from exception import RagError, RagErrorCode
from provider.embedding.types import EmbeddingRequest, EmbeddingUsage
from provider.enums import UsageSource
from rag.shared.models import Chunk, ChunkDraft, EmbeddingBatch, EmbeddingInput
from utils.tokens import truncate_to_tokens

__all__ = ["EmbeddingResult", "EmbeddingService"]


def _chunk_embedding_text(chunk: Chunk) -> str:
    """重建 chunk 的 embedding 文本：完整标题路径 + 正文。"""
    heading = " ".join(chunk.heading_path) if chunk.heading_path else (
        chunk.heading or ""
    )
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


class EmbeddingService:
    """EmbeddingModel 的业务封装。"""

    def __init__(
        self,
        model,
        *,
        model_id: str | None = None,
        batch_size: int = 32,
    ) -> None:
        """保存模型句柄和批大小；模型必须提供 ``embed()``。"""
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
        self._caps = getattr(spec, "embedding_capabilities", None)
        if self._caps is not None:
            self.batch_size = min(self.batch_size, int(self._caps.max_batch_size))

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
        与切片阶段的“标题 + 正文”口径保持一致。
        """
        return await self.embed_texts(
            [_chunk_embedding_text(chunk) for chunk in chunks],
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
        """批量向量化文本，返回与输入顺序一致的向量。"""
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

        total_batches = (len(prepared) + self.batch_size - 1) // self.batch_size
        vectors: list[list[float]] = []
        batches: list[EmbeddingBatch] = []
        usage_total = 0
        usage_input = 0
        resolved_dimensions: int | None = dimensions

        for batch_no, start in enumerate(
            range(0, len(prepared), self.batch_size),
            start=1,
        ):
            batch_texts = prepared[start : start + self.batch_size]
            batch_ids = (
                input_ids[start : start + self.batch_size]
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
            vectors.extend(response.vectors)
            usage_input += int(response.usage.input_tokens)
            usage_total += int(response.usage.total_tokens)
            batches.append(
                EmbeddingBatch(
                    model_id=self.model_id,
                    dimensions=batch_dimensions,
                    inputs=[
                        EmbeddingInput(chunk_id=chunk_id, text=text)
                        for chunk_id, text in zip(batch_ids, batch_texts, strict=True)
                    ],
                    vectors=response.vectors,
                    usage=response.usage,
                    batch_no=batch_no,
                    batch_total=total_batches,
                )
            )

        return EmbeddingResult(
            vectors=vectors,
            dimensions=resolved_dimensions or 0,
            usage=EmbeddingUsage(
                input_tokens=usage_input,
                total_tokens=usage_total or usage_input,
                source=UsageSource.PROVIDER,
            ),
            batches=batches,
            truncated_count=truncated,
        )

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
