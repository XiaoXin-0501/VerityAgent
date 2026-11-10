"""Embedding provider 的 wire 模型。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from provider.enums import UsageSource

__all__ = [
    "EmbeddingRequest",
    "EmbeddingUsage",
    "EmbeddingResponse",
]


class EmbeddingWireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EmbeddingRequest(EmbeddingWireModel):
    model: str
    inputs: list[str] = Field(min_length=1)
    dimensions: int | None = Field(default=None, gt=0)
    normalize: bool = True
    batch_size: int | None = Field(default=None, ge=1)
    input_type: str | None = None


class EmbeddingUsage(EmbeddingWireModel):
    input_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    source: UsageSource = UsageSource.PROVIDER


class EmbeddingResponse(EmbeddingWireModel):
    vectors: list[list[float]]
    dimensions: int = Field(gt=0)
    usage: EmbeddingUsage = Field(default_factory=EmbeddingUsage)
    model: str
    request_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
