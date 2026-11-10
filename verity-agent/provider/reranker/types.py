"""Rerank provider 的 wire 模型。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from provider.enums import UsageSource

__all__ = [
    "RerankDocument",
    "RerankRequest",
    "RerankUsage",
    "RerankResult",
    "RerankResponse",
]


class RerankWireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RerankDocument(RerankWireModel):
    id: str
    text: str


class RerankRequest(RerankWireModel):
    model: str
    query: str
    documents: list[RerankDocument] = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1)
    return_documents: bool = False


class RerankUsage(RerankWireModel):
    query_tokens: int = Field(default=0, ge=0)
    document_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    source: UsageSource = UsageSource.PROVIDER


class RerankResult(RerankWireModel):
    index: int = Field(ge=0)
    id: str
    score: float
    document: str | None = None


class RerankResponse(RerankWireModel):
    results: list[RerankResult]
    model: str
    usage: RerankUsage = Field(default_factory=RerankUsage)
    request_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
