"""Embedding provider / EmbeddingModel 抽象。"""

from __future__ import annotations

from abc import abstractmethod

from provider.base import BaseProvider, ProviderModelSpec
from config.provider import EmbeddingProviderConfig
from provider.enums import ProviderKind

from .types import EmbeddingRequest, EmbeddingResponse

__all__ = ["EmbeddingModel", "EmbeddingProvider"]


class EmbeddingModel:
    def __init__(self, spec: ProviderModelSpec) -> None:
        if spec.kind.value != ProviderKind.EMBEDDING.value:
            raise ValueError(f"{spec.id} 不是 embedding 模型")
        self.spec = spec

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        raise NotImplementedError


class EmbeddingProvider(BaseProvider[EmbeddingProviderConfig]):
    kind = ProviderKind.EMBEDDING

    def __init__(self, config: EmbeddingProviderConfig) -> None:
        super().__init__(config)

    @abstractmethod
    def get_model(self, model_id: str) -> EmbeddingModel:
        raise NotImplementedError
