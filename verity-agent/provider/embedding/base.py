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
        """校验模型类型并保存模型规格。"""
        if spec.kind.value != ProviderKind.EMBEDDING.value:
            raise ValueError(f"{spec.id} 不是 embedding 模型")
        self.spec = spec

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """非流式向量化；由具体厂商适配器实现。"""
        raise NotImplementedError


class EmbeddingProvider(BaseProvider[EmbeddingProviderConfig]):
    kind = ProviderKind.EMBEDDING

    def __init__(self, config: EmbeddingProviderConfig) -> None:
        """保存 Provider 配置。"""
        super().__init__(config)

    @abstractmethod
    def get_model(self, model_id: str) -> EmbeddingModel:
        """返回指定 embedding 模型句柄。"""
        raise NotImplementedError
