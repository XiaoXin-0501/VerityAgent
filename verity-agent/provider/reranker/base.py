"""Rerank provider / RerankModel 抽象。"""

from __future__ import annotations

from abc import abstractmethod

from provider.base import BaseProvider, ProviderModelSpec
from config.provider import RerankProviderConfig
from provider.enums import ProviderKind

from .types import RerankRequest, RerankResponse

__all__ = ["RerankModel", "RerankProvider"]


class RerankModel:
    def __init__(self, spec: ProviderModelSpec) -> None:
        if spec.kind.value != ProviderKind.RERANK.value:
            raise ValueError(f"{spec.id} 不是 rerank 模型")
        self.spec = spec

    async def rerank(self, request: RerankRequest) -> RerankResponse:
        raise NotImplementedError


class RerankProvider(BaseProvider[RerankProviderConfig]):
    kind = ProviderKind.RERANK

    def __init__(self, config: RerankProviderConfig) -> None:
        super().__init__(config)

    @abstractmethod
    def get_model(self, model_id: str) -> RerankModel:
        raise NotImplementedError
