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
        """校验模型类型并保存模型规格。"""
        if spec.kind.value != ProviderKind.RERANK.value:
            raise ValueError(f"{spec.id} 不是 rerank 模型")
        self.spec = spec

    async def rerank(self, request: RerankRequest) -> RerankResponse:
        """对候选文档重排；由具体厂商适配器实现。"""
        raise NotImplementedError


class RerankProvider(BaseProvider[RerankProviderConfig]):
    kind = ProviderKind.RERANK

    def __init__(self, config: RerankProviderConfig) -> None:
        """保存 Provider 配置。"""
        super().__init__(config)

    @abstractmethod
    def get_model(self, model_id: str) -> RerankModel:
        """返回指定 rerank 模型句柄。"""
        raise NotImplementedError
