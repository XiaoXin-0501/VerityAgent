"""Provider 注册表与生命周期（运行期只读）。"""

from __future__ import annotations

import asyncio
import logging

from exception import ConfigurationError

from .base import BaseProvider, ModelHandle, ProviderHealth
from .enums import ProviderKind

logger = logging.getLogger(__name__)

__all__ = ["ProviderRegistry"]


class ProviderRegistry:
    """按名字注册 provider，按 ``(provider, model_id)`` 缓存模型句柄。

    - provider 名唯一；启动时注册，运行期不动态增删；
    - provider 不可用时 ``get_model`` 仍返回句柄，调用时由 provider 抛
      ``ProviderError(code=UNAVAILABLE)``，由 LLM 层决定 fallback（D-42）。
    """

    def __init__(self) -> None:
        self._providers: dict[str, BaseProvider] = {}
        self._models: dict[tuple[str, str], ModelHandle] = {}
        self._models_by_id: dict[str, ModelHandle] = {}

    def register(self, provider: BaseProvider) -> None:
        name = provider.name
        if name in self._providers:
            raise ConfigurationError(f"Provider 重复注册: {name}")
        self._providers[name] = provider
        for spec in provider.list_models():
            handle = provider.get_model(spec.id)
            self._models[(name, spec.id)] = handle
            self._models_by_id[spec.id] = handle

    def register_many(self, providers: list[BaseProvider]) -> None:
        for provider in providers:
            self.register(provider)

    def get(self, name: str, *, kind: ProviderKind | None = None) -> BaseProvider:
        provider = self._providers.get(name)
        if provider is None:
            raise ConfigurationError(
                f"Provider 不存在: {name}",
                details={"provider": name, "available": sorted(self._providers)},
            )
        if kind is not None and provider.kind is not kind:
            raise ConfigurationError(
                f"Provider {name} 类型为 {provider.kind}，不是 {kind}",
                details={"provider": name, "kind": str(kind)},
            )
        return provider

    def maybe_get(self, name: str) -> BaseProvider | None:
        return self._providers.get(name)

    def list(self, kind: ProviderKind | None = None) -> list[BaseProvider]:
        providers = list(self._providers.values())
        if kind is not None:
            providers = [p for p in providers if p.kind is kind]
        return providers

    def get_model(self, provider: str, model_id: str) -> ModelHandle:
        key = (provider, model_id)
        handle = self._models.get(key)
        if handle is None:
            # 触发 provider 的明确错误（模型未注册/配置错误）
            handle = self.get(provider).get_model(model_id)
            self._models[key] = handle
        return handle

    def get_model_by_id(self, model_id: str) -> ModelHandle:
        """按应用模型 id 取句柄；LLM ModelRegistry 后续直接使用。"""
        handle = self._models_by_id.get(model_id)
        if handle is None:
            raise ConfigurationError(
                f"模型不存在: {model_id}",
                details={"model_id": model_id},
            )
        return handle

    async def init_all(self) -> None:
        """并发打开所有启用的 provider；单个失败不阻断其他 provider。"""
        providers = [p for p in self._providers.values() if p.enabled]
        results = await asyncio.gather(
            *(p.open() for p in providers), return_exceptions=True
        )
        for provider, result in zip(providers, results, strict=True):
            if isinstance(result, Exception):
                logger.error(
                    "Provider 初始化失败: %s", provider.name, exc_info=result
                )

    async def close_all(self) -> None:
        await asyncio.gather(
            *(p.close() for p in self._providers.values()), return_exceptions=True
        )

    async def health(self) -> list[ProviderHealth]:
        providers = [p for p in self._providers.values() if p.enabled]
        results = await asyncio.gather(
            *(p.health_check() for p in providers), return_exceptions=True
        )
        health: list[ProviderHealth] = []
        for provider, result in zip(providers, results, strict=True):
            if isinstance(result, ProviderHealth):
                health.append(result)
            else:
                from .base import ProviderHealth as HealthModel
                from .enums import ProviderHealthStatus

                health.append(
                    HealthModel(
                        provider=provider.name,
                        status=ProviderHealthStatus.DOWN,
                        message=str(result),
                    )
                )
        return health

    def __len__(self) -> int:
        return len(self._providers)
