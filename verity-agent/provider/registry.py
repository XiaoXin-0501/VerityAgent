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
        """初始化 provider、模型句柄与全局模型索引。"""
        self._providers: dict[str, BaseProvider] = {}
        self._models: dict[tuple[str, str], ModelHandle] = {}
        self._models_by_id: dict[str, ModelHandle] = {}

    def register(self, provider: BaseProvider) -> None:
        """注册 provider 及其模型句柄；拒绝重名 provider 和重复模型 id。"""
        name = provider.name
        if name in self._providers:
            raise ConfigurationError(f"Provider 重复注册: {name}")
        handles = [(spec.id, provider.get_model(spec.id)) for spec in provider.list_models()]
        for model_id, _handle in handles:
            existing = self._models_by_id.get(model_id)
            if existing is not None:
                raise ConfigurationError(
                    f"模型 id 被多个 provider 注册: {model_id}",
                    details={"provider": name},
                )
        self._providers[name] = provider
        for model_id, handle in handles:
            self._models[(name, model_id)] = handle
            self._models_by_id[model_id] = handle

    def register_many(self, providers: list[BaseProvider]) -> None:
        """批量注册 provider，任一失败即抛出。"""
        for provider in providers:
            self.register(provider)

    def get(self, name: str, *, kind: ProviderKind | None = None) -> BaseProvider:
        """按名称取 provider；可校验 kind。"""
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
        """按名称取 provider；不存在返回 None。"""
        return self._providers.get(name)

    def list(self, kind: ProviderKind | None = None) -> list[BaseProvider]:
        """列出全部或指定 kind 的 provider。"""
        providers = list(self._providers.values())
        if kind is not None:
            providers = [p for p in providers if p.kind is kind]
        return providers

    def get_model(self, provider: str, model_id: str) -> ModelHandle:
        """按 (provider, model_id) 取句柄，首次访问时惰性创建。"""
        key = (provider, model_id)
        handle = self._models.get(key)
        if handle is None:
            # 触发 provider 的明确错误（模型未注册/配置错误）
            handle = self.get(provider).get_model(model_id)
            self._models[key] = handle
        return handle

    def get_model_by_id(self, model_id: str, *, include_disabled: bool = False) -> ModelHandle:
        """按应用模型 id 取句柄；默认拒绝禁用模型，避免延迟到调用期报错。"""
        handle = self._models_by_id.get(model_id)
        if handle is None:
            raise ConfigurationError(
                f"模型不存在: {model_id}",
                details={"model_id": model_id},
            )
        if not include_disabled and not handle.spec.enabled:
            raise ConfigurationError(
                f"模型已禁用: {model_id}",
                details={"model_id": model_id, "provider": handle.spec.provider},
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
        """并发关闭所有 provider，忽略单个关闭异常。"""
        await asyncio.gather(
            *(p.close() for p in self._providers.values()), return_exceptions=True
        )

    async def health(self) -> list[ProviderHealth]:
        """并发探活并汇总健康状态，异常的 provider 标记为 DOWN。"""
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
        """已注册 provider 数量。"""
        return len(self._providers)
