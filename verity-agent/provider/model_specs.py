"""把顶层 ``ModelConfig`` 转成 Provider 层 ``ProviderModelSpec``。

Provider 不再是独立配置块；每个模型自带连接信息。Factory 为一个模型创建
一个 provider 壳，模型 id 在 registry 中作为全局查找键。
"""

from __future__ import annotations

from typing import Any, Mapping

from config.model import ModelConfig
from .base import (
    ChatModelCapabilities,
    EmbeddingModelCapabilities,
    ProviderModelSpec,
    ProviderPricing,
    RerankModelCapabilities,
)
from .enums import ProviderModelKind

__all__ = ["model_spec_from_config"]


def _pricing(value: Any) -> ProviderPricing | None:
    """把可选定价配置规范化为 ProviderPricing。"""
    if value is None:
        return None
    if isinstance(value, ProviderPricing):
        return value
    if isinstance(value, Mapping):
        return ProviderPricing.model_validate(value)
    return None


def _capabilities(model: ModelConfig, defaults: Mapping[str, Any] | None = None):
    """按模型 kind 合并默认能力与模型覆盖，生成对应能力模型。"""
    kind = ProviderModelKind(model.kind.value)
    caps = dict(defaults or {})
    caps.update(model.capabilities)
    if kind is ProviderModelKind.CHAT:
        base: dict[str, Any] = {
            "streaming": True,
            "tool_calling": True,
            "parallel_tool_calls": True,
            "vision": False,
            "file_input": False,
            "thinking": False,
            "prefix_completion": False,
            "json_object": False,
            "json_schema": False,
            "prompt_caching": False,
            "max_context_tokens": 8192,
            "max_output_tokens": 4096,
        }
        base.update(caps)
        return ChatModelCapabilities(**base)
    if kind is ProviderModelKind.EMBEDDING:
        base: dict[str, Any] = {"dimensions": 1024, "max_batch_size": 64}
        base.update(caps)
        return EmbeddingModelCapabilities(**base)
    base: dict[str, Any] = {"max_documents": 100, "max_batch_size": 32}
    base.update(caps)
    return RerankModelCapabilities(**base)


def model_spec_from_config(
    model: ModelConfig,
    *,
    provider_name: str,
    defaults: Mapping[str, Any] | None = None,
) -> ProviderModelSpec:
    """把顶层 ModelConfig 转换为 ProviderModelSpec（含默认能力与定价）。"""
    metadata = dict(model.metadata)
    metadata.setdefault("vendor", model.provider)
    if model.defaults:
        metadata["defaults"] = dict(model.defaults)
    return ProviderModelSpec(
        id=model.id,
        provider=provider_name,
        kind=ProviderModelKind(model.kind.value),
        model=model.model,
        capabilities=_capabilities(model, defaults),
        pricing=_pricing(model.pricing),
        enabled=model.enabled,
        metadata=metadata,
    )
