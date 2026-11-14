"""从模型列表构建 Provider 与注册表。

Provider 是“连接池单位”而不是“模型单位”：相同
``(vendor, kind, api_style, base_url, key, timeout, retries, headers, metadata)``
的模型共享一个 provider 壳，内部注册多个 ``ProviderModelSpec``。
这样同一网关的多个模型共用 SDK client/连接池，provider 级健康检查也有明确
作用域；模型仍通过全局唯一的 ``ModelConfig.id`` 查找。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from config.model import ModelConfig
from config.provider import (
    ChatProviderConfig,
    EmbeddingProviderConfig,
    ProviderKind,
    RerankProviderConfig,
    ChatApiStyle,
)
from exception import ConfigurationError
from provider.base import BaseProvider
from provider.embedding.openai import BgeEmbeddingProvider, OpenAIEmbeddingProvider
from provider.llm.anthropic import AnthropicChatProvider
from provider.llm.openai_chat import (
    DeepSeekChatProvider,
    OpenAIChatProvider,
    VLLMChatProvider,
)
from provider.llm.openai_responses import OpenAIResponsesChatProvider
from provider.model_specs import model_spec_from_config
from provider.reranker.cohere import BgeRerankProvider, CohereRerankProvider
from provider.reranker.dashscope import DashScopeRerankProvider
from provider.registry import ProviderRegistry

__all__ = ["build_provider", "build_registry"]


def _provider_name(model: ModelConfig) -> str:
    """直接构建单个 provider 时使用的兼容名字。"""
    return f"{model.provider}:{model.id}"


def _vendor(model: ModelConfig) -> str:
    """规范化厂商名；实际连接由 provider 配置承载。"""
    return model.provider.strip().lower()


def _connection_key(model: ModelConfig) -> tuple[Any, ...]:
    """决定哪些模型可以共享一个 provider / SDK client。"""
    return (
        _vendor(model),
        model.kind.value,
        model.api_style.value if model.api_style is not None else None,
        model.base_url.strip(),
        model.api_key_env,
        model.api_key.get_secret_value(),
        model.timeout_s,
        model.max_retries,
        tuple(sorted(model.headers.items())),
        json.dumps(model.metadata, sort_keys=True, ensure_ascii=False, default=str),
    )


def _base_kwargs(
    model: ModelConfig,
    *,
    name: str | None = None,
    enabled: bool | None = None,
) -> dict[str, Any]:
    return {
        "name": name or _provider_name(model),
        "base_url": model.base_url,
        "api_key_env": model.api_key_env,
        "api_key": model.api_key,
        "timeout_s": model.timeout_s,
        "max_retries": model.max_retries,
        "headers": dict(model.headers),
        "enabled": model.enabled if enabled is None else enabled,
        "metadata": dict(model.metadata),
    }


def build_provider(
    model: ModelConfig,
    *,
    name: str | None = None,
    enabled: bool | None = None,
) -> BaseProvider:
    """为一个模型/连接创建 provider 壳；不在这里注册模型。"""
    provider = _vendor(model)
    variant = str(model.metadata.get("variant") or "").strip().lower()
    if model.kind is ProviderKind.CHAT:
        if provider == "anthropic":
            if model.api_style not in (None, ChatApiStyle.ANTHROPIC_MESSAGES):
                raise ConfigurationError(
                    f"Anthropic 模型 {model.id} 的 api_style 必须是 anthropic_messages",
                    details={"api_style": str(model.api_style)},
                )
            api_style = ChatApiStyle.ANTHROPIC_MESSAGES
        elif provider in {"deepseek", "vllm", "llama.cpp", "llamacpp"}:
            # 这些厂商走 OpenAI Chat Completions 兼容协议，差异在 Model 子类。
            if model.api_style not in (None, ChatApiStyle.OPENAI_CHAT):
                raise ConfigurationError(
                    f"{model.provider} 模型 {model.id} 只支持 openai_chat",
                    details={"api_style": str(model.api_style)},
                )
            api_style = ChatApiStyle.OPENAI_CHAT
        elif provider == "openai":
            if model.api_style not in (
                None,
                ChatApiStyle.OPENAI_CHAT,
                ChatApiStyle.OPENAI_RESPONSES,
            ):
                raise ConfigurationError(
                    f"OpenAI 模型 {model.id} 的 api_style 必须是 "
                    "openai_chat 或 openai_responses",
                    details={"api_style": str(model.api_style)},
                )
            api_style = model.api_style or ChatApiStyle.OPENAI_CHAT
        else:
            raise ConfigurationError(
                f"暂不支持的 chat provider: {model.provider}",
                details={"model_id": model.id},
            )
        config = ChatProviderConfig(
            kind=ProviderKind.CHAT,
            api_style=api_style,
            **_base_kwargs(model, name=name, enabled=enabled),
        )
        if api_style is ChatApiStyle.ANTHROPIC_MESSAGES:
            return AnthropicChatProvider(config)
        if api_style is ChatApiStyle.OPENAI_RESPONSES:
            return OpenAIResponsesChatProvider(config)
        if (
            variant in {"deepseek", "vllm", "llama.cpp", "llamacpp"}
            and api_style is not ChatApiStyle.OPENAI_CHAT
        ):
            raise ConfigurationError(
                f"模型 {model.id} 的 variant={variant} 只支持 openai_chat",
                details={"api_style": str(api_style)},
            )
        if provider == "deepseek":
            return DeepSeekChatProvider(config)
        if provider in {"vllm", "llama.cpp", "llamacpp"}:
            return VLLMChatProvider(config)
        if variant == "deepseek":
            return DeepSeekChatProvider(config)
        if variant in {"vllm", "llama.cpp", "llamacpp"}:
            return VLLMChatProvider(config)
        return OpenAIChatProvider(config)
    if model.kind is ProviderKind.EMBEDDING:
        if provider not in {"openai", "bge"}:
            raise ConfigurationError(
                f"不支持的 embedding provider: {model.provider}",
                details={"model_id": model.id},
            )
        config = EmbeddingProviderConfig(
            kind=ProviderKind.EMBEDDING,
            **_base_kwargs(model, name=name, enabled=enabled),
        )
        if provider == "bge":
            return BgeEmbeddingProvider(config)
        return OpenAIEmbeddingProvider(config)
    if model.kind is ProviderKind.RERANK:
        if provider not in {"cohere", "bge", "dashscope", "qwen"}:
            raise ConfigurationError(
                f"不支持的 rerank provider: {model.provider}",
                details={"model_id": model.id},
            )
        config = RerankProviderConfig(
            kind=ProviderKind.RERANK,
            **_base_kwargs(model, name=name, enabled=enabled),
        )
        if provider in {"dashscope", "qwen"}:
            return DashScopeRerankProvider(config)
        if provider == "bge":
            return BgeRerankProvider(config)
        return CohereRerankProvider(config)
    raise ConfigurationError(f"未知模型类型: {model.kind}")


def _group_name(model: ModelConfig, key: tuple[Any, ...], used: set[str]) -> str:
    """为连接组生成稳定、可读的 provider 名。

    第一个使用某厂商名的连接直接叫 ``openai``；同厂商的不同网关/协议用
    连接 key 的短摘要区分，避免模型顺序变化导致名字漂移。
    """
    base = _vendor(model)
    if base not in used:
        return base
    digest = hashlib.sha1(repr(key).encode("utf-8")).hexdigest()[:8]
    name = f"{base}:{digest}"
    suffix = 1
    while name in used:
        suffix += 1
        name = f"{base}:{digest}-{suffix}"
    return name


def build_registry(models: list[ModelConfig]) -> ProviderRegistry:
    """按连接分组构建 provider，并注册全局模型 id。"""
    seen: set[str] = set()
    groups: dict[tuple[Any, ...], list[ModelConfig]] = {}
    for model in models:
        if model.id in seen:
            raise ConfigurationError(f"模型 id 重复: {model.id}")
        seen.add(model.id)
        groups.setdefault(_connection_key(model), []).append(model)

    registry = ProviderRegistry()
    used_names: set[str] = set()
    # 按组内最小模型 id 排序，保证 provider 名不依赖 MODELS 的书写顺序。
    for key, group in sorted(
        groups.items(),
        key=lambda item: min(model.id for model in item[1]),
    ):
        representative = group[0]
        name = _group_name(representative, key, used_names)
        used_names.add(name)
        provider = build_provider(
            representative,
            name=name,
            enabled=any(item.enabled for item in group),
        )
        for model in group:
            spec = model_spec_from_config(
                model,
                provider_name=provider.name,
                defaults=getattr(provider, "default_capabilities", {}),
            )
            provider.register_model(spec)
        registry.register(provider)
    return registry
