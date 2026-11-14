"""从模型列表构建 Provider 与注册表。

每个 ``ModelConfig`` 自带 base_url/key/协议，所以这里按模型创建 provider 壳；
provider 的 name 使用 ``vendor:model_id``，registry 同时提供按模型 id 查找。
"""

from __future__ import annotations

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
from provider.llm.openai_chat import OpenAIChatProvider
from provider.llm.openai_responses import OpenAIResponsesChatProvider
from provider.model_specs import model_spec_from_config
from provider.reranker.cohere import BgeRerankProvider, CohereRerankProvider
from provider.reranker.dashscope import DashScopeRerankProvider
from provider.registry import ProviderRegistry

__all__ = ["build_provider", "build_registry"]


def _provider_name(model: ModelConfig) -> str:
    return f"{model.provider}:{model.id}"


def _base_kwargs(model: ModelConfig) -> dict:
    return {
        "name": _provider_name(model),
        "base_url": model.base_url,
        "api_key_env": model.api_key_env,
        "api_key": model.api_key,
        "timeout_s": model.timeout_s,
        "max_retries": model.max_retries,
        "headers": dict(model.headers),
        "enabled": model.enabled,
        "metadata": dict(model.metadata),
    }


def build_provider(model: ModelConfig) -> BaseProvider:
    """为一个模型创建 provider 连接壳；不在这里注册模型。"""
    provider = model.provider.strip().lower()
    if model.kind is ProviderKind.CHAT:
        if provider == "anthropic":
            if model.api_style not in (None, ChatApiStyle.ANTHROPIC_MESSAGES):
                raise ConfigurationError(
                    f"Anthropic 模型 {model.id} 的 api_style 必须是 anthropic_messages",
                    details={"api_style": str(model.api_style)},
                )
            api_style = ChatApiStyle.ANTHROPIC_MESSAGES
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
            **_base_kwargs(model),
        )
        if api_style is ChatApiStyle.ANTHROPIC_MESSAGES:
            return AnthropicChatProvider(config)
        if api_style is ChatApiStyle.OPENAI_RESPONSES:
            return OpenAIResponsesChatProvider(config)
        return OpenAIChatProvider(config)
    if model.kind is ProviderKind.EMBEDDING:
        if provider not in {"openai", "bge"}:
            raise ConfigurationError(
                f"不支持的 embedding provider: {model.provider}",
                details={"model_id": model.id},
            )
        config = EmbeddingProviderConfig(kind=ProviderKind.EMBEDDING, **_base_kwargs(model))
        if provider == "bge":
            return BgeEmbeddingProvider(config)
        return OpenAIEmbeddingProvider(config)
    if model.kind is ProviderKind.RERANK:
        if provider not in {"cohere", "bge", "dashscope", "qwen"}:
            raise ConfigurationError(
                f"不支持的 rerank provider: {model.provider}",
                details={"model_id": model.id},
            )
        config = RerankProviderConfig(kind=ProviderKind.RERANK, **_base_kwargs(model))
        if provider in {"dashscope", "qwen"}:
            return DashScopeRerankProvider(config)
        if provider == "bge":
            return BgeRerankProvider(config)
        return CohereRerankProvider(config)
    raise ConfigurationError(f"未知模型类型: {model.kind}")


def build_registry(models: list[ModelConfig]) -> ProviderRegistry:
    seen: set[str] = set()
    registry = ProviderRegistry()
    for model in models:
        if model.id in seen:
            raise ConfigurationError(f"模型 id 重复: {model.id}")
        seen.add(model.id)
        provider = build_provider(model)
        spec = model_spec_from_config(
            model,
            provider_name=provider.name,
            defaults=getattr(provider, "default_capabilities", {}),
        )
        provider.register_model(spec)
        registry.register(provider)
    return registry
