"""Provider 层公共导出。

具体厂商适配器（OpenAI/Anthropic/DeepSeek/BGE 等）后续放在
``provider.llm.*`` / ``provider.embedding.*`` / ``provider.reranker.*``；
本包先提供稳定的抽象、配置、注册表与错误模型。
"""

from config.provider import (
    ChatProviderConfig,
    EmbeddingProviderConfig,
    ProviderConfig,
    RerankProviderConfig,
)
from .base import (
    BaseProvider,
    ChatModelCapabilities,
    EmbeddingModelCapabilities,
    ModelCapabilities,
    ModelHandle,
    ProviderHealth,
    ProviderModelSpec,
    ProviderPricing,
    RerankModelCapabilities,
)
from .enums import (
    ChatApiStyle,
    ChatFinishReason,
    ModelRole,
    ProviderErrorCode,
    ProviderHealthStatus,
    ProviderKind,
    ProviderModelKind,
    ResponseFormatType,
    ToolChoiceMode,
    UsageSource,
)
from exception.provider import ProviderError
from .registry import ProviderRegistry

__all__ = [
    "BaseProvider",
    "ProviderRegistry",
    "ProviderConfig",
    "ChatProviderConfig",
    "EmbeddingProviderConfig",
    "RerankProviderConfig",
    "ProviderKind",
    "ProviderModelKind",
    "ProviderHealthStatus",
    "ProviderErrorCode",
    "ProviderError",
    "ProviderHealth",
    "ProviderModelSpec",
    "ProviderPricing",
    "ChatModelCapabilities",
    "EmbeddingModelCapabilities",
    "RerankModelCapabilities",
    "ModelCapabilities",
    "ModelHandle",
    "ChatApiStyle",
    "ChatFinishReason",
    "ModelRole",
    "ToolChoiceMode",
    "ResponseFormatType",
    "UsageSource",
]
