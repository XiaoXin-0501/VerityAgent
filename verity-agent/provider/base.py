"""Provider 层公共模型与基类。

- 配置模型在 :mod:`config.provider`；
- 三类模型句柄分别在 ``provider.llm`` / ``provider.embedding`` /
  ``provider.reranker``；
- 本模块只定义厂商无关的元数据、能力、价格、健康状态与生命周期。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from exception import ConfigurationError
from utils.time import utcnow

from config.provider import ProviderConfig
from .enums import ProviderHealthStatus, ProviderKind, ProviderModelKind

__all__ = [
    "ProviderPricing",
    "ChatModelCapabilities",
    "EmbeddingModelCapabilities",
    "RerankModelCapabilities",
    "ModelCapabilities",
    "ProviderModelSpec",
    "ProviderHealth",
    "ModelHandle",
    "BaseProvider",
]


class _SpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProviderPricing(_SpecModel):
    input_per_million: float | None = None
    output_per_million: float | None = None
    cached_input_per_million: float | None = None
    per_request: float | None = None
    per_document: float | None = None
    currency: str = "USD"
    effective_from: datetime = Field(default_factory=utcnow)


class ChatModelCapabilities(_SpecModel):
    streaming: bool = True
    tool_calling: bool = False
    parallel_tool_calls: bool = False
    vision: bool = False
    file_input: bool = False
    thinking: bool = False
    prefix_completion: bool = False
    json_object: bool = False
    json_schema: bool = False
    prompt_caching: bool = False
    max_context_tokens: int = Field(default=8192, ge=1)
    max_output_tokens: int = Field(default=4096, ge=1)
    max_tools: int | None = Field(default=None, ge=1)
    max_images: int | None = Field(default=None, ge=1)
    input_modalities: set[str] = Field(default_factory=lambda: {"text"})
    output_modalities: set[str] = Field(default_factory=lambda: {"text"})
    tokenizer: str | None = None


class EmbeddingModelCapabilities(_SpecModel):
    dimensions: int = Field(ge=1)
    max_batch_size: int = Field(default=64, ge=1)
    max_input_tokens: int = Field(default=8192, ge=1)
    supports_dimensions: bool = False
    supports_normalize: bool = True
    input_type: str | None = None


class RerankModelCapabilities(_SpecModel):
    max_documents: int = Field(default=100, ge=1)
    max_query_tokens: int = Field(default=512, ge=1)
    max_document_tokens: int = Field(default=4096, ge=1)
    max_batch_size: int = Field(default=32, ge=1)
    supports_return_documents: bool = False


ModelCapabilities = (
    ChatModelCapabilities | EmbeddingModelCapabilities | RerankModelCapabilities
)


class ProviderModelSpec(_SpecModel):
    """Provider 内一个模型的元数据（D-39：kind 使用 Provider 自有枚举）。"""

    id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    kind: ProviderModelKind
    model: str = Field(min_length=1)
    capabilities: ModelCapabilities
    pricing: ProviderPricing | None = None
    enabled: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _capabilities_match_kind(self) -> "ProviderModelSpec":
        expected = {
            ProviderModelKind.CHAT: ChatModelCapabilities,
            ProviderModelKind.EMBEDDING: EmbeddingModelCapabilities,
            ProviderModelKind.RERANK: RerankModelCapabilities,
        }[self.kind]
        if not isinstance(self.capabilities, expected):
            raise ValueError(
                f"{self.kind} 模型的 capabilities 必须是 {expected.__name__}"
            )
        return self

    @property
    def chat_capabilities(self) -> ChatModelCapabilities:
        if not isinstance(self.capabilities, ChatModelCapabilities):
            raise TypeError("不是 chat 模型")
        return self.capabilities

    @property
    def embedding_capabilities(self) -> EmbeddingModelCapabilities:
        if not isinstance(self.capabilities, EmbeddingModelCapabilities):
            raise TypeError("不是 embedding 模型")
        return self.capabilities

    @property
    def rerank_capabilities(self) -> RerankModelCapabilities:
        if not isinstance(self.capabilities, RerankModelCapabilities):
            raise TypeError("不是 rerank 模型")
        return self.capabilities


class ProviderHealth(_SpecModel):
    provider: str
    status: ProviderHealthStatus
    latency_ms: int | None = None
    checked_at: datetime = Field(default_factory=utcnow)
    message: str | None = None


@runtime_checkable
class ModelHandle(Protocol):
    """模型句柄的最小公共面；三类句柄结构相同，行为不同。"""

    spec: ProviderModelSpec


class BaseProvider(ABC):
    """Provider 生命周期与模型清单。

    启动时一次性注册，运行期只读；具体厂商在子类里实现 ``get_model``。
    """

    kind: ProviderKind

    def __init__(self, config: ProviderConfig) -> None:
        if config.kind is not self.kind:
            raise ConfigurationError(
                f"provider {config.name} 配置类型为 {config.kind}，"
                f"但实例是 {self.kind}"
            )
        self.config = config
        self._models: dict[str, ProviderModelSpec] = {}
        self._opened = False

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def register_model(self, spec: ProviderModelSpec) -> None:
        if spec.provider != self.name:
            raise ConfigurationError(
                f"模型 {spec.id} 属于 provider {spec.provider}，不能注册到 {self.name}"
            )
        if spec.kind.value != self.kind.value:
            raise ConfigurationError(
                f"模型 {spec.id} 类型 {spec.kind} 与 provider {self.kind} 不一致"
            )
        if spec.id in self._models:
            raise ConfigurationError(f"模型重复注册: {self.name}/{spec.id}")
        self._models[spec.id] = spec

    def list_models(self) -> list[ProviderModelSpec]:
        return list(self._models.values())

    def get_spec(self, model_id: str) -> ProviderModelSpec:
        spec = self._models.get(model_id)
        if spec is None:
            raise ConfigurationError(
                f"模型不存在: {self.name}/{model_id}",
                details={"provider": self.name, "model_id": model_id},
            )
        return spec

    async def open(self) -> None:
        """创建客户端/连接池；幂等。默认无资源。"""
        self._opened = True

    async def close(self) -> None:
        """关闭资源；幂等。默认无资源。"""
        self._opened = False

    async def health_check(self) -> ProviderHealth:
        """轻量探活；D-66：只写指标/日志，不自动摘除 provider。"""
        return ProviderHealth(
            provider=self.name,
            status=ProviderHealthStatus.OK if self._opened else ProviderHealthStatus.DOWN,
        )

    @abstractmethod
    def get_model(self, model_id: str) -> ModelHandle:
        """返回模型句柄；provider 不可用时仍返回句柄，调用时抛错误。"""
