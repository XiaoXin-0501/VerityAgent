"""Provider 配置模型（由 config 包统一管理）。

所有 provider 都通过配置声明，不在代码里写死地址、模型或密钥；密钥只存
环境变量名。``config.Settings`` 是唯一的 env → config 映射入口。

模型统一用 ``models`` 列表声明，不再同时维护 ``default_model`` 和
``metadata.models`` 两套写法。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from .secrets import resolve_api_key

__all__ = [
    "ProviderKind",
    "ChatApiStyle",
    "ProviderConfig",
    "ChatProviderConfig",
    "EmbeddingProviderConfig",
    "RerankProviderConfig",
]

_DEFAULT_TIMEOUT_S = 60.0
_DEFAULT_RETRIES = 2
_DEFAULT_BACKOFF_MS = 200


class ProviderKind(StrEnum):
    """Provider 配置类型；与分析层的 ProviderModelKind 一一映射。"""

    CHAT = "chat"
    EMBEDDING = "embedding"
    RERANK = "rerank"


class ChatApiStyle(StrEnum):
    """Chat provider 的 wire 协议风格。"""

    OPENAI_CHAT = "openai_chat"                 # OpenAI Chat Completions 兼容
    OPENAI_RESPONSES = "openai_responses"       # OpenAI Responses API
    ANTHROPIC_MESSAGES = "anthropic_messages"   # Anthropic Messages API


class ProviderConfig(BaseModel):
    """所有 provider 的公共配置。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # provider 唯一名；注册表主键
    name: str = Field(min_length=1)
    # provider 类型：chat / embedding / rerank
    kind: ProviderKind
    # API 基地址；不同厂商路径不同，由 adapter 拼接
    base_url: str = ""
    # 只存环境变量/secret ref 名，不存原始 key
    api_key_env: str = ""
    # 可直接提供密钥（优先于 api_key_env）；不建议提交到仓库
    api_key: SecretStr = SecretStr("")
    # 单次请求超时（秒）
    timeout_s: float = Field(default=_DEFAULT_TIMEOUT_S, gt=0)
    # transport 层重试次数；不含 LLM 层模型 fallback
    max_retries: int = Field(default=_DEFAULT_RETRIES, ge=0, le=10)
    # transport 重试退避基数（毫秒）
    retry_backoff_ms: int = Field(default=_DEFAULT_BACKOFF_MS, ge=0)
    # 额外 HTTP header；不得放密钥
    headers: dict[str, str] = Field(default_factory=dict)
    # 是否启用；禁用时仍注册但调用报 UNAVAILABLE
    enabled: bool = True
    # 厂商/适配器扩展字段：variant / send_input_type / documents_format 等
    metadata: dict[str, Any] = Field(default_factory=dict)

    def resolve_api_key(self, environ: dict[str, str] | None = None) -> str | None:
        return resolve_api_key(self.api_key, self.api_key_env, environ=environ)

class ChatProviderConfig(ProviderConfig):
    """Chat provider 专有配置。"""

    # 类型固定为 chat
    kind: Literal[ProviderKind.CHAT] = ProviderKind.CHAT
    # wire 协议风格；决定使用哪个 adapter
    api_style: ChatApiStyle = ChatApiStyle.OPENAI_CHAT


class EmbeddingProviderConfig(ProviderConfig):
    """Embedding provider 专有配置。

    维度、批大小、归一化能力统一写在 ``models[].capabilities``，
    不再在 provider 级重复。
    """

    # 类型固定为 embedding
    kind: Literal[ProviderKind.EMBEDDING] = ProviderKind.EMBEDDING


class RerankProviderConfig(ProviderConfig):
    """Rerank provider 专有配置。

    文档数上限、批大小统一写在 ``models[].capabilities``。
    """

    # 类型固定为 rerank
    kind: Literal[ProviderKind.RERANK] = ProviderKind.RERANK
