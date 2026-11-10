"""模型配置：模型与连接信息放在同一个列表里。

Provider 不再是单独的 env 配置块；每个模型自己声明：

- 用哪个厂商/适配器（``provider`` + ``api_style``）；
- 厂商模型名（``model``）；
- API 地址和密钥（``base_url`` / ``api_key_env`` / ``api_key``）；
- 能力、价格、默认生成参数。

这样同一厂商可以有多个模型，也可以让不同模型指向不同网关地址。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from .provider import ChatApiStyle, ProviderKind
from .secrets import resolve_api_key

__all__ = ["ModelConfig"]


class ModelConfig(BaseModel):
    """一个可调用模型的完整配置（连接 + 模型 + 参数）。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # 应用模型 id；LLM / RAG 配置引用它，全局唯一
    id: str = Field(min_length=1)
    # 厂商/适配器名：openai / anthropic；embedding/rerank 可扩展 bge/cohere
    provider: str = Field(min_length=1)
    # 厂商实际模型名（调用 API 时发送）
    model: str = Field(min_length=1)
    # 模型类型：chat / embedding / rerank
    kind: ProviderKind = ProviderKind.CHAT
    # 可选 wire 协议；chat 为空时按 provider 自动选
    api_style: ChatApiStyle | None = None
    # API 基地址；每个模型可以指向不同网关
    base_url: str = ""
    # 密钥环境变量名
    api_key_env: str = ""
    # 直接密钥；优先于 api_key_env，不建议提交仓库
    api_key: SecretStr = SecretStr("")
    # 单次请求超时（秒）
    timeout_s: float = Field(default=60.0, gt=0)
    # transport 重试次数；不含 LLM 模型 fallback
    max_retries: int = Field(default=2, ge=0, le=10)
    # transport 重试退避基数（毫秒）
    retry_backoff_ms: int = Field(default=200, ge=0)
    # 额外 HTTP header；不得放密钥
    headers: dict[str, str] = Field(default_factory=dict)
    # 是否启用该模型
    enabled: bool = True
    # 能力：上下文/输出/工具/视觉/维度等
    capabilities: dict[str, Any] = Field(default_factory=dict)
    # 原始价格；LLM 层映射为 ModelPricing
    pricing: dict[str, Any] | None = None
    # chat 默认生成参数：temperature/top_p/max_output_tokens/stop 等
    defaults: dict[str, Any] = Field(default_factory=dict)
    # 扩展字段
    metadata: dict[str, Any] = Field(default_factory=dict)

    def resolve_api_key(
        self, environ: dict[str, str] | None = None
    ) -> str | None:
        return resolve_api_key(self.api_key, self.api_key_env, environ=environ)
