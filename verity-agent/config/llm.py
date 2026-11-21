"""LLM 应用层配置。

这里只保留**实际被运行期消费**的字段：

- 应用模型白名单/默认模型/fallback；
- 本地 token 估算编码器。

请求超时由模型配置 ``ModelConfig.timeout_s``（Provider SDK）与
``AgentPolicy.turn_timeout_s``（回合总预算）共同控制；流式空闲/首包超时
目前由 Provider SDK 的传输超时覆盖，不再提供第二套无效开关。
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from .base import ConfigModel

__all__ = ["LLMConfig", "TokenizerConfig"]

# 已删除、但可能残留在旧部署 .env 里的键；启动时丢弃，避免 extra=forbid
# 让历史配置直接无法启动。新部署不要再写这些键。
_LEGACY_ENV_FIELDS = frozenset(
    {
        "request_timeout_s",
        "stream_first_token_timeout_s",
        "stream_total_timeout_s",
        "stream_idle_timeout_s",
        "max_tool_calls_per_response",
        "structured_output",
    }
)


class TokenizerConfig(ConfigModel):
    """本地 token 估算配置。"""

    # tiktoken 编码器名；默认 o200k_base 更适合中英文混合
    encoding: str = Field(default="o200k_base", min_length=1, max_length=64)


class LLMConfig(ConfigModel):
    """LLM 应用层默认策略。"""

    # LLM 可使用的应用模型 id 列表（引用顶层 MODELS[].id）
    model_ids: list[str] = Field(default_factory=list)
    # 默认应用模型 id（ModelRegistry 中的 id），不是厂商模型名
    default_model_id: str | None = None
    # 默认 fallback 应用模型 id 列表；按顺序尝试
    fallback_model_ids: list[str] = Field(default_factory=list)
    # 是否启用模型级 fallback；仅首 token 前允许
    enable_model_fallback: bool = True
    # 本地 token 估算
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)

    @model_validator(mode="before")
    @classmethod
    def _drop_legacy_fields(cls, data: Any) -> Any:
        """丢弃旧的、未实现的 env 字段；只做启动兼容，不再参与行为。"""
        if not isinstance(data, dict):
            return data
        return {
            key: value
            for key, value in data.items()
            if key not in _LEGACY_ENV_FIELDS
        }
