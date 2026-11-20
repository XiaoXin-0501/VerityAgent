"""LLM 应用层配置。"""

from __future__ import annotations

from pydantic import Field

from .base import ConfigModel

__all__ = ["LLMConfig", "StructuredOutputConfig", "TokenizerConfig"]


class TokenizerConfig(ConfigModel):
    """本地 token 估算配置。"""

    # tiktoken 编码器名；默认 o200k_base 更适合中英文混合
    encoding: str = Field(default="o200k_base", min_length=1, max_length=64)


class StructuredOutputConfig(ConfigModel):
    """结构化输出的默认策略（D-44）。"""

    # 优先使用模型原生 JSON Schema；不支持时由 LLM 层决定降级
    prefer_native_schema: bool = True
    # 原生 schema 不可用时是否允许退到 JSON object mode
    allow_json_object_fallback: bool = True
    # 连 JSON mode 都没有时是否允许纯提示词约束（解析失败风险高）
    allow_text_fallback: bool = False
    # 结构化输出解析失败后的最多修复次数
    max_repair_attempts: int = Field(default=1, ge=0, le=3)


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
    # 非流式请求总超时（秒）
    request_timeout_s: float = Field(default=120.0, gt=0)
    # 流式首 token 超时（秒）；超时可 fallback
    stream_first_token_timeout_s: float = Field(default=60.0, gt=0)
    # 流式总时长上限（秒）
    stream_total_timeout_s: float = Field(default=600.0, gt=0)
    # 流式两个事件之间的最大空闲（秒）
    stream_idle_timeout_s: float = Field(default=60.0, gt=0)
    # 单次模型响应允许的最大工具调用数
    max_tool_calls_per_response: int = Field(default=16, ge=1)
    # 结构化输出策略
    structured_output: StructuredOutputConfig = Field(
        default_factory=StructuredOutputConfig
    )
    # 本地 token 估算
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
