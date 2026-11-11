"""Hooks 模块配置。

Hook 只做观察，不参与缓存/幂等/熔断等控制面。
所有默认值在这里维护，env 由 ``config.Settings`` 映射。
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from .base import ConfigModel

__all__ = [
    "HooksConfig",
    "LoggingHookConfig",
    "TokenUsageHookConfig",
    "MetricsHookConfig",
    "TracingHookConfig",
    "AuditHookConfig",
    "RedactionConfig",
]


class LoggingHookConfig(ConfigModel):
    """结构化日志 Hook。"""

    # 是否启用 LoggingHook
    enabled: bool = True
    # 日志级别：DEBUG / INFO / WARNING / ERROR
    level: str = "INFO"
    # 是否把 payload 摘要附在日志中；默认只记 ID/长度/状态
    include_payload: bool = False
    # payload 摘要最大字节数；超出截断
    max_payload_bytes: int = Field(default=2048, gt=0)
    # 订阅事件；为空表示使用内置关键事件集合
    events: list[str] = Field(default_factory=list)
    # 执行模式；日志默认 background
    mode: str = "background"


class TokenUsageHookConfig(ConfigModel):
    """Token 用量 / 成本 Hook。"""

    # 是否启用 TokenUsageHook
    enabled: bool = True
    # 是否写 Redis 实时用量计数
    redis_counters: bool = True
    # 写哪些计数 scope：turn / conversation / user_day
    counter_scopes: list[str] = Field(
        default_factory=lambda: ["conversation", "user_day"]
    )
    # 各 scope 的 TTL（秒）
    counter_ttl_s: dict[str, int] = Field(
        default_factory=lambda: {
            "turn": 86_400,
            "conversation": 2_592_000,
            "user_day": 172_800,
        }
    )
    # usage:seen:<call_id> 去重 TTL（秒）
    dedupe_ttl_s: int = Field(default=86_400, gt=0)
    # 回合结束是否写 conversations.total_* 投影
    update_conversation_totals: bool = True
    # 模型价格；key 为应用模型 id 或厂商模型名
    pricing: dict[str, dict[str, Any]] = Field(default_factory=dict)


class MetricsHookConfig(ConfigModel):
    """指标 Hook；v1 只写日志 sink。"""

    # 是否启用 MetricsHook
    enabled: bool = True
    # sink：log / noop
    sink: str = "log"
    # 采样率 0~1
    sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    # 指标名前缀
    prefix: str = "verity"


class TracingHookConfig(ConfigModel):
    """追踪 Hook；v1 只支持 none / log。"""

    # 是否启用 TracingHook
    enabled: bool = False
    # exporter：none / log
    exporter: str = "none"
    # 采样率 0~1
    sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)


class AuditHookConfig(ConfigModel):
    """审计 Hook；v1 只写结构化日志。"""

    # 是否启用 AuditHook
    enabled: bool = True
    # sink：log
    sink: str = "log"
    # 是否附带脱敏后的 details
    include_details: bool = False
    # 订阅事件；为空表示审计关键事件集合
    events: list[str] = Field(default_factory=list)


class RedactionConfig(ConfigModel):
    """日志/审计共用脱敏策略。"""

    # 是否启用脱敏；业务执行器仍使用原值
    enabled: bool = True
    # 需要按字段名脱敏的字段（小写比较）
    fields: list[str] = Field(
        default_factory=lambda: [
            "password",
            "token",
            "secret",
            "api_key",
            "authorization",
            "phone",
            "email",
            "id_card",
        ]
    )
    # 文本正则脱敏；手机号/邮箱/身份证等
    patterns: list[str] = Field(default_factory=list)
    # 替换文本
    replacement: str = "***"
    # 单段文本最大长度
    max_text_length: int = Field(default=2048, gt=0)


class HooksConfig(ConfigModel):
    """Hooks 总配置。"""

    # 全局开关；false 时 dispatcher 直接返回
    enabled: bool = True
    # Hook 失败默认不阻断主流程
    fail_open: bool = True
    # 单 Hook 默认超时（秒）
    dispatch_timeout_s: float = Field(default=1.0, gt=0)
    # 后台队列上限
    max_queue_size: int = Field(default=10_000, ge=1)
    # 队列满时丢弃低优先级事件
    drop_on_overflow: bool = True
    # 日志 Hook
    logging: LoggingHookConfig = Field(default_factory=LoggingHookConfig)
    # Token 用量 Hook
    token_usage: TokenUsageHookConfig = Field(default_factory=TokenUsageHookConfig)
    # 指标 Hook
    metrics: MetricsHookConfig = Field(default_factory=MetricsHookConfig)
    # 追踪 Hook
    tracing: TracingHookConfig = Field(default_factory=TracingHookConfig)
    # 审计 Hook
    audit: AuditHookConfig = Field(default_factory=AuditHookConfig)
    # 脱敏策略
    redaction: RedactionConfig = Field(default_factory=RedactionConfig)
