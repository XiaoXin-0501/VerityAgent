"""工具策略模型。

策略只描述**横切能力**，不参与 LLM 描述，也不包含执行器：

- 单工具策略由 ``ToolSpec.policy`` 声明；
- 选择/路由策略由 ``config.tools`` 映射为 ``ToolRoutingPolicy``；
- v1 写工具默认不自动重试，只有幂等 REQUIRED + supports_reconcile 才允许。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from config.tools import ToolRoutingConfig, ToolRoutingMode
from exception.tool import ToolErrorCode

from .enums import (
    IdempotencyConflictPolicy,
    IdempotencyMode,
    RateLimitScope,
    ToolStateBackend,
)

__all__ = [
    "RetryPolicy",
    "CachePolicy",
    "CircuitBreakerPolicy",
    "RateLimitPolicy",
    "IdempotencyPolicy",
    "ResultLimitPolicy",
    "ToolPolicy",
    "ToolRoutingPolicy",
]


class _Policy(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, populate_by_name=True
    )


class RetryPolicy(_Policy):
    """执行失败后的重试策略。"""

    # 含首次的总尝试次数；1 表示不重试
    max_attempts: int = Field(default=1, ge=1)
    # 首次退避（毫秒）
    backoff_ms: int = Field(default=200, ge=0)
    # 指数退避倍数
    backoff_multiplier: float = Field(default=2.0, ge=1.0)
    # 是否加随机抖动，避免并发重试同时打下游
    jitter: bool = True
    # 只对这些错误重试；其余失败直接返回 LLM
    retry_on: frozenset[ToolErrorCode] = Field(
        default_factory=lambda: frozenset(
            {
                ToolErrorCode.TIMEOUT,
                ToolErrorCode.UPSTREAM_ERROR,
                ToolErrorCode.RATE_LIMITED,
            }
        )
    )


class CachePolicy(_Policy):
    """读工具短期缓存策略；固定 Redis。"""

    # 缓存后端；v1 固定 REDIS
    backend: ToolStateBackend = ToolStateBackend.REDIS
    # 是否启用缓存；默认关闭，按工具显式打开
    enabled: bool = False
    # 缓存 TTL（秒）
    ttl_s: int = Field(default=60, gt=0)
    # 参与 key 的参数；null 表示全部参数
    key_fields: list[str] | None = None
    # 不参与 key 的字段（时间戳、nonce 等）
    exclude_fields: list[str] = Field(default_factory=list)
    # 是否缓存空结果
    cache_empty: bool = False


class CircuitBreakerPolicy(_Policy):
    """按工具维度的熔断策略。"""

    # 状态后端；v1 固定 REDIS
    backend: ToolStateBackend = ToolStateBackend.REDIS
    # 是否启用
    enabled: bool = True
    # 窗口内失败次数阈值
    failure_threshold: int = Field(default=5, ge=1)
    # 失败率阈值；null 表示只看次数
    failure_rate_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    # 统计窗口（秒）
    window_s: int = Field(default=60, gt=0)
    # OPEN 后多久进入 HALF_OPEN
    open_cooldown_s: int = Field(default=30, gt=0)
    # HALF_OPEN 允许的试探调用数
    half_open_max_calls: int = Field(default=3, ge=1)
    # HALF_OPEN 连续成功多少次回 CLOSED
    success_threshold: int = Field(default=2, ge=1)
    # 哪些错误计入失败
    count_errors: frozenset[ToolErrorCode] = Field(
        default_factory=lambda: frozenset(
            {ToolErrorCode.TIMEOUT, ToolErrorCode.UPSTREAM_ERROR}
        )
    )


class RateLimitPolicy(_Policy):
    """按工具维度的限流策略。"""

    # 计数后端；v1 固定 REDIS
    backend: ToolStateBackend = ToolStateBackend.REDIS
    # 是否启用
    enabled: bool = False
    # 窗口内最大调用数；启用时必填
    max_calls: int | None = Field(default=None, ge=1)
    # 窗口长度（秒）
    window_s: int = Field(default=60, gt=0)
    # 限流维度；v1 固定 TOOL
    scope: RateLimitScope = RateLimitScope.TOOL

    @model_validator(mode="after")
    def _max_calls_required(self) -> "RateLimitPolicy":
        """启用限流时必须配置 max_calls。"""
        if self.enabled and self.max_calls is None:
            raise ValueError("启用 rate_limit 时 max_calls 必填")
        return self


class IdempotencyPolicy(_Policy):
    """写工具幂等策略；固定 Redis 记录 + 锁。"""

    backend: ToolStateBackend = ToolStateBackend.REDIS
    # NONE / OPTIONAL / REQUIRED
    mode: IdempotencyMode = IdempotencyMode.NONE
    # 幂等记录保留时间（秒）
    ttl_s: int = Field(default=86_400, gt=0)
    # 参与幂等 key 的业务参数；null 表示全部参数
    key_fields: list[str] | None = None
    # 不参与 key 的非确定性字段
    exclude_fields: list[str] = Field(
        default_factory=lambda: ["timestamp", "now", "nonce", "request_id"]
    )
    # 是否保存结果供重放
    store_result: bool = True
    # IN_PROGRESS 锁的过期时间（秒）
    lock_ttl_s: int = Field(default=60, gt=0)
    # 冲突处理策略
    on_conflict: IdempotencyConflictPolicy = (
        IdempotencyConflictPolicy.RETURN_EXISTING
    )


class ResultLimitPolicy(_Policy):
    """结果上限；token 为主口径，字节只做安全兜底。"""

    # 给 LLM 的文本 token 上限
    max_content_tokens: int = Field(default=4096, ge=1)
    # 结构化数据 token 上限
    max_data_tokens: int = Field(default=16_384, ge=1)
    # 文本字节硬上限
    max_content_bytes: int = Field(default=64 * 1024, ge=1)
    # 结构化数据字节硬上限
    max_data_bytes: int = Field(default=64 * 1024, ge=1)
    # 文件/图片数量上限
    max_files: int = Field(default=10, ge=0)
    max_images: int = Field(default=10, ge=0)
    # 超限时是否写 MinIO
    overflow_enabled: bool = True
    # 溢出对象 TTL；null 表示跟随会话
    overflow_ttl_s: int | None = Field(default=None, gt=0)


class ToolPolicy(_Policy):
    """单个工具的横切策略集合。"""

    # 单次执行超时（秒）
    timeout_s: float = Field(default=30.0, gt=0)
    # 重试
    retry: RetryPolicy = Field(default_factory=RetryPolicy)
    # 缓存
    cache: CachePolicy = Field(default_factory=CachePolicy)
    # 熔断
    circuit_breaker: CircuitBreakerPolicy = Field(
        default_factory=CircuitBreakerPolicy
    )
    # 限流
    rate_limit: RateLimitPolicy = Field(default_factory=RateLimitPolicy)
    # 幂等
    idempotency: IdempotencyPolicy = Field(default_factory=IdempotencyPolicy)
    # 结果上限
    result_limit: ResultLimitPolicy = Field(default_factory=ResultLimitPolicy)
    # 同工具/同用户并发上限；null 表示不额外限制
    max_concurrency: int | None = Field(default=None, ge=1)


class ToolRoutingPolicy(_Policy):
    """工具选择/路由策略；由 ``config.tools.ToolRoutingConfig`` 映射。"""

    # none / single_level / multi_level
    mode: ToolRoutingMode = ToolRoutingMode.NONE
    # 低于该数量直接全量暴露
    flat_threshold: int = Field(default=15, ge=1)
    # 单次暴露上限
    max_exposed_tools: int = Field(default=30, ge=1)
    # 一级路由组上限
    max_groups: int = Field(default=8, ge=1)
    # 路由工具名前缀
    router_prefix: str = "router_"

    @classmethod
    def from_config(cls, config: ToolRoutingConfig) -> "ToolRoutingPolicy":
        """从顶层路由配置构造路由策略。"""
        return cls(
            mode=config.mode,
            flat_threshold=config.flat_threshold,
            max_exposed_tools=config.max_exposed_tools,
            max_groups=config.max_groups,
            router_prefix=config.router_prefix,
        )
