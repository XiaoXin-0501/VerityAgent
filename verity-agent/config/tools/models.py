"""工具模块配置。

工具静态声明集中在 ``agent/tools/manifest.json``，执行器在 builtin 中实现；
这里只放**运行期策略**：

- 工具选择的扁平/路由阈值；
- 单批并行上限；
- 入参大小保护；
- 危险工具确认开关（v1 只预留，不实现暂停流程）。

``ToolsConfig.policy`` 是**所有工具横切能力的默认值**（超时/重试/缓存/熔断/
限流/幂等/结果上限/并发）；默认值来自 ``policies.default.json``。manifest 中
每个工具直接写需要覆盖的能力字段，装配时与顶层默认值深合并，不使用
policy_preset/profile 打包。
"""

from __future__ import annotations

import json
from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator

from ..base import ConfigModel

__all__ = [
    "SearchKbToolConfig",
    "RecallRoundToolConfig",
    "ToolRoutingMode",
    "ToolRoutingConfig",
    "ToolRetryDefaults",
    "ToolCacheDefaults",
    "ToolCircuitBreakerDefaults",
    "ToolRateLimitDefaults",
    "ToolIdempotencyDefaults",
    "ToolResultLimitDefaults",
    "ToolPolicyDefaults",
    "ToolsConfig",
    "load_policy_defaults",
]


class SearchKbToolConfig(ConfigModel):
    """内置 ``search_kb`` 的默认检索策略。

    参数 schema 的下限/上限属于工具契约（改 schema 会影响 LLM）；
    横切策略（缓存/熔断/限流）由 ``ToolsConfig.policy`` + manifest 覆盖决定。
    """

    # 未显式指定 top_k 时使用的召回数
    default_top_k: int = Field(default=8, ge=1)
    # 单次调用允许的最大召回数；执行器会把超过的值收紧到该上限
    # 工具 schema 约定 top_k <= 50，配置上限不能突破该契约。
    max_top_k: int = Field(default=50, ge=1, le=50)

    @model_validator(mode="after")
    def _check_top_k(self) -> "SearchKbToolConfig":
        """默认召回数不能超过允许上限。"""
        if self.default_top_k > self.max_top_k:
            raise ValueError("search_kb.default_top_k 不能大于 max_top_k")
        return self


class RecallRoundToolConfig(ConfigModel):
    """内置 ``recall_round`` 的启用条件与返回上限。

    该工具按轮次 id 召回“用户提问 + 最终回答”，中间工具调用与过程消息省略。
    manifest 将其声明为 ``AFTER_COMPRESSIONS`` 暴露策略：只有会话摘要成功压缩到
    ``min_compressions`` 次后，才会在本次请求的 ToolSelection 中暴露；
    摘要 topic 的 ``source_round_ids`` 是模型可用的轮次 id 来源。
    """

    # 是否注册轮次召回工具；注册不等于每次运行都暴露
    enabled: bool = True
    # 摘要压缩次数达到该值后向 LLM 暴露工具；0 表示始终暴露
    min_compressions: int = Field(default=2, ge=0)
    # 单次最多召回几个轮次，避免一次拉回过多历史
    max_rounds: int = Field(default=5, ge=1, le=20)
    # 单次返回文本的字符上限；超出截断并提示
    max_chars: int = Field(default=4000, ge=200)


class ToolRetryDefaults(ConfigModel):
    """所有工具默认重试策略。"""

    max_attempts: int = Field(default=1, ge=1)
    backoff_ms: int = Field(default=200, ge=0)
    backoff_multiplier: float = Field(default=2.0, ge=1.0)
    jitter: bool = True
    retry_on: list[str] = Field(
        default_factory=lambda: ["TIMEOUT", "UPSTREAM_ERROR", "RATE_LIMITED"]
    )


class ToolCacheDefaults(ConfigModel):
    """所有工具默认缓存策略。"""

    backend: str = "REDIS"
    enabled: bool = False
    ttl_s: int = Field(default=60, gt=0)
    key_fields: list[str] | None = None
    exclude_fields: list[str] = Field(default_factory=list)
    cache_empty: bool = False


class ToolCircuitBreakerDefaults(ConfigModel):
    """所有工具默认熔断策略。"""

    backend: str = "REDIS"
    enabled: bool = True
    failure_threshold: int = Field(default=5, ge=1)
    failure_rate_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    window_s: int = Field(default=60, gt=0)
    open_cooldown_s: int = Field(default=30, gt=0)
    half_open_max_calls: int = Field(default=3, ge=1)
    success_threshold: int = Field(default=2, ge=1)
    count_errors: list[str] = Field(
        default_factory=lambda: ["TIMEOUT", "UPSTREAM_ERROR"]
    )


class ToolRateLimitDefaults(ConfigModel):
    """所有工具默认限流策略。"""

    backend: str = "REDIS"
    enabled: bool = False
    max_calls: int | None = Field(default=60, ge=1)
    window_s: int = Field(default=60, gt=0)
    scope: str = "TOOL"


class ToolIdempotencyDefaults(ConfigModel):
    """所有工具默认幂等策略。"""

    backend: str = "REDIS"
    mode: str = "NONE"
    ttl_s: int = Field(default=86_400, gt=0)
    key_fields: list[str] | None = None
    exclude_fields: list[str] = Field(
        default_factory=lambda: ["timestamp", "now", "nonce", "request_id"]
    )
    store_result: bool = True
    lock_ttl_s: int = Field(default=60, gt=0)
    on_conflict: str = "RETURN_EXISTING"


class ToolResultLimitDefaults(ConfigModel):
    """所有工具默认结果上限。"""

    max_content_tokens: int = Field(default=4096, ge=1)
    max_data_tokens: int = Field(default=16_384, ge=1)
    max_content_bytes: int = Field(default=64 * 1024, ge=1)
    max_data_bytes: int = Field(default=64 * 1024, ge=1)
    max_files: int = Field(default=10, ge=0)
    max_images: int = Field(default=10, ge=0)
    overflow_enabled: bool = True
    overflow_ttl_s: int | None = Field(default=None, gt=0)


class ToolPolicyDefaults(ConfigModel):
    """所有工具横切能力的顶层默认值。"""

    timeout_s: float = Field(default=30.0, gt=0)
    retry: ToolRetryDefaults = Field(default_factory=ToolRetryDefaults)
    cache: ToolCacheDefaults = Field(default_factory=ToolCacheDefaults)
    circuit_breaker: ToolCircuitBreakerDefaults = Field(
        default_factory=ToolCircuitBreakerDefaults
    )
    rate_limit: ToolRateLimitDefaults = Field(
        default_factory=ToolRateLimitDefaults
    )
    idempotency: ToolIdempotencyDefaults = Field(
        default_factory=ToolIdempotencyDefaults
    )
    result_limit: ToolResultLimitDefaults = Field(
        default_factory=ToolResultLimitDefaults
    )
    max_concurrency: int | None = Field(default=None, ge=1)


class ToolRoutingMode(StrEnum):
    """工具选择模式；与《工具模块数据结构设计》§5.4 对齐。"""

    NONE = "none"
    SINGLE_LEVEL = "single_level"
    MULTI_LEVEL = "multi_level"


class ToolRoutingConfig(ConfigModel):
    """工具选择与路由阈值。"""

    # 路由模式：none 全量；single_level 一级路由；multi_level 预留给后续
    mode: ToolRoutingMode = ToolRoutingMode.NONE
    # 可用工具数低于该值直接全量暴露（FLAT）
    flat_threshold: int = Field(default=15, ge=1)
    # 单次暴露给 LLM 的工具上限；超过时优先走路由
    max_exposed_tools: int = Field(default=30, ge=1)
    # 一级路由组上限
    max_groups: int = Field(default=8, ge=1)
    # 路由工具名前缀；例如 router_kb
    router_prefix: str = "router_"

    @model_validator(mode="after")
    def _check_thresholds(self) -> "ToolRoutingConfig":
        """平铺阈值不能超过单次暴露上限。"""
        if self.flat_threshold > self.max_exposed_tools:
            raise ValueError("flat_threshold 不能大于 max_exposed_tools")
        return self


_POLICY_DEFAULTS_PATH = Path(__file__).with_name("policies.default.json")


@lru_cache(maxsize=1)
def load_policy_defaults() -> ToolPolicyDefaults:
    """读取横切能力全量默认值；文件缺失/非法直接阻断启动。"""
    payload = json.loads(_POLICY_DEFAULTS_PATH.read_text(encoding="utf-8"))
    return ToolPolicyDefaults.model_validate(payload)


class ToolsConfig(ConfigModel):
    """工具模块全局配置。"""

    # 所有工具横切能力的默认值；来自 policies.default.json，manifest 只覆盖差异
    policy: ToolPolicyDefaults = Field(default_factory=load_policy_defaults)
    # 工具选择/路由阈值
    routing: ToolRoutingConfig = Field(default_factory=ToolRoutingConfig)
    # 内置 search_kb 的默认召回策略
    search_kb: SearchKbToolConfig = Field(default_factory=SearchKbToolConfig)
    # 内置 recall_round 的启用条件与返回上限
    recall_round: RecallRoundToolConfig = Field(
        default_factory=RecallRoundToolConfig
    )
    # 单批工具并行上限；Agent 可再用 AgentPolicy.max_parallel_tools 收紧
    max_batch_concurrency: int = Field(default=4, ge=1)
    # 单个工具入参 JSON 的字节上限；防止超大参数打爆校验/日志
    max_argument_bytes: int = Field(default=64 * 1024, gt=0)
    # 危险工具确认开关；v1 仅预留 ToolApprovalRequest，不实现暂停/恢复
    approval_enabled: bool = False
