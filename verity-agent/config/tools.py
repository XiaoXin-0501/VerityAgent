"""工具模块配置。

工具定义与执行器由代码注册，不建 PG 表；这里只放**运行期策略**：

- 工具选择的扁平/路由阈值；
- 单批并行上限；
- 入参大小保护；
- 危险工具确认开关（v1 只预留，不实现暂停流程）。

单工具的超时/重试/缓存/幂等/熔断/限流/结果上限属于 ``ToolPolicy``，
跟随 ``ToolSpec`` 一起声明，不在这里重复。内置 ``search_kb`` 是例外：
它的默认召回数与缓存/限流值需要在注册工具时由运维配置，因此放在
``SearchKbToolConfig``，注册后仍落到对应 ``ToolPolicy``。
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from .base import ConfigModel

__all__ = [
    "SearchKbToolConfig",
    "ToolRoutingMode",
    "ToolRoutingConfig",
    "ToolsConfig",
]


class SearchKbToolConfig(ConfigModel):
    """内置 ``search_kb`` 的默认检索与横切策略。

    参数 schema 的下限/上限属于工具契约（改 schema 会影响 LLM），
    这里只放可运维调整的默认值与保护阈值。
    """

    # 未显式指定 top_k 时使用的召回数
    default_top_k: int = Field(default=8, ge=1)
    # 单次调用允许的最大召回数；执行器会把超过的值收紧到该上限
    # 工具 schema 约定 top_k <= 50，配置上限不能突破该契约。
    max_top_k: int = Field(default=50, ge=1, le=50)
    # 检索结果缓存 TTL（秒）
    cache_ttl_s: int = Field(default=60, gt=0)
    # 限流窗口内最大调用次数
    rate_limit_calls: int = Field(default=60, ge=1)
    # 限流窗口长度（秒）
    rate_limit_window_s: int = Field(default=60, gt=0)

    @model_validator(mode="after")
    def _check_top_k(self) -> "SearchKbToolConfig":
        """默认召回数不能超过允许上限。"""
        if self.default_top_k > self.max_top_k:
            raise ValueError("search_kb.default_top_k 不能大于 max_top_k")
        return self


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


class ToolsConfig(ConfigModel):
    """工具模块全局配置。"""

    # 工具选择/路由阈值
    routing: ToolRoutingConfig = Field(default_factory=ToolRoutingConfig)
    # 内置 search_kb 的默认召回数与缓存/限流策略
    search_kb: SearchKbToolConfig = Field(default_factory=SearchKbToolConfig)
    # 单批工具并行上限；Agent 可再用 AgentPolicy.max_parallel_tools 收紧
    max_batch_concurrency: int = Field(default=4, ge=1)
    # 单个工具入参 JSON 的字节上限；防止超大参数打爆校验/日志
    max_argument_bytes: int = Field(default=64 * 1024, gt=0)
    # 危险工具确认开关；v1 仅预留 ToolApprovalRequest，不实现暂停/恢复
    approval_enabled: bool = False
