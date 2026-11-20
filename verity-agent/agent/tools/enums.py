"""工具模块枚举。

错误码/错误大类由 ``exception.tool`` 统一定义，这里重新导出，避免工具模块
直接依赖异常实现文件；路由模式属于部署配置，由 ``config.tools`` 定义。
"""

from __future__ import annotations

from enum import StrEnum

from config.tools import ToolRoutingMode
from exception.tool import (
    TOOL_ERROR_CATEGORIES,
    TOOL_ERROR_RETRYABLE,
    ToolErrorCategory,
    ToolErrorCode,
)

__all__ = [
    "ToolSideEffect",
    "ToolKind",
    "ToolExposureMode",
    "ToolSelectionMode",
    "ToolResultStatus",
    "ToolBatchStatus",
    "ToolStateBackend",
    "ToolCircuitState",
    "IdempotencyMode",
    "IdempotencyState",
    "IdempotencyConflictPolicy",
    "RateLimitScope",
    "ToolApprovalStatus",
    "ToolRoutingMode",
    "ToolErrorCode",
    "ToolErrorCategory",
    "TOOL_ERROR_CATEGORIES",
    "TOOL_ERROR_RETRYABLE",
]


class ToolSideEffect(StrEnum):
    """工具副作用等级；v1 只自动执行 READ。"""

    READ = "READ"
    WRITE = "WRITE"
    DESTRUCTIVE = "DESTRUCTIVE"


class ToolKind(StrEnum):
    """普通执行工具 / 路由工具。"""

    EXECUTABLE = "EXECUTABLE"
    ROUTER = "ROUTER"


class ToolExposureMode(StrEnum):
    """工具在单次请求中的暴露策略。

    注册表常驻全部已装配工具；每次请求按本策略 + 用户权限生成
    ``ToolSelection``，不动态修改注册表。
    """

    # 登录且有静态权限即可暴露
    ALWAYS = "ALWAYS"
    # 仅当本次查询规划判定需要 RAG 召回时暴露（如片段冲突工具）
    RAG_REQUIRED = "RAG_REQUIRED"
    # 会话摘要压缩次数达到工具声明阈值后暴露（如 recall_round）
    AFTER_COMPRESSIONS = "AFTER_COMPRESSIONS"


class ToolSelectionMode(StrEnum):
    """本次运行的工具暴露方式。"""

    FLAT = "FLAT"
    ROUTED = "ROUTED"


class ToolResultStatus(StrEnum):
    """工具结果终态；超时归类为 FAILED + TIMEOUT。"""

    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ToolBatchStatus(StrEnum):
    """并行批次状态。"""

    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ToolStateBackend(StrEnum):
    """横切状态后端；缓存/熔断/限流/幂等 v1 固定 Redis。"""

    REDIS = "REDIS"


class ToolCircuitState(StrEnum):
    """熔断状态机。"""

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class IdempotencyMode(StrEnum):
    """幂等模式。"""

    NONE = "NONE"
    OPTIONAL = "OPTIONAL"
    REQUIRED = "REQUIRED"


class IdempotencyState(StrEnum):
    """幂等记录状态；IN_PROGRESS 锁过期后按 UNKNOWN 处理。"""

    IN_PROGRESS = "IN_PROGRESS"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


class IdempotencyConflictPolicy(StrEnum):
    """幂等冲突处理策略。"""

    RETURN_EXISTING = "RETURN_EXISTING"
    REJECT = "REJECT"
    WAIT = "WAIT"


class RateLimitScope(StrEnum):
    """限流维度；v1 固定按工具。"""

    TOOL = "TOOL"
    USER = "USER"
    USER_TOOL = "USER_TOOL"
    GLOBAL = "GLOBAL"


class ToolApprovalStatus(StrEnum):
    """危险工具确认状态（v1 只预留模型）。"""

    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
