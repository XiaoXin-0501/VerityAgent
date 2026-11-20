"""工具模块：定义、注册、选择、执行、结果与横切状态。

公开入口：

- ``ToolRegistry``：启动时注册工具/路由组；
- ``ToolCatalog.for_permissions``：生成一次运行的 ``ToolSelection``；
- ``ToolPipeline``：执行单次调用/并行批次，返回 ``ToolResult``；
- ``ToolExecutor``：业务执行器协议。

工具层不写 ``ToolMessage``，由 Agent 主循环完成 ``ToolResult -> ToolMessage``。
"""

from .enums import (
    IdempotencyConflictPolicy,
    IdempotencyMode,
    IdempotencyState,
    RateLimitScope,
    ToolApprovalStatus,
    ToolBatchStatus,
    ToolStateBackend,
    ToolCircuitState,
    ToolExposureMode,
    ToolErrorCategory,
    ToolErrorCode,
    ToolKind,
    ToolResultStatus,
    ToolRoutingMode,
    ToolSelectionMode,
    ToolSideEffect,
)
from .models import (
    CircuitBreakerRecord,
    IdempotencyRecord,
    RateLimitRecord,
    RegisteredTool,
    ToolApprovalDecision,
    ToolApprovalRequest,
    ToolBatch,
    ToolCacheEntry,
    ToolCatalog,
    ToolContext,
    ToolDescriptor,
    ToolError,
    ToolExecutionRecord,
    ToolExecutor,
    ToolExposureContext,
    ToolGroup,
    ToolGroupDescriptor,
    ToolInvocation,
    ToolMetrics,
    ToolResult,
    ToolSelection,
    ToolSpec,
    classify_batch_status,
)
from .pipeline import ToolPipeline
from .policies import (
    CachePolicy,
    CircuitBreakerPolicy,
    IdempotencyPolicy,
    RateLimitPolicy,
    ResultLimitPolicy,
    RetryPolicy,
    ToolPolicy,
    ToolRoutingPolicy,
)
from .registry import ToolRegistry
from .selection import activate_group, build_selection, is_router_tool
from .state import (
    CircuitBreakerBackend,
    IdempotencyBackend,
    OverflowWriter,
    RateLimitBackend,
    ToolCacheStore,
)

__all__ = [
    # registry / selection / pipeline
    "ToolRegistry",
    "ToolCatalog",
    "ToolSelection",
    "ToolPipeline",
    "build_selection",
    "activate_group",
    "is_router_tool",
    # definition
    "ToolSpec",
    "ToolDescriptor",
    "ToolExposureContext",
    "ToolExposureMode",
    "ToolGroup",
    "ToolGroupDescriptor",
    "ToolPolicy",
    "ToolRoutingPolicy",
    "RetryPolicy",
    "CachePolicy",
    "CircuitBreakerPolicy",
    "RateLimitPolicy",
    "IdempotencyPolicy",
    "ResultLimitPolicy",
    # registration / invocation
    "RegisteredTool",
    "ToolExecutor",
    "ToolContext",
    "ToolInvocation",
    "ToolBatch",
    "ToolResult",
    "ToolError",
    "ToolMetrics",
    "ToolExecutionRecord",
    "ToolApprovalRequest",
    "ToolApprovalDecision",
    # state
    "ToolCacheEntry",
    "IdempotencyRecord",
    "CircuitBreakerRecord",
    "RateLimitRecord",
    "ToolCacheStore",
    "IdempotencyBackend",
    "CircuitBreakerBackend",
    "RateLimitBackend",
    "OverflowWriter",
    # enums
    "ToolSideEffect",
    "ToolKind",
    "ToolSelectionMode",
    "ToolRoutingMode",
    "ToolResultStatus",
    "ToolBatchStatus",
    "ToolStateBackend",
    "ToolCircuitState",
    "IdempotencyMode",
    "IdempotencyState",
    "IdempotencyConflictPolicy",
    "RateLimitScope",
    "ToolApprovalStatus",
    "ToolErrorCode",
    "ToolErrorCategory",
    "classify_batch_status",
]
