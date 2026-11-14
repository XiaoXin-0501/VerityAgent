"""Agent 运行态枚举。

只定义 Agent 编排层自己的状态；消息状态、工具状态、LLM finish reason
继续复用 ``agent.message`` / ``agent.tools`` / ``agent.llm`` 的权威枚举。
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "RunStatus",
    "TurnStatus",
    "CancelReason",
    "AgentEventType",
    "AgentErrorCode",
    "AgentWarningCode",
    "AgentAction",
    "AgentDecisionReason",
    "ToolRoundStatus",
]


class RunStatus(StrEnum):
    """一次 Agent run 的状态；终态不回到运行态。"""

    PREFLIGHT = "preflight"
    ACQUIRING_LEASE = "acquiring_lease"
    RUNNING = "running"
    WAITING_TOOL = "waiting_tool"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    CONFLICT = "conflict"


class TurnStatus(StrEnum):
    """一个用户回合的状态；v1 Run 与 Turn 1:1。"""

    CREATED = "created"
    PREPARING_CONTEXT = "preparing_context"
    CALLING_LLM = "calling_llm"
    EXECUTING_TOOLS = "executing_tools"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CancelReason(StrEnum):
    """取消来源；只用于审计/事件，不参与状态机判定。"""

    USER = "user"
    CLIENT_DISCONNECT = "client_disconnect"
    TIMEOUT = "timeout"
    SHUTDOWN = "shutdown"
    ADMIN = "admin"


class AgentEventType(StrEnum):
    """对外 SSE 事件类型；值与《API 与流式契约》一致。"""

    RUN_STARTED = "run.started"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    RUN_CANCELLED = "run.cancelled"
    TURN_STARTED = "turn.started"
    TURN_COMPLETED = "turn.completed"
    TURN_FAILED = "turn.failed"
    TURN_CANCELLED = "turn.cancelled"
    ASSISTANT_STARTED = "assistant.started"
    ASSISTANT_DELTA = "assistant.delta"
    ASSISTANT_TOOL_CALLS = "assistant.tool_calls"
    ASSISTANT_COMPLETED = "assistant.completed"
    TOOL_STARTED = "tool.started"
    TOOL_RESULT = "tool.result"
    WARNING = "warning"
    USAGE = "usage"
    HEARTBEAT = "heartbeat"
    DONE = "done"


class AgentErrorCode(StrEnum):
    """Agent 对外错误码；只暴露可操作/可展示的信息。"""

    LEASE_CONFLICT = "LEASE_CONFLICT"
    RUN_NOT_FOUND = "RUN_NOT_FOUND"
    INVALID_CONFIG = "INVALID_CONFIG"
    DEPENDENCY_UNAVAILABLE = "DEPENDENCY_UNAVAILABLE"
    FILE_NOT_READY = "FILE_NOT_READY"
    STALE_RUN = "STALE_RUN"
    CONTEXT_LENGTH_EXCEEDED = "CONTEXT_LENGTH_EXCEEDED"
    CONTEXT_REBUILD_LIMIT = "CONTEXT_REBUILD_LIMIT"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    MODEL_CAPABILITY_UNSUPPORTED = "MODEL_CAPABILITY_UNSUPPORTED"
    TOOL_ROUND_LIMIT = "TOOL_ROUND_LIMIT"
    TOOL_CALL_LIMIT = "TOOL_CALL_LIMIT"
    TOOL_NO_PROGRESS = "TOOL_NO_PROGRESS"
    TOOL_EXECUTION_FAILED = "TOOL_EXECUTION_FAILED"
    EMPTY_RESPONSE = "EMPTY_RESPONSE"
    MODEL_CONTENT_FILTERED = "MODEL_CONTENT_FILTERED"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"
    LEASE_LOST = "LEASE_LOST"
    PERSISTENCE_ERROR = "PERSISTENCE_ERROR"
    INVALID_STATE = "INVALID_STATE"
    INTERNAL = "INTERNAL"


class AgentWarningCode(StrEnum):
    """可降级告警；不影响 run 继续。"""

    TOOL_ROUND_LIMIT = "TOOL_ROUND_LIMIT"
    TOOL_CALL_LIMIT = "TOOL_CALL_LIMIT"
    TOOL_LOOP_DETECTED = "TOOL_LOOP_DETECTED"
    CONTEXT_TRUNCATED = "CONTEXT_TRUNCATED"
    MODEL_FALLBACK = "MODEL_FALLBACK"
    COMPRESSION_SKIPPED = "COMPRESSION_SKIPPED"
    COMPRESSION_FAILED = "COMPRESSION_FAILED"
    EXTRACTION_DEFERRED = "EXTRACTION_DEFERRED"
    FILE_PARSE_PENDING = "FILE_PARSE_PENDING"
    PERMISSION_FILTERED = "PERMISSION_FILTERED"
    EMPTY_RESPONSE = "EMPTY_RESPONSE"
    USAGE_MISSING = "USAGE_MISSING"
    STREAM_RESYNC = "STREAM_RESYNC"


class AgentAction(StrEnum):
    """LLM 响应后的内部动作。"""

    FINAL_ANSWER = "final_answer"
    EXECUTE_TOOLS = "execute_tools"
    FORCE_FINAL = "force_final"
    RETRY_LLM = "retry_llm"
    ABORT = "abort"


class AgentDecisionReason(StrEnum):
    """决策依据；写入 trace/Hooks，不直接给前端。"""

    STOP = "stop"
    TOOL_CALLS = "tool_calls"
    MAX_ROUNDS = "max_rounds"
    NO_PROGRESS = "no_progress"
    EMPTY_RESPONSE = "empty_response"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"
    CANCELLED = "cancelled"


class ToolRoundStatus(StrEnum):
    """一轮工具调用的执行状态。"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"
