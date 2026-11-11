"""Hooks 枚举：事件、来源、执行模式、失败策略、状态、用量维度。"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "HookEvent",
    "HookSource",
    "HookExecutionMode",
    "HookFailurePolicy",
    "HookStatus",
    "UsageKind",
    "UsageCounterScope",
    "MetricKind",
    "SpanStatus",
    "AuditResult",
]


class HookEvent(StrEnum):
    # 运行
    RUN_START = "run.start"
    RUN_END = "run.end"
    RUN_ERROR = "run.error"
    # 回合
    TURN_START = "turn.start"
    TURN_END = "turn.end"
    TURN_PAUSE = "turn.pause"
    TURN_RESUME = "turn.resume"
    TURN_ERROR = "turn.error"
    # LLM
    LLM_REQUEST_START = "llm.request.start"
    LLM_REQUEST_END = "llm.request.end"
    LLM_USAGE = "llm.usage"
    LLM_STREAM_START = "llm.stream.start"
    LLM_STREAM_CHUNK = "llm.stream.chunk"
    LLM_STREAM_END = "llm.stream.end"
    LLM_ERROR = "llm.error"
    LLM_RETRY = "llm.retry"
    # 工具
    TOOL_SELECTION = "tool.selection"
    TOOL_CALL_START = "tool.call.start"
    TOOL_VALIDATION_FAILED = "tool.validation.failed"
    TOOL_PERMISSION_DENIED = "tool.permission.denied"
    TOOL_CACHE_HIT = "tool.cache.hit"
    TOOL_IDEMPOTENT_REPLAY = "tool.idempotent.replay"
    TOOL_CIRCUIT_OPEN = "tool.circuit.open"
    TOOL_RATE_LIMITED = "tool.rate.limited"
    TOOL_EXECUTION_START = "tool.execution.start"
    TOOL_EXECUTION_END = "tool.execution.end"
    TOOL_RESULT = "tool.result"
    TOOL_ERROR = "tool.error"
    # 消息
    MESSAGE_APPENDED = "message.appended"
    MESSAGE_FINALIZED = "message.finalized"
    MESSAGE_DECODE_ERROR = "message.decode.error"
    # 记忆
    MEMORY_LOAD = "memory.load"
    MEMORY_UPDATE = "memory.update"
    SUMMARY_COMPRESS_START = "summary.compress.start"
    SUMMARY_COMPRESS_END = "summary.compress.end"
    SUMMARY_COMPRESS_WARNING = "summary.compress.warning"
    LONG_TERM_EXTRACT_START = "long_term.extract.start"
    LONG_TERM_EXTRACT_END = "long_term.extract.end"
    # 上下文
    CONTEXT_ASSEMBLE_START = "context.assemble.start"
    CONTEXT_ASSEMBLE_END = "context.assemble.end"
    CONTEXT_SECTION_DROPPED = "context.section.dropped"
    CONTEXT_BUDGET_EXCEEDED = "context.budget.exceeded"
    # RAG
    RAG_INDEX_START = "rag.index.start"
    RAG_INDEX_STAGE = "rag.index.stage"
    RAG_INDEX_WARNING = "rag.index.warning"
    RAG_INDEX_END = "rag.index.end"
    RAG_INDEX_COMMIT = "rag.index.commit"
    RAG_RETRIEVAL_START = "rag.retrieval.start"
    RAG_RETRIEVAL_END = "rag.retrieval.end"
    RAG_RERANK_END = "rag.rerank.end"
    RAG_GRAPH_END = "rag.graph.end"
    RAG_CACHE_HIT = "rag.cache.hit"
    RAG_CITATION = "rag.citation"
    RAG_CONTEXT_INJECTED = "rag.context.injected"
    # Embedding / Rerank
    EMBEDDING_REQUEST_START = "embedding.request.start"
    EMBEDDING_REQUEST_END = "embedding.request.end"
    EMBEDDING_USAGE = "embedding.usage"
    RERANK_REQUEST_START = "rerank.request.start"
    RERANK_REQUEST_END = "rerank.request.end"
    RERANK_USAGE = "rerank.usage"
    # 文件
    FILE_UPLOADED = "file.uploaded"
    FILE_PARSED = "file.parsed"
    FILE_PARSE_FAILED = "file.parse.failed"
    FILE_INJECTED = "file.injected"
    # 删除
    DELETION_START = "deletion.start"
    DELETION_STEP = "deletion.step"
    DELETION_END = "deletion.end"
    DELETION_FAILED = "deletion.failed"
    # Prompt
    PROMPT_ASSEMBLED = "prompt.assembled"
    PROMPT_TOKEN_ESTIMATED = "prompt.token.estimated"
    PROMPT_AUDIT = "prompt.audit"


class HookSource(StrEnum):
    AGENT = "agent"
    LLM = "llm"
    TOOL = "tool"
    MEMORY = "memory"
    RAG = "rag"
    FILE = "file"
    MESSAGE = "message"
    EMBEDDING = "embedding"
    RERANK = "rerank"
    SYSTEM = "system"


class HookExecutionMode(StrEnum):
    INLINE = "inline"
    INLINE_PARALLEL = "inline_parallel"
    BACKGROUND = "background"


class HookFailurePolicy(StrEnum):
    FAIL_OPEN = "fail_open"
    FAIL_CLOSED = "fail_closed"
    IGNORE = "ignore"


class HookStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class UsageKind(StrEnum):
    CHAT = "chat"
    EMBEDDING = "embedding"
    RERANK = "rerank"


class UsageCounterScope(StrEnum):
    TURN = "turn"
    CONVERSATION = "conversation"
    USER_DAY = "user_day"


class MetricKind(StrEnum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


class SpanStatus(StrEnum):
    OK = "ok"
    ERROR = "error"
    CANCELLED = "cancelled"


class AuditResult(StrEnum):
    SUCCESS = "success"
    DENIED = "denied"
    ERROR = "error"
