"""请求/运行级日志上下文。

HTTP 请求、Agent run、RAG 入库分别把关键 id 写入 ContextVar；日志 Filter
读取后生成一行紧凑上下文，便于把同一次问答或同一次入库的日志串起来。
"""

from __future__ import annotations

from contextvars import ContextVar, Token

__all__ = [
    "bind_index_context",
    "bind_run_context",
    "current_log_context",
    "current_request_id",
    "reset_index_context",
    "reset_request_id",
    "reset_run_context",
    "set_request_id",
]

_request_id: ContextVar[str | None] = ContextVar("verity_request_id", default=None)
_run_id: ContextVar[str | None] = ContextVar("verity_run_id", default=None)
_conversation_id: ContextVar[str | None] = ContextVar(
    "verity_conversation_id", default=None
)
_doc_id: ContextVar[str | None] = ContextVar("verity_doc_id", default=None)
_kb_id: ContextVar[str | None] = ContextVar("verity_kb_id", default=None)


def _short(value: str | None) -> str:
    """日志里只显示 8 位 id；完整 id 仍在 trace/DB 中。"""
    if not value:
        return "-"
    text = value[4:] if value.startswith("req_") else value
    return text[:8]


def set_request_id(value: str) -> Token[str | None]:
    """设置当前异步上下文的请求 id；返回用于恢复的 token。"""
    return _request_id.set(value)


def reset_request_id(token: Token[str | None]) -> None:
    """恢复请求入口前的上下文；必须在请求结束时调用。"""
    _request_id.reset(token)


def bind_run_context(
    *,
    run_id: str,
    conversation_id: str,
) -> tuple[Token[str | None], Token[str | None]]:
    """绑定一次 Agent run 的 run/会话 id。"""
    return (
        _run_id.set(run_id),
        _conversation_id.set(conversation_id),
    )


def reset_run_context(
    tokens: tuple[Token[str | None], Token[str | None]],
) -> None:
    """恢复 Agent run 前的上下文。"""
    _conversation_id.reset(tokens[1])
    _run_id.reset(tokens[0])


def bind_index_context(
    *,
    doc_id: str,
    kb_id: str,
) -> tuple[Token[str | None], Token[str | None]]:
    """绑定一次 RAG 入库任务的 doc/kb id。"""
    return (
        _doc_id.set(doc_id),
        _kb_id.set(kb_id),
    )


def reset_index_context(
    tokens: tuple[Token[str | None], Token[str | None]],
) -> None:
    """恢复入库任务前的上下文。"""
    _kb_id.reset(tokens[1])
    _doc_id.reset(tokens[0])


def current_request_id() -> str:
    """当前请求 id；没有请求上下文时返回 ``-``。"""
    return _request_id.get() or "-"


def current_log_context() -> str:
    """生成 ``req=... run=... conv=... doc=... kb=...`` 紧凑上下文。"""
    parts: list[str] = []
    if _request_id.get():
        parts.append(f"req={_short(_request_id.get())}")
    if _run_id.get():
        parts.append(f"run={_short(_run_id.get())}")
    if _conversation_id.get():
        parts.append(f"conv={_short(_conversation_id.get())}")
    if _doc_id.get():
        parts.append(f"doc={_short(_doc_id.get())}")
    if _kb_id.get():
        parts.append(f"kb={_short(_kb_id.get())}")
    return " ".join(parts) or "-"
