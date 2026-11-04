"""与 PostgreSQL 表字段一一对应的状态枚举与状态机。

所有枚举都是 ``str`` 子类，既方便直接写进 SQL，也便于 JSON 序列化。
状态机只描述"允许的迁移"，真正的校验在仓库层（``repositories/base.py``）。
"""

from __future__ import annotations

from enum import StrEnum


class UserRole(StrEnum):
    ADMIN = "admin"
    USER = "user"


class UserStatus(StrEnum):
    ACTIVE = "active"
    DISABLED = "disabled"
    LOCKED = "locked"


class KnowledgeBaseStatus(StrEnum):
    ACTIVE = "active"
    DISABLED = "disabled"
    DELETING = "deleting"


class DocumentStatus(StrEnum):
    """documents.status 状态机。

    ``completed`` 的语义是"解析 + Weaviate 写入 + Neo4j 引用建完"三者都成功，
    解析完不能直接置 completed。
    """

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DELETING = "deleting"
    DELETED = "deleted"


class ConversationStatus(StrEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"


class RunStatus(StrEnum):
    """conversations.current_run_status：单次 Agent 运行的状态。"""

    IDLE = "idle"
    RUNNING = "running"
    WAITING_TOOL = "waiting_tool"
    PAUSED = "paused"
    FAILED = "failed"
    COMPLETED = "completed"


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


class MessageStatus(StrEnum):
    STREAMING = "streaming"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class UserMessageStatus(StrEnum):
    """user_messages 暂存行状态：worker 领走 pending 后置 processing。"""

    PENDING = "pending"
    PROCESSING = "processing"
    FAILED = "failed"


class SessionFileStatus(StrEnum):
    UPLOADING = "uploading"
    PROCESSING = "processing"
    AVAILABLE = "available"
    FAILED = "failed"
    DELETING = "deleting"
    DELETED = "deleted"


# —— 允许的状态迁移 ——

DOCUMENT_TRANSITIONS: dict[DocumentStatus, frozenset[DocumentStatus]] = {
    # 任何未删除状态都允许发起删除（用户随时可以删文档）
    DocumentStatus.PENDING: frozenset(
        {DocumentStatus.PROCESSING, DocumentStatus.FAILED, DocumentStatus.DELETING}
    ),
    DocumentStatus.PROCESSING: frozenset(
        {DocumentStatus.COMPLETED, DocumentStatus.FAILED, DocumentStatus.DELETING}
    ),
    DocumentStatus.FAILED: frozenset(
        {DocumentStatus.PROCESSING, DocumentStatus.DELETING}
    ),
    DocumentStatus.COMPLETED: frozenset({DocumentStatus.DELETING}),
    DocumentStatus.DELETING: frozenset({DocumentStatus.DELETED}),
    DocumentStatus.DELETED: frozenset(),
}

SESSION_FILE_TRANSITIONS: dict[SessionFileStatus, frozenset[SessionFileStatus]] = {
    SessionFileStatus.UPLOADING: frozenset(
        {
            SessionFileStatus.PROCESSING,
            SessionFileStatus.FAILED,
            SessionFileStatus.DELETING,
        }
    ),
    SessionFileStatus.PROCESSING: frozenset(
        {
            SessionFileStatus.AVAILABLE,
            SessionFileStatus.FAILED,
            SessionFileStatus.DELETING,
        }
    ),
    SessionFileStatus.FAILED: frozenset(
        {SessionFileStatus.PROCESSING, SessionFileStatus.DELETING}
    ),
    SessionFileStatus.AVAILABLE: frozenset({SessionFileStatus.DELETING}),
    SessionFileStatus.DELETING: frozenset({SessionFileStatus.DELETED}),
    SessionFileStatus.DELETED: frozenset(),
}

# 用户长期事实硬上限（与 user_memory.fact_count 的 CHECK 约束保持一致）
MAX_USER_FACTS = 150
