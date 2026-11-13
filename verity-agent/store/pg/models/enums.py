"""与 PostgreSQL 表字段一一对应的状态枚举与状态机。

所有枚举都是 ``StrEnum``：既能直接作为 SQL 参数，也能 JSON 序列化。
状态机只描述允许的迁移；真正的校验在仓库层。
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "UserRole",
    "UserStatus",
    "KnowledgeBaseStatus",
    "DocumentStatus",
    "ConversationStatus",
    "RunStatus",
    "MessageRole",
    "MessageStatus",
    "UserMessageStatus",
    "SessionFileStatus",
    "IndexingStage",
    "IndexingStatus",
    "DOCUMENT_TRANSITIONS",
    "SESSION_FILE_TRANSITIONS",
    "MAX_USER_FACTS",
]


class UserRole(StrEnum):
    ADMIN = "admin"
    USER = "user"


class UserStatus(StrEnum):
    ACTIVE = "active"
    DISABLED = "disabled"
    LOCKED = "locked"
    # 跨存储删除状态机（先软删 PG，再删 Weaviate/Neo4j/MinIO）
    DELETING = "deleting"
    DELETED = "deleted"


class KnowledgeBaseStatus(StrEnum):
    ACTIVE = "active"
    DISABLED = "disabled"
    DELETING = "deleting"


class DocumentStatus(StrEnum):
    """documents.status 状态机。

    ``completed`` 的语义是“解析 + Weaviate 写入 + Neo4j 建图”三者都成功（D-56），
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
    DELETING = "deleting"
    DELETED = "deleted"


class RunStatus(StrEnum):
    """conversations.current_run_status：单次 Agent 运行的状态（租约投影）。"""

    IDLE = "idle"
    RUNNING = "running"
    WAITING_TOOL = "waiting_tool"
    PAUSED = "paused"
    FAILED = "failed"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


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
    """user_messages 暂存行状态；dead 表示超过重试上限。"""

    PENDING = "pending"
    PROCESSING = "processing"
    FAILED = "failed"
    DEAD = "dead"


class SessionFileStatus(StrEnum):
    UPLOADING = "uploading"
    PROCESSING = "processing"
    AVAILABLE = "available"
    FAILED = "failed"
    DELETING = "deleting"
    DELETED = "deleted"


class IndexingStage(StrEnum):
    """documents.index_stage：入库/删除任务阶段（RAG 文档 §8）。"""

    ACCEPTED = "ACCEPTED"
    FETCHING = "FETCHING"
    PARSING = "PARSING"
    NORMALIZING = "NORMALIZING"
    CLASSIFYING = "CLASSIFYING"
    PROCESSING = "PROCESSING"
    CHUNKING = "CHUNKING"
    DEDUPING = "DEDUPING"
    ENRICHING = "ENRICHING"
    DESCRIBING = "DESCRIBING"
    EMBEDDING = "EMBEDDING"
    VECTOR_UPSERT = "VECTOR_UPSERT"
    GRAPH_UPSERT = "GRAPH_UPSERT"
    COMMITTING = "COMMITTING"
    DONE = "DONE"
    # 删除 job
    DELETING_PG = "DELETING_PG"
    DELETING_VECTOR = "DELETING_VECTOR"
    DELETING_GRAPH = "DELETING_GRAPH"
    DELETING_OBJECT = "DELETING_OBJECT"


class IndexingStatus(StrEnum):
    """IndexingJobView.status 的代码映射；不落 documents 列（D-77）。"""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DEAD = "dead"


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
    DocumentStatus.COMPLETED: frozenset(
        # completed → processing：重切片（REINDEX）先让文档退出检索
        {DocumentStatus.PROCESSING, DocumentStatus.DELETING}
    ),
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
