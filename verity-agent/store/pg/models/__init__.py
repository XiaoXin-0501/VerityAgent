"""PostgreSQL 行模型与 JSONB 协议。"""

from .base import DBJsonModel, DBModel, utcnow
from .conversation import (
    Conversation,
    ConversationMessage,
    ConversationSummary,
    SessionSummary,
    SummaryConstraint,
    SummaryFact,
    SummaryTopic,
)
from .enums import (
    DOCUMENT_TRANSITIONS,
    MAX_USER_FACTS,
    SESSION_FILE_TRANSITIONS,
    ConversationStatus,
    DocumentStatus,
    IndexingStage,
    IndexingStatus,
    KnowledgeBaseStatus,
    MessageRole,
    MessageStatus,
    RunStatus,
    SessionFileStatus,
    UserMessageStatus,
    UserRole,
    UserStatus,
)
from .knowledge import Document, KnowledgeBase
from .memory import (
    MemoryChange,
    MemoryHistorySections,
    MemorySection,
    MemoryUserSections,
    MemoryValue,
    UserMemory,
    UserMessage,
    empty_memory_value,
)
from .session_file import SessionFile
from .user import Permission, User, UserPermission

__all__ = [
    # 基类
    "DBModel",
    "DBJsonModel",
    "utcnow",
    # 用户
    "User",
    "Permission",
    "UserPermission",
    # 知识库
    "KnowledgeBase",
    "Document",
    # 会话
    "Conversation",
    "ConversationMessage",
    "ConversationSummary",
    "SessionSummary",
    "SummaryTopic",
    "SummaryFact",
    "SummaryConstraint",
    # 记忆
    "UserMemory",
    "UserMessage",
    "MemoryValue",
    "MemorySection",
    "MemoryUserSections",
    "MemoryHistorySections",
    "MemoryChange",
    "empty_memory_value",
    # 会话文件
    "SessionFile",
    # 枚举与状态机
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
