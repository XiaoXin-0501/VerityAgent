"""PostgreSQL 仓库集合。

跨表事务示例（设计文档 §3 关键约束 3）：::

    from store.pg import get_pg, ConversationMessageRepository, UserMemoryRepository

    pg = get_pg()
    msg_repo = ConversationMessageRepository()
    memory_repo = UserMemoryRepository()

    # 用户消息与长期记忆暂存行同一事务双写（append_user_turn 已内置）
    message, queued = await msg_repo.append_user_turn(user_id, conversation_id, "你好")

    # 摘要与 covered_through_seq 同一事务更新（upsert 已内置）
"""

from .base import (
    BaseRepository,
    ensure_transition,
    normalize_message_content,
    to_jsonb,
    translate_pg_error,
)
from .conversations import (
    ConversationMessageRepository,
    ConversationRepository,
    ConversationSummaryRepository,
)
from .knowledge import DocumentRepository, KnowledgeBaseRepository
from .memory import UserMemoryRepository, UserMessageRepository
from .session_files import SessionFileRepository
from .users import PermissionRepository, UserRepository

__all__ = [
    "BaseRepository",
    "UserRepository",
    "PermissionRepository",
    "KnowledgeBaseRepository",
    "DocumentRepository",
    "ConversationRepository",
    "ConversationMessageRepository",
    "ConversationSummaryRepository",
    "UserMemoryRepository",
    "UserMessageRepository",
    "SessionFileRepository",
    "translate_pg_error",
    "ensure_transition",
    "normalize_message_content",
    "to_jsonb",
]
