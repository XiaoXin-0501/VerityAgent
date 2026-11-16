"""PG 端口适配器：run 租约与会话消息事实源。"""

from __future__ import annotations

from uuid import UUID

from agent.message.codec import decode_stored_row, encode_content
from agent.message.enum import MessageStatus as AgentMessageStatus
from agent.message.enum import MessageRole as AgentMessageRole
from agent.message.models import (
    AssistantMessage,
    StoredMessage,
    ToolMessage,
    UserMessage,
)
from store.pg.models.enums import (
    MessageRole as PgMessageRole,
    MessageStatus as PgMessageStatus,
    RunStatus as PgRunStatus,
)
from store.pg.repositories import (
    ConversationMessageRepository,
    ConversationRepository,
    UserMessageRepository,
)

__all__ = ["PgLeaseAdapter", "PgMessageAdapter", "PgMemoryQueueAdapter"]


class PgLeaseAdapter:
    """会话 run 租约适配器；映射到 conversations.current_run_id。"""

    def __init__(
        self,
        repository: ConversationRepository | None = None,
        *,
        lease_seconds: int = 60,
    ) -> None:
        """保存租约仓储与默认租约时长。"""
        self.repository = repository or ConversationRepository()
        self.lease_seconds = lease_seconds

    async def acquire(self, conversation_id: UUID, run_id: str, lease_seconds: int) -> bool:
        """条件抢占租约；已有活跃 run 返回 False。"""
        try:
            run_uuid = UUID(str(run_id))
        except ValueError:
            return False
        conversation = await self.repository.try_acquire_run(
            conversation_id,
            run_uuid,
            lease_seconds=lease_seconds,
        )
        return conversation is not None

    async def heartbeat(self, conversation_id: UUID, run_id: str, lease_seconds: int) -> bool:
        """续租；租约不属于该 run 时返回 False。"""
        return await self.repository.heartbeat_run(
            conversation_id,
            run_id,
            lease_seconds=lease_seconds,
        )

    async def release(self, conversation_id: UUID, run_id: str, status: str) -> None:
        """释放租约；幂等，终态映射到 PG RunStatus。"""
        try:
            run_uuid = UUID(str(run_id))
        except ValueError:
            return
        await self.repository.release_run(
            conversation_id,
            run_uuid,
            final_status=PgRunStatus(status),
        )


class PgMessageAdapter:
    """ConversationMessageRepository -> Agent MessagePort。"""

    def __init__(
        self,
        repository: ConversationMessageRepository | None = None,
    ) -> None:
        """保存消息仓储；seq_no 由仓储事务分配。"""
        self.repository = repository or ConversationMessageRepository()

    async def find_by_client_message_id(
        self,
        conversation_id: UUID,
        client_message_id: UUID,
    ) -> StoredMessage | None:
        """幂等查询；解码失败时按未命中处理，避免阻断新消息。"""
        row = await self.repository.find_by_client_message_id(
            conversation_id,
            client_message_id,
        )
        if row is None:
            return None
        try:
            return decode_stored_row(row.model_dump())
        except Exception:  # noqa: BLE001 - 坏行不能阻断运行
            return None

    async def append_user(
        self,
        message: UserMessage,
        *,
        client_message_id: UUID,
        user_id: UUID | None = None,
    ) -> StoredMessage:
        """追加用户消息；``user_id`` 非空时与记忆暂存行同事务写入。"""
        content = encode_content(message)
        if user_id is not None:
            # 双写同事务：conversation_messages + user_messages(ready_at=NULL)。
            # 回合终态只负责 mark_ready，不在这里入队。
            row, _ = await self.repository.append_user_turn(
                user_id,
                message.conversation_id,
                content,
                client_message_id=client_message_id,
            )
            return decode_stored_row(row.model_dump())
        row = await self.repository.append(
            message.conversation_id,
            PgMessageRole(AgentMessageRole.USER.value),
            content,
            status=PgMessageStatus(AgentMessageStatus.COMPLETED.value),
            client_message_id=client_message_id,
        )
        return decode_stored_row(row.model_dump())

    async def append_assistant(self, message: AssistantMessage) -> StoredMessage:
        """追加 assistant 消息；framework 阶段一次写入终态。"""
        row = await self.repository.append(
            message.conversation_id,
            PgMessageRole(AgentMessageRole.ASSISTANT.value),
            encode_content(message),
            status=PgMessageStatus(message.status.value),
        )
        return decode_stored_row(row.model_dump())

    async def append_tool(self, message: ToolMessage) -> StoredMessage:
        """追加 ToolMessage；ToolResult -> ToolMessage 已由 Agent 完成。"""
        row = await self.repository.append(
            message.conversation_id,
            PgMessageRole(AgentMessageRole.TOOL.value),
            encode_content(message),
            status=PgMessageStatus(message.status.value),
        )
        return decode_stored_row(row.model_dump())

    async def load_window(
        self,
        conversation_id: UUID,
        *,
        limit: int = 40,
    ) -> list[StoredMessage]:
        """按 seq_no 升序读取最近消息，作为 v1 消息窗口。"""
        rows = await self.repository.get_recent(conversation_id, limit=limit)
        return [decode_stored_row(row.model_dump()) for row in rows]

    async def load_after(
        self,
        conversation_id: UUID,
        seq_no: int,
        *,
        limit: int = 200,
    ) -> list[StoredMessage]:
        """读取指定 seq 之后的消息；幂等命中时重建已完成回合。"""
        rows = await self.repository.list_range(
            conversation_id,
            after_seq=seq_no,
            limit=limit,
        )
        return [decode_stored_row(row.model_dump()) for row in rows]


class PgMemoryQueueAdapter:
    """UserMessageRepository -> Agent MemoryQueuePort（只负责回合终态 ready）。"""

    def __init__(
        self,
        repository: UserMessageRepository | None = None,
    ) -> None:
        """保存长期记忆暂存队列仓储。"""
        self.repository = repository or UserMessageRepository()

    async def mark_ready(self, conversation_id: UUID) -> None:
        """回合终态后置 ready_at，供长期记忆 worker 领取。"""
        await self.repository.mark_ready(conversation_id)
