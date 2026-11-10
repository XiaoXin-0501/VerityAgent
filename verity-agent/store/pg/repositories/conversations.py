"""conversations / conversation_messages / conversation_summaries 仓库。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import UUID

from psycopg import AsyncConnection

from exception import (
    ConflictError,
    NotFoundError,
    ValidationError,
)

from ..connection import Row
from ..models.conversation import (
    Conversation,
    ConversationMessage,
    ConversationSummary,
    SessionSummary,
)
from ..models.enums import (
    ConversationStatus,
    MessageRole,
    MessageStatus,
    RunStatus,
)
from ..models.memory import UserMessage
from .base import (
    BaseRepository,
    _param,
    as_uuid,
    normalize_message_content,
    pagination,
    to_jsonb,
    validate_title,
)
from .memory import UserMessageRepository


class ConversationRepository(BaseRepository):
    """会话元数据、运行状态与 token / 轮数累计。"""

    async def create(
        self,
        user_id: UUID | str,
        *,
        title: str = "",
        conn: AsyncConnection[Row] | None = None,
    ) -> Conversation:
        title = validate_title(title)
        async with self._transaction(conn) as c:
            row = await self._fetch_one(
                c,
                """
                INSERT INTO conversations (user_id, title)
                VALUES (%s, %s)
                RETURNING *
                """,
                (as_uuid(user_id), title),
            )
        assert row is not None
        return Conversation.model_validate(row)

    async def get(
        self,
        conversation_id: UUID | str,
        *,
        include_deleted: bool = False,
        conn: AsyncConnection[Row] | None = None,
    ) -> Conversation | None:
        sql = "SELECT * FROM conversations WHERE id = %s"
        if not include_deleted:
            sql += " AND deleted_at IS NULL"
        async with self._transaction(conn) as c:
            row = await self._fetch_one(c, sql, (as_uuid(conversation_id),))
        return self._model(Conversation, row)

    async def get_for_user(
        self,
        user_id: UUID | str,
        conversation_id: UUID | str,
        *,
        conn: AsyncConnection[Row] | None = None,
    ) -> Conversation:
        """按归属取会话，越权一律当作不存在，避免探测他人会话 id。"""
        async with self._transaction(conn) as c:
            row = await self._fetch_one(
                c,
                """
                SELECT * FROM conversations
                WHERE id = %s AND user_id = %s AND deleted_at IS NULL
                """,
                (as_uuid(conversation_id), as_uuid(user_id)),
            )
        if row is None:
            raise NotFoundError("会话", conversation_id)
        return Conversation.model_validate(row)

    async def list_by_user(
        self,
        user_id: UUID | str,
        *,
        status: ConversationStatus | None = None,
        limit: int = 50,
        offset: int = 0,
        conn: AsyncConnection[Row] | None = None,
    ) -> list[Conversation]:
        """会话列表按 last_message_at 倒序，避免子查 conversation_messages。"""
        limit, offset = pagination(limit, offset)
        conditions = ["user_id = %s"]
        params: list[Any] = [as_uuid(user_id)]
        if status is None:
            conditions.append("deleted_at IS NULL")
        else:
            conditions.append("status = %s")
            params.append(_param(status))
        params.extend([limit, offset])
        async with self._transaction(conn) as c:
            rows = await self._fetch_all(
                c,
                f"""
                SELECT * FROM conversations
                WHERE {' AND '.join(conditions)}
                ORDER BY last_message_at DESC NULLS LAST, created_at DESC
                LIMIT %s OFFSET %s
                """,
                params,
            )
        return self._models(Conversation, rows)

    async def update_title(
        self,
        conversation_id: UUID | str,
        title: str,
        *,
        conn: AsyncConnection[Row] | None = None,
    ) -> Conversation:
        title = validate_title(title)
        async with self._transaction(conn) as c:
            row = await self._fetch_one(
                c,
                """
                UPDATE conversations SET title = %s
                WHERE id = %s AND deleted_at IS NULL
                RETURNING *
                """,
                (title, as_uuid(conversation_id)),
            )
        if row is None:
            raise NotFoundError("会话", conversation_id)
        return Conversation.model_validate(row)

    async def set_status(
        self,
        conversation_id: UUID | str,
        status: ConversationStatus,
        *,
        conn: AsyncConnection[Row] | None = None,
    ) -> Conversation:
        async with self._transaction(conn) as c:
            row = await self._fetch_one(
                c,
                """
                UPDATE conversations SET status = %s
                WHERE id = %s
                RETURNING *
                """,
                (_param(status), as_uuid(conversation_id)),
            )
        if row is None:
            raise NotFoundError("会话", conversation_id)
        return Conversation.model_validate(row)

    async def set_run_status(
        self,
        conversation_id: UUID | str,
        run_status: RunStatus,
        *,
        conn: AsyncConnection[Row] | None = None,
    ) -> Conversation:
        """只改当前运行状态，不动会话生命周期（暂停的是一次运行）。"""
        async with self._transaction(conn) as c:
            row = await self._fetch_one(
                c,
                """
                UPDATE conversations SET current_run_status = %s
                WHERE id = %s AND deleted_at IS NULL
                RETURNING *
                """,
                (_param(run_status), as_uuid(conversation_id)),
            )
        if row is None:
            raise NotFoundError("会话", conversation_id)
        return Conversation.model_validate(row)

    # ——————————————————— run 租约（P0-8） ———————————————————

    async def try_acquire_run(
        self,
        conversation_id: UUID | str,
        run_id: UUID | str,
        *,
        lease_seconds: int,
        conn: AsyncConnection[Row] | None = None,
    ) -> Conversation | None:
        """抢占会话的 run 租约；已被活跃 run 占用时返回 None。

        条件更新保证同一会话同一时刻只有一个 run；调用方拿到 None 时应返回
        409（或进入排队），不能并行写消息/工具链。
        """
        if lease_seconds <= 0:
            raise ValidationError("lease_seconds 必须 > 0")
        async with self._transaction(conn) as c:
            row = await self._fetch_one(
                c,
                """
                UPDATE conversations
                SET current_run_id = %s,
                    run_started_at = now(),
                    lock_expires_at = now() + make_interval(secs => %s),
                    current_run_status = 'running'
                WHERE id = %s
                  AND deleted_at IS NULL
                  AND (current_run_id IS NULL OR lock_expires_at < now())
                RETURNING *
                """,
                (as_uuid(run_id), lease_seconds, as_uuid(conversation_id)),
            )
        return self._model(Conversation, row)

    async def acquire_run(
        self,
        conversation_id: UUID | str,
        run_id: UUID | str,
        *,
        lease_seconds: int,
        conn: AsyncConnection[Row] | None = None,
    ) -> Conversation:
        conversation = await self.try_acquire_run(
            conversation_id, run_id, lease_seconds=lease_seconds, conn=conn
        )
        if conversation is None:
            raise ConflictError(
                "会话已有活跃 run",
                detail={
                    "conversation_id": str(conversation_id),
                    "run_id": str(run_id),
                },
            )
        return conversation

    async def heartbeat_run(
        self,
        conversation_id: UUID | str,
        run_id: UUID | str,
        *,
        lease_seconds: int,
        conn: AsyncConnection[Row] | None = None,
    ) -> bool:
        """续租；只有仍持有该 run 租约时返回 True。"""
        if lease_seconds <= 0:
            raise ValidationError("lease_seconds 必须 > 0")
        async with self._transaction(conn) as c:
            affected = await self._execute(
                c,
                """
                UPDATE conversations
                SET lock_expires_at = now() + make_interval(secs => %s)
                WHERE id = %s AND current_run_id = %s AND deleted_at IS NULL
                """,
                (lease_seconds, as_uuid(conversation_id), as_uuid(run_id)),
            )
        return affected > 0

    async def release_run(
        self,
        conversation_id: UUID | str,
        run_id: UUID | str,
        *,
        final_status: RunStatus,
        conn: AsyncConnection[Row] | None = None,
    ) -> Conversation | None:
        """释放租约并置终态；租约已不属于该 run 时返回 None。"""
        async with self._transaction(conn) as c:
            row = await self._fetch_one(
                c,
                """
                UPDATE conversations
                SET current_run_id = NULL,
                    lock_expires_at = NULL,
                    current_run_status = %s
                WHERE id = %s AND current_run_id = %s
                RETURNING *
                """,
                (_param(final_status), as_uuid(conversation_id), as_uuid(run_id)),
            )
        return self._model(Conversation, row)

    async def clear_expired_run(
        self,
        conversation_id: UUID | str,
        *,
        final_status: RunStatus = RunStatus.FAILED,
        conn: AsyncConnection[Row] | None = None,
    ) -> Conversation | None:
        """进程崩溃后的收敛：租约过期时清空 current_run_id。"""
        async with self._transaction(conn) as c:
            row = await self._fetch_one(
                c,
                """
                UPDATE conversations
                SET current_run_id = NULL,
                    lock_expires_at = NULL,
                    current_run_status = %s
                WHERE id = %s
                  AND current_run_id IS NOT NULL
                  AND lock_expires_at < now()
                RETURNING *
                """,
                (_param(final_status), as_uuid(conversation_id)),
            )
        return self._model(Conversation, row)

    async def record_turn(
        self,
        conversation_id: UUID | str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        turn_increment: int = 1,
        conn: AsyncConnection[Row] | None = None,
    ) -> Conversation:
        """一轮对话结束后累计 token 与回合数（原子自增，无读改写竞争）。"""
        for value, name in (
            (input_tokens, "input_tokens"),
            (output_tokens, "output_tokens"),
            (turn_increment, "turn_increment"),
        ):
            if value < 0:
                raise ValidationError(f"{name} 不能为负数")
        async with self._transaction(conn) as c:
            row = await self._fetch_one(
                c,
                """
                UPDATE conversations
                SET turn_count = turn_count + %s,
                    total_input_tokens = total_input_tokens + %s,
                    total_output_tokens = total_output_tokens + %s
                WHERE id = %s AND deleted_at IS NULL
                RETURNING *
                """,
                (
                    turn_increment,
                    input_tokens,
                    output_tokens,
                    as_uuid(conversation_id),
                ),
            )
        if row is None:
            raise NotFoundError("会话", conversation_id)
        return Conversation.model_validate(row)

    async def archive(
        self, conversation_id: UUID | str, *, conn: AsyncConnection[Row] | None = None
    ) -> Conversation:
        return await self.set_status(
            conversation_id, ConversationStatus.ARCHIVED, conn=conn
        )

    async def mark_deleted(
        self, conversation_id: UUID | str, *, conn: AsyncConnection[Row] | None = None
    ) -> Conversation:
        """发起删除：先软删 PG，读路径立即过滤；跨存储清理由 worker 继续。

        原始消息保留在 ``conversation_messages``，直到删除状态机完成硬删。
        """
        async with self._transaction(conn) as c:
            row = await self._fetch_one(
                c,
                """
                UPDATE conversations
                SET status = 'deleting',
                    deleted_at = COALESCE(deleted_at, now())
                WHERE id = %s AND status <> 'deleted'
                RETURNING *
                """,
                (as_uuid(conversation_id),),
            )
        if row is None:
            raise NotFoundError("会话", conversation_id)
        return Conversation.model_validate(row)

    async def record_delete_failure(
        self,
        conversation_id: UUID | str,
        error: str,
        *,
        retry_at: Any | None = None,
        conn: AsyncConnection[Row] | None = None,
    ) -> Conversation:
        async with self._transaction(conn) as c:
            row = await self._fetch_one(
                c,
                """
                UPDATE conversations
                SET delete_attempts = delete_attempts + 1,
                    delete_error = %s,
                    delete_next_retry_at = %s
                WHERE id = %s
                RETURNING *
                """,
                (error, retry_at, as_uuid(conversation_id)),
            )
        if row is None:
            raise NotFoundError("会话", conversation_id)
        return Conversation.model_validate(row)

    async def finish_deletion(
        self,
        conversation_id: UUID | str,
        *,
        conn: AsyncConnection[Row] | None = None,
    ) -> Conversation:
        """跨存储清理完成：置 deleted tombstone，读路径永久过滤。"""
        async with self._transaction(conn) as c:
            row = await self._fetch_one(
                c,
                """
                UPDATE conversations
                SET status = 'deleted',
                    deleted_at = COALESCE(deleted_at, now()),
                    delete_error = NULL,
                    delete_next_retry_at = NULL
                WHERE id = %s
                RETURNING *
                """,
                (as_uuid(conversation_id),),
            )
        if row is None:
            raise NotFoundError("会话", conversation_id)
        return Conversation.model_validate(row)


class ConversationMessageRepository(BaseRepository):
    """会话原始消息（append-only ground truth）。

    ``seq_no`` 在事务内先锁住 conversations 行再取 ``MAX(seq_no) + 1``，
    同一会话的追加被串行化，避免并发产生重复序号（也避免唯一冲突重试）。
    """

    def __init__(self, db: Any | None = None) -> None:
        super().__init__(db)
        self._user_messages = UserMessageRepository(db)

    async def append(
        self,
        conversation_id: UUID | str,
        role: MessageRole,
        content: Any,
        *,
        status: MessageStatus = MessageStatus.COMPLETED,
        seq_no: int | None = None,
        client_message_id: UUID | str | None = None,
        conn: AsyncConnection[Row] | None = None,
    ) -> ConversationMessage:
        async with self._transaction(conn) as c:
            return await self._append(
                c,
                conversation_id,
                role,
                content,
                status=status,
                seq_no=seq_no,
                client_message_id=client_message_id,
            )

    async def _append(
        self,
        conn: AsyncConnection[Row],
        conversation_id: UUID | str,
        role: MessageRole,
        content: Any,
        *,
        status: MessageStatus,
        seq_no: int | None,
        client_message_id: UUID | str | None = None,
    ) -> ConversationMessage:
        conversation_uuid = as_uuid(conversation_id)
        payload = normalize_message_content(content)

        locked = await self._fetch_one(
            conn,
            "SELECT id FROM conversations WHERE id = %s AND deleted_at IS NULL FOR UPDATE",
            (conversation_uuid,),
        )
        if locked is None:
            raise NotFoundError("会话", conversation_id)

        if seq_no is None:
            seq_no = await self._fetch_val(
                conn,
                """
                SELECT COALESCE(MAX(seq_no), 0) + 1
                FROM conversation_messages WHERE conversation_id = %s
                """,
                (conversation_uuid,),
            )
        if not isinstance(seq_no, int) or seq_no <= 0:
            raise ValidationError("seq_no 必须是正整数")

        row = await self._fetch_one(
            conn,
            """
            INSERT INTO conversation_messages
                (conversation_id, seq_no, client_message_id, role, content, status)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING *
            """,
            (
                conversation_uuid,
                seq_no,
                as_uuid(client_message_id) if client_message_id is not None else None,
                _param(role),
                to_jsonb(payload),
                _param(status),
            ),
        )
        # 冗余时间戳：列表排序不子查消息表
        await self._execute(
            conn,
            "UPDATE conversations SET last_message_at = now() WHERE id = %s",
            (conversation_uuid,),
        )
        assert row is not None
        return ConversationMessage.model_validate(row)

    async def find_by_client_message_id(
        self,
        conversation_id: UUID | str,
        client_message_id: UUID | str,
        *,
        conn: AsyncConnection[Row] | None = None,
    ) -> ConversationMessage | None:
        async with self._transaction(conn) as c:
            row = await self._fetch_one(
                c,
                """
                SELECT * FROM conversation_messages
                WHERE conversation_id = %s AND client_message_id = %s
                """,
                (as_uuid(conversation_id), as_uuid(client_message_id)),
            )
        return self._model(ConversationMessage, row)

    async def append_user_turn(
        self,
        user_id: UUID | str,
        conversation_id: UUID | str,
        content: Any,
        *,
        client_message_id: UUID | str | None = None,
        conn: AsyncConnection[Row] | None = None,
    ) -> tuple[ConversationMessage, UserMessage]:
        """写用户消息 + 同步写入长期记忆暂存表（同一事务双写）。

        设计文档要求落原始消息和落 ``user_messages`` 必须一次事务完成，
        否则 worker 会漏扫或产生无主暂存行。暂存行此时 ``ready_at=NULL``，
        回合终态后再置位。
        """
        async with self._transaction(conn) as c:
            message = await self._append(
                c,
                conversation_id,
                MessageRole.USER,
                content,
                status=MessageStatus.COMPLETED,
                seq_no=None,
                client_message_id=client_message_id,
            )
            queued = await self._user_messages.enqueue(
                user_id,
                conversation_id,
                message.id,
                content,
                conn=c,
            )
        return message, queued

    async def get(
        self,
        message_id: UUID | str,
        *,
        conn: AsyncConnection[Row] | None = None,
    ) -> ConversationMessage | None:
        async with self._transaction(conn) as c:
            row = await self._fetch_one(
                c,
                "SELECT * FROM conversation_messages WHERE id = %s",
                (as_uuid(message_id),),
            )
        return self._model(ConversationMessage, row)

    async def list_range(
        self,
        conversation_id: UUID | str,
        *,
        after_seq: int = 0,
        through_seq: int | None = None,
        limit: int | None = None,
        conn: AsyncConnection[Row] | None = None,
    ) -> list[ConversationMessage]:
        """按 ``seq_no`` 升序取一段消息。

        压缩工作窗口时用 ``after_seq=covered_through_seq`` 拉待压缩批次；
        ``through_seq`` 用于把边界钉死，避免压缩过程中新消息被误纳入。
        """
        if after_seq < 0:
            raise ValidationError("after_seq 不能为负数")
        conditions = ["conversation_id = %s", "seq_no > %s"]
        params: list[Any] = [as_uuid(conversation_id), after_seq]
        if through_seq is not None:
            conditions.append("seq_no <= %s")
            params.append(through_seq)
        sql = f"""
            SELECT * FROM conversation_messages
            WHERE {' AND '.join(conditions)}
            ORDER BY seq_no ASC
        """
        if limit is not None:
            limit, _ = pagination(limit, 0)
            sql += " LIMIT %s"
            params.append(limit)
        async with self._transaction(conn) as c:
            rows = await self._fetch_all(c, sql, params)
        return self._models(ConversationMessage, rows)

    async def get_recent(
        self,
        conversation_id: UUID | str,
        limit: int = 2,
        *,
        conn: AsyncConnection[Row] | None = None,
    ) -> list[ConversationMessage]:
        """历史会话恢复工作窗口：取最近 N 条，按 seq_no 升序返回。"""
        limit, _ = pagination(limit, 0)
        async with self._transaction(conn) as c:
            rows = await self._fetch_all(
                c,
                """
                SELECT * FROM (
                    SELECT * FROM conversation_messages
                    WHERE conversation_id = %s
                    ORDER BY seq_no DESC
                    LIMIT %s
                ) recent
                ORDER BY seq_no ASC
                """,
                (as_uuid(conversation_id), limit),
            )
        return self._models(ConversationMessage, rows)

    async def max_seq(
        self,
        conversation_id: UUID | str,
        *,
        conn: AsyncConnection[Row] | None = None,
    ) -> int:
        async with self._transaction(conn) as c:
            value = await self._fetch_val(
                c,
                """
                SELECT COALESCE(MAX(seq_no), 0) FROM conversation_messages
                WHERE conversation_id = %s
                """,
                (as_uuid(conversation_id),),
            )
        return int(value or 0)

    async def count(
        self,
        conversation_id: UUID | str,
        *,
        conn: AsyncConnection[Row] | None = None,
    ) -> int:
        async with self._transaction(conn) as c:
            value = await self._fetch_val(
                c,
                "SELECT count(*) FROM conversation_messages WHERE conversation_id = %s",
                (as_uuid(conversation_id),),
            )
        return int(value or 0)

    async def update_status(
        self,
        message_id: UUID | str,
        status: MessageStatus,
        *,
        conn: AsyncConnection[Row] | None = None,
    ) -> ConversationMessage:
        """流式输出时把 streaming 消息落成 completed / partial / failed。"""
        async with self._transaction(conn) as c:
            row = await self._fetch_one(
                c,
                """
                UPDATE conversation_messages
                SET status = %s, updated_at = now()
                WHERE id = %s
                RETURNING *
                """,
                (_param(status), as_uuid(message_id)),
            )
        if row is None:
            raise NotFoundError("消息", message_id)
        return ConversationMessage.model_validate(row)

    async def finalize(
        self,
        message_id: UUID | str,
        *,
        content: Any,
        status: MessageStatus,
        conn: AsyncConnection[Row] | None = None,
    ) -> ConversationMessage:
        """assistant 流式消息的**一次**终态写入（D-01/D-25）。

        流式过程中不逐 delta 写库；结束时把完整 content、thinking、
        tool_calls、generation 与 status 一并写回。
        """
        payload = normalize_message_content(content)
        async with self._transaction(conn) as c:
            row = await self._fetch_one(
                c,
                """
                UPDATE conversation_messages
                SET content = %s, status = %s, updated_at = now()
                WHERE id = %s
                RETURNING *
                """,
                (to_jsonb(payload), _param(status), as_uuid(message_id)),
            )
        if row is None:
            raise NotFoundError("消息", message_id)
        return ConversationMessage.model_validate(row)


class ConversationSummaryRepository(BaseRepository):
    """会话摘要：``summary_json`` 与 ``covered_through_seq`` 永远同事务更新。"""

    async def get(
        self,
        conversation_id: UUID | str,
        *,
        conn: AsyncConnection[Row] | None = None,
    ) -> ConversationSummary | None:
        async with self._transaction(conn) as c:
            row = await self._fetch_one(
                c,
                "SELECT * FROM conversation_summaries WHERE conversation_id = %s",
                (as_uuid(conversation_id),),
            )
        return self._model(ConversationSummary, row)

    async def upsert(
        self,
        conversation_id: UUID | str,
        summary: SessionSummary | Mapping[str, Any],
        covered_through_seq: int,
        *,
        expected_covered_through_seq: int | None = None,
        increment_compression: bool = True,
        conn: AsyncConnection[Row] | None = None,
    ) -> ConversationSummary:
        """同事务写摘要、覆盖序号与压缩次数。

        - 传入 ``expected_covered_through_seq`` 时按 CAS 更新（P0-4）：水位线
          已被其他压缩推进则 0 行，抛 ``ConflictError``，不要复用旧 LLM 输出；
        - 未传时使用单调递增保护，拒绝序号回退；
        - 压缩成功才 ``compression_count + 1``。
        """
        if covered_through_seq < 0:
            raise ValidationError("covered_through_seq 不能为负数")
        if isinstance(summary, SessionSummary):
            model = summary
        else:
            try:
                model = SessionSummary.model_validate(summary)
            except ValueError as exc:
                raise ValidationError(
                    "summary_json 不符合固定结构",
                    detail={"error": str(exc)},
                ) from exc

        payload = to_jsonb(model)
        increment = 1 if increment_compression else 0
        async with self._transaction(conn) as c:
            if expected_covered_through_seq is not None:
                if expected_covered_through_seq > 0:
                    row = await self._fetch_one(
                        c,
                        """
                        UPDATE conversation_summaries
                        SET summary_json = %s,
                            covered_through_seq = %s,
                            compression_count = compression_count + %s,
                            updated_at = now()
                        WHERE conversation_id = %s
                          AND covered_through_seq = %s
                        RETURNING *
                        """,
                        (
                            payload,
                            covered_through_seq,
                            increment,
                            as_uuid(conversation_id),
                            expected_covered_through_seq,
                        ),
                    )
                else:
                    row = await self._fetch_one(
                        c,
                        """
                        INSERT INTO conversation_summaries
                            (conversation_id, summary_json, covered_through_seq,
                             compression_count)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (conversation_id) DO UPDATE
                           SET summary_json = EXCLUDED.summary_json,
                               covered_through_seq = EXCLUDED.covered_through_seq,
                               compression_count = conversation_summaries.compression_count
                                                    + EXCLUDED.compression_count,
                               updated_at = now()
                         WHERE conversation_summaries.covered_through_seq = 0
                        RETURNING *
                        """,
                        (
                            as_uuid(conversation_id),
                            payload,
                            covered_through_seq,
                            increment,
                        ),
                    )
            else:
                row = await self._fetch_one(
                    c,
                    """
                    INSERT INTO conversation_summaries
                        (conversation_id, summary_json, covered_through_seq,
                         compression_count)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (conversation_id) DO UPDATE
                       SET summary_json = EXCLUDED.summary_json,
                           covered_through_seq = EXCLUDED.covered_through_seq,
                           compression_count = conversation_summaries.compression_count
                                                + EXCLUDED.compression_count,
                           updated_at = now()
                     WHERE conversation_summaries.covered_through_seq
                           <= EXCLUDED.covered_through_seq
                    RETURNING *
                    """,
                    (
                        as_uuid(conversation_id),
                        payload,
                        covered_through_seq,
                        increment,
                    ),
                )
            if row is None:
                current = await self._fetch_val(
                    c,
                    """
                    SELECT covered_through_seq FROM conversation_summaries
                    WHERE conversation_id = %s
                    """,
                    (as_uuid(conversation_id),),
                )
                raise ConflictError(
                    "摘要覆盖序号冲突",
                    detail={
                        "conversation_id": str(conversation_id),
                        "current_covered_through_seq": current,
                        "expected_covered_through_seq": expected_covered_through_seq,
                        "new_covered_through_seq": covered_through_seq,
                    },
                )
        return ConversationSummary.model_validate(row)

    async def delete(
        self,
        conversation_id: UUID | str,
        *,
        conn: AsyncConnection[Row] | None = None,
    ) -> bool:
        async with self._transaction(conn) as c:
            affected = await self._execute(
                c,
                "DELETE FROM conversation_summaries WHERE conversation_id = %s",
                (as_uuid(conversation_id),),
            )
        return affected > 0


__all__ = [
    "ConversationRepository",
    "ConversationMessageRepository",
    "ConversationSummaryRepository",
]
