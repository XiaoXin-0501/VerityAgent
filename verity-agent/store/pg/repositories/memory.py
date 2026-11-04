"""user_memory / user_messages 仓库。

- ``user_memory`` 走独立 ``revision`` 列做乐观锁，冲突重试；
- ``user_messages`` 是长期记忆 worker 的任务队列，用
  ``FOR UPDATE SKIP LOCKED`` 领取，重启后重新扫 pending 即可断点续跑。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from uuid import UUID

from psycopg import AsyncConnection

from exception import (
    MemoryFactLimitExceeded,
    NotFoundError,
    OptimisticLockError,
    ValidationError,
)

from ..connection import Row
from ..models.enums import MAX_USER_FACTS, UserMessageStatus
from ..models.memory import (
    MemoryChange,
    MemoryValue,
    UserMemory,
    UserMessage,
    empty_memory_value,
)
from .base import (
    BaseRepository,
    _UNSET,
    _param,
    as_uuid,
    normalize_message_content,
    pagination,
    to_jsonb,
)


class UserMemoryRepository(BaseRepository):
    """每用户一份长期记忆画像（JSONB + revision 乐观锁）。"""

    async def get(
        self, user_id: UUID | str, *, conn: AsyncConnection[Row] | None = None
    ) -> UserMemory | None:
        async with self._transaction(conn) as c:
            row = await self._fetch_one(
                c, "SELECT * FROM user_memory WHERE user_id = %s", (as_uuid(user_id),)
            )
        return self._model(UserMemory, row)

    async def get_or_create(
        self, user_id: UUID | str, *, conn: AsyncConnection[Row] | None = None
    ) -> UserMemory:
        """取画像；不存在则初始化一份空画像（revision=0, fact_count=0）。"""
        user_uuid = as_uuid(user_id)
        async with self._transaction(conn) as c:
            row = await self._fetch_one(
                c,
                """
                INSERT INTO user_memory (user_id, value)
                VALUES (%s, %s)
                ON CONFLICT (user_id) DO NOTHING
                RETURNING *
                """,
                (user_uuid, to_jsonb(empty_memory_value())),
            )
            if row is None:
                row = await self._fetch_one(
                    c,
                    "SELECT * FROM user_memory WHERE user_id = %s",
                    (user_uuid,),
                )
        assert row is not None
        return UserMemory.model_validate(row)

    async def update(
        self,
        user_id: UUID | str,
        *,
        expected_revision: int,
        value: Any = _UNSET,
        fact_count: Any = _UNSET,
        conn: AsyncConnection[Row] | None = None,
    ) -> UserMemory:
        """带版本号的更新。

        冲突（无行返回但记录存在）抛 :class:`OptimisticLockError`，
        调用方可以重读后重试，或直接用 :meth:`update_with_retry`。
        """
        if expected_revision < 0:
            raise ValidationError("expected_revision 不能为负数")

        sets = ["revision = revision + 1", "updated_at = now()"]
        params: list[Any] = []
        if value is not _UNSET:
            if isinstance(value, Mapping):
                try:
                    value = MemoryValue.model_validate(value)
                except ValueError as exc:
                    raise ValidationError(
                        "user_memory.value 不符合固定 7 段结构",
                        detail={"error": str(exc)},
                    ) from exc
            sets.append("value = %s")
            params.append(to_jsonb(value))
        if fact_count is not _UNSET:
            if fact_count < 0:
                raise ValidationError("fact_count 不能为负数")
            if fact_count > MAX_USER_FACTS:
                raise MemoryFactLimitExceeded(fact_count, MAX_USER_FACTS)
            sets.append("fact_count = %s")
            params.append(fact_count)
        if len(sets) == 2:  # 只加了 revision / updated_at，说明没有任何业务字段
            raise ValidationError("没有需要更新的字段")

        user_uuid = as_uuid(user_id)
        params.extend([user_uuid, expected_revision])
        async with self._transaction(conn) as c:
            row = await self._fetch_one(
                c,
                f"""
                UPDATE user_memory SET {', '.join(sets)}
                WHERE user_id = %s AND revision = %s
                RETURNING *
                """,
                params,
            )
            if row is None:
                current = await self._fetch_val(
                    c,
                    "SELECT revision FROM user_memory WHERE user_id = %s",
                    (user_uuid,),
                )
                if current is None:
                    raise NotFoundError("用户长期记忆", user_id)
                raise OptimisticLockError(
                    "长期记忆已被并发修改，请重新读取后重试",
                    detail={
                        "user_id": str(user_id),
                        "expected_revision": expected_revision,
                        "current_revision": current,
                    },
                )
        return UserMemory.model_validate(row)

    async def update_with_retry(
        self,
        user_id: UUID | str,
        mutate: Callable[[UserMemory], MemoryChange | None],
        *,
        max_retries: int = 3,
        backoff: float = 0.05,
    ) -> UserMemory:
        """读 -> mutate -> 乐观锁写，版本冲突时指数退避重试。

        每次尝试都是独立事务，不能在外部事务里使用（冲突后 PG 事务已中止）。
        """
        if max_retries < 0:
            raise ValidationError("max_retries 不能为负数")
        for attempt in range(max_retries + 1):
            current = await self.get_or_create(user_id)
            change = mutate(current)
            if change is None or (
                change.value is None and change.fact_count is None
            ):
                return current
            try:
                return await self.update(
                    user_id,
                    expected_revision=current.revision,
                    value=change.value if change.value is not None else _UNSET,
                    fact_count=(
                        change.fact_count if change.fact_count is not None else _UNSET
                    ),
                )
            except OptimisticLockError:
                if attempt >= max_retries:
                    raise
                await asyncio.sleep(backoff * (2**attempt))
        raise AssertionError("unreachable: update_with_retry 至少执行一次")

    async def delete(
        self, user_id: UUID | str, *, conn: AsyncConnection[Row] | None = None
    ) -> bool:
        async with self._transaction(conn) as c:
            affected = await self._execute(
                c, "DELETE FROM user_memory WHERE user_id = %s", (as_uuid(user_id),)
            )
        return affected > 0


class UserMessageRepository(BaseRepository):
    """长期记忆暂存队列。

    状态机：``pending -> processing -> (成功删除 | 失败置 failed)``。
    删除只允许在 ``user_memory`` 更新成功之后执行，失败留 failed 待重试。
    """

    async def enqueue(
        self,
        user_id: UUID | str,
        conversation_id: UUID | str,
        message_id: UUID | str,
        content: Any,
        *,
        conn: AsyncConnection[Row] | None = None,
    ) -> UserMessage:
        """写入暂存行；同一 message_id 重复入队时返回已有行（幂等）。"""
        payload = normalize_message_content(content)
        async with self._transaction(conn) as c:
            row = await self._fetch_one(
                c,
                """
                INSERT INTO user_messages
                    (user_id, conversation_id, message_id, content)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (message_id) DO NOTHING
                RETURNING *
                """,
                (
                    as_uuid(user_id),
                    as_uuid(conversation_id),
                    as_uuid(message_id),
                    to_jsonb(payload),
                ),
            )
            if row is None:
                row = await self._fetch_one(
                    c,
                    "SELECT * FROM user_messages WHERE message_id = %s",
                    (as_uuid(message_id),),
                )
        assert row is not None
        return UserMessage.model_validate(row)

    async def claim_batch(
        self,
        limit: int = 20,
        *,
        conn: AsyncConnection[Row] | None = None,
    ) -> list[UserMessage]:
        """领取一批 pending 消息并置为 processing（SKIP LOCKED，多 worker 安全）。"""
        limit, _ = pagination(limit, 0)
        async with self._transaction(conn) as c:
            rows = await self._fetch_all(
                c,
                """
                UPDATE user_messages SET status = 'processing'
                WHERE id IN (
                    SELECT id FROM user_messages
                    WHERE status = 'pending'
                    ORDER BY created_at
                    LIMIT %s
                    FOR UPDATE SKIP LOCKED
                )
                RETURNING *
                """,
                (limit,),
            )
        rows.sort(key=lambda row: row["created_at"])
        return self._models(UserMessage, rows)

    async def list_pending(
        self,
        limit: int = 20,
        *,
        user_id: UUID | str | None = None,
        conn: AsyncConnection[Row] | None = None,
    ) -> list[UserMessage]:
        limit, _ = pagination(limit, 0)
        conditions = ["status = 'pending'"]
        params: list[Any] = []
        if user_id is not None:
            conditions.append("user_id = %s")
            params.append(as_uuid(user_id))
        params.append(limit)
        async with self._transaction(conn) as c:
            rows = await self._fetch_all(
                c,
                f"""
                SELECT * FROM user_messages
                WHERE {' AND '.join(conditions)}
                ORDER BY created_at
                LIMIT %s
                """,
                params,
            )
        return self._models(UserMessage, rows)

    async def update_status(
        self,
        message_id: UUID | str,
        status: UserMessageStatus,
        *,
        conn: AsyncConnection[Row] | None = None,
    ) -> UserMessage:
        try:
            status = UserMessageStatus(status)
        except ValueError as exc:
            raise ValidationError(f"非法暂存状态: {status}") from exc
        async with self._transaction(conn) as c:
            row = await self._fetch_one(
                c,
                """
                UPDATE user_messages SET status = %s
                WHERE id = %s
                RETURNING *
                """,
                (_param(status), as_uuid(message_id)),
            )
        if row is None:
            raise NotFoundError("暂存消息", message_id)
        return UserMessage.model_validate(row)

    async def mark_failed(
        self,
        message_ids: Sequence[UUID | str],
        *,
        conn: AsyncConnection[Row] | None = None,
    ) -> int:
        """把处理失败的消息置 failed，等待下一轮重试。"""
        ids = [as_uuid(mid) for mid in message_ids]
        if not ids:
            return 0
        async with self._transaction(conn) as c:
            return await self._execute(
                c,
                """
                UPDATE user_messages SET status = 'failed'
                WHERE id = ANY(%s) AND status = 'processing'
                """,
                (ids,),
            )

    async def delete_batch(
        self,
        message_ids: Sequence[UUID | str],
        *,
        conn: AsyncConnection[Row] | None = None,
    ) -> int:
        """长期记忆 apply 成功后删除暂存行。"""
        ids = [as_uuid(mid) for mid in message_ids]
        if not ids:
            return 0
        async with self._transaction(conn) as c:
            return await self._execute(
                c, "DELETE FROM user_messages WHERE id = ANY(%s)", (ids,)
            )

    async def count_pending(
        self,
        *,
        user_id: UUID | str | None = None,
        conn: AsyncConnection[Row] | None = None,
    ) -> int:
        conditions = ["status = 'pending'"]
        params: list[Any] = []
        if user_id is not None:
            conditions.append("user_id = %s")
            params.append(as_uuid(user_id))
        async with self._transaction(conn) as c:
            value = await self._fetch_val(
                c,
                f"SELECT count(*) FROM user_messages WHERE {' AND '.join(conditions)}",
                params,
            )
        return int(value or 0)


__all__ = ["UserMemoryRepository", "UserMessageRepository"]
