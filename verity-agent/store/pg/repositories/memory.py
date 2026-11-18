"""user_memory / user_messages 仓库。

- ``user_memory`` 走独立 ``revision`` 列做乐观锁，冲突重试；
  ``facts_revision`` 是事实集版本，用于事实缓存失效（D-50）。
- ``user_messages`` 是长期记忆 worker 的任务队列：回合终态置 ``ready_at``，
  worker 按用户分批领取，``FOR UPDATE SKIP LOCKED`` + 租约 + 退避重试。
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
    require_text,
    to_jsonb,
)

__all__ = ["UserMemoryRepository", "UserMessageRepository"]


class UserMemoryRepository(BaseRepository):
    """每用户一份长期记忆画像（JSONB + revision 乐观锁）。"""

    async def get(
        self, user_id: UUID | str, *, conn: AsyncConnection[Row] | None = None
    ) -> UserMemory | None:
        """按用户取长期记忆；不存在返回 None。"""
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

    async def list_user_ids(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        conn: AsyncConnection[Row] | None = None,
    ) -> list[UUID]:
        """列出有画像的用户 id；事实对账 worker 用。"""
        limit, offset = pagination(limit, offset)
        async with self._transaction(conn) as c:
            rows = await self._fetch_all(
                c,
                """
                SELECT user_id FROM user_memory
                ORDER BY updated_at DESC
                LIMIT %s OFFSET %s
                """,
                (limit, offset),
            )
        return [as_uuid(row["user_id"]) for row in rows]

    async def update(
        self,
        user_id: UUID | str,
        *,
        expected_revision: int,
        value: Any = _UNSET,
        fact_count: Any = _UNSET,
        facts_revision: Any = _UNSET,
        conn: AsyncConnection[Row] | None = None,
    ) -> UserMemory:
        """带版本号的更新；冲突抛 :class:`OptimisticLockError`。"""
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
                        details={"error": str(exc)},
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
        if facts_revision is not _UNSET:
            if facts_revision < 0:
                raise ValidationError("facts_revision 不能为负数")
            sets.append("facts_revision = %s")
            params.append(facts_revision)
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
                    details={
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
                change.value is None
                and change.fact_count is None
                and change.facts_revision is None
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
                    facts_revision=(
                        change.facts_revision
                        if change.facts_revision is not None
                        else _UNSET
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
        """删除用户长期记忆，返回是否实际删除。"""
        async with self._transaction(conn) as c:
            affected = await self._execute(
                c, "DELETE FROM user_memory WHERE user_id = %s", (as_uuid(user_id),)
            )
        return affected > 0


class UserMessageRepository(BaseRepository):
    """长期记忆暂存队列（可恢复、可重试）。

    状态机::

        pending ──(回合终态 ready_at)──> pending(ready)
           └─(claim)─> processing ──(成功)──> 删除
                         ├─(失败)─> failed ──(退避后 claim)──> processing
                         └─(超过上限)─> dead

    - 只领取 ``ready_at IS NOT NULL`` 的行，避免在助手回复前抽“再优化一下”
      这类指代句；
    - ``processing`` 有租约，过期可被重新领取；
    - 一批只包含一个用户，保证画像/事实 apply 目标唯一；
    - 成功删除暂存行，失败保留证据并按退避重试。
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

    async def mark_ready(
        self,
        conversation_id: UUID | str,
        *,
        conn: AsyncConnection[Row] | None = None,
    ) -> int:
        """回合终态：把该会话所有未置位的暂存行标为可抽取。"""
        async with self._transaction(conn) as c:
            return await self._execute(
                c,
                """
                UPDATE user_messages
                SET ready_at = now(), updated_at = now()
                WHERE conversation_id = %s
                  AND ready_at IS NULL
                  AND status <> 'dead'
                """,
                (as_uuid(conversation_id),),
            )

    async def claim_batch(
        self,
        worker_id: str,
        *,
        user_id: UUID | str | None = None,
        limit: int = 20,
        lease_seconds: int = 60,
        conn: AsyncConnection[Row] | None = None,
    ) -> list[UserMessage]:
        """领取一个用户的一批可处理消息并置为 processing。

        - ``user_id`` 为空时，优先选择最早有可处理消息的用户；
        - ``FOR UPDATE SKIP LOCKED`` 保证多 worker 不重复领取；
        - ``processing`` 租约过期（``locked_until < now()``）可重新领取。
        """
        limit, _ = pagination(limit, 0)
        if lease_seconds <= 0:
            raise ValidationError("lease_seconds 必须 > 0")
        worker = require_text(worker_id, "worker_id")
        scope = as_uuid(user_id) if user_id is not None else None
        async with self._transaction(conn) as c:
            rows = await self._fetch_all(
                c,
                """
                WITH candidate_user AS (
                    SELECT user_id
                    FROM user_messages
                    WHERE (
                            (status IN ('pending', 'failed')
                             AND ready_at IS NOT NULL
                             AND (next_retry_at IS NULL OR next_retry_at <= now()))
                         OR (status = 'processing'
                             AND (locked_until IS NULL OR locked_until < now()))
                          )
                      AND (%s::uuid IS NULL OR user_id = %s::uuid)
                    ORDER BY created_at
                    LIMIT 1
                ),
                claimed AS (
                    SELECT id
                    FROM user_messages
                    WHERE user_id = (SELECT user_id FROM candidate_user)
                      AND (
                            (status IN ('pending', 'failed')
                             AND ready_at IS NOT NULL
                             AND (next_retry_at IS NULL OR next_retry_at <= now()))
                         OR (status = 'processing'
                             AND (locked_until IS NULL OR locked_until < now()))
                          )
                    ORDER BY created_at
                    LIMIT %s
                    FOR UPDATE SKIP LOCKED
                )
                UPDATE user_messages um
                SET status = 'processing',
                    locked_by = %s,
                    locked_until = now() + make_interval(secs => %s),
                    updated_at = now()
                FROM claimed
                WHERE um.id = claimed.id
                RETURNING um.*
                """,
                (scope, scope, limit, worker, lease_seconds),
            )
        rows.sort(key=lambda row: row["created_at"])
        return self._models(UserMessage, rows)

    async def mark_processing_done(
        self,
        message_ids: Sequence[UUID | str],
        *,
        conn: AsyncConnection[Row] | None = None,
    ) -> int:
        """完成后删除暂存行（成功路径）。"""
        ids = [as_uuid(mid) for mid in message_ids]
        if not ids:
            return 0
        async with self._transaction(conn) as c:
            return await self._execute(
                c,
                "DELETE FROM user_messages WHERE id = ANY(%s::uuid[]) AND status = 'processing'",
                (ids,),
            )

    async def mark_failed(
        self,
        message_ids: Sequence[UUID | str],
        *,
        error: str,
        retry_backoff_s: float = 30.0,
        max_attempts: int = 5,
        conn: AsyncConnection[Row] | None = None,
    ) -> int:
        """失败回退：``attempt_count++``、写错误、按指数退避；超限置 dead。"""
        ids = [as_uuid(mid) for mid in message_ids]
        if not ids:
            return 0
        if retry_backoff_s <= 0:
            raise ValidationError("retry_backoff_s 必须 > 0")
        if max_attempts < 1:
            raise ValidationError("max_attempts 必须 >= 1")
        async with self._transaction(conn) as c:
            return await self._execute(
                c,
                """
                UPDATE user_messages
                SET status = CASE
                        WHEN attempt_count + 1 >= %s THEN 'dead'
                        ELSE 'failed'
                    END,
                    attempt_count = attempt_count + 1,
                    last_error = %s,
                    next_retry_at = CASE
                        WHEN attempt_count + 1 >= %s THEN NULL
                        ELSE now() + make_interval(
                            secs => %s * power(2, attempt_count)
                        )
                    END,
                    locked_until = NULL,
                    locked_by = NULL,
                    updated_at = now()
                WHERE id = ANY(%s::uuid[]) AND status = 'processing'
                """,
                (max_attempts, error, max_attempts, retry_backoff_s, ids),
            )

    async def delete_batch(
        self,
        message_ids: Sequence[UUID | str],
        *,
        conn: AsyncConnection[Row] | None = None,
    ) -> int:
        """按 id 删除暂存行（不校验状态，管理/清理用）。"""
        ids = [as_uuid(mid) for mid in message_ids]
        if not ids:
            return 0
        async with self._transaction(conn) as c:
            return await self._execute(
                c, "DELETE FROM user_messages WHERE id = ANY(%s::uuid[])", (ids,)
            )

    async def count_ready(
        self,
        *,
        user_id: UUID | str | None = None,
        conn: AsyncConnection[Row] | None = None,
    ) -> int:
        """统计待抽取的用户消息数量，可按用户过滤。"""
        conditions = [
            "ready_at IS NOT NULL",
            "(status IN ('pending', 'failed')"
            " AND (next_retry_at IS NULL OR next_retry_at <= now()))",
        ]
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

    async def list_ready(
        self,
        limit: int = 20,
        *,
        user_id: UUID | str | None = None,
        conn: AsyncConnection[Row] | None = None,
    ) -> list[UserMessage]:
        """只读查看可处理行（不领取）；调试/监控用。"""
        limit, _ = pagination(limit, 0)
        conditions = [
            "ready_at IS NOT NULL",
            "status IN ('pending', 'failed')",
            "(next_retry_at IS NULL OR next_retry_at <= now())",
        ]
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
        """管理/恢复用：显式改状态；正常 worker 走 claim/failed/done。"""
        try:
            status = UserMessageStatus(status)
        except ValueError as exc:
            raise ValidationError(f"非法暂存状态: {status}") from exc
        async with self._transaction(conn) as c:
            row = await self._fetch_one(
                c,
                """
                UPDATE user_messages
                SET status = %s, updated_at = now()
                WHERE id = %s
                RETURNING *
                """,
                (_param(status), as_uuid(message_id)),
            )
        if row is None:
            raise NotFoundError("暂存消息", message_id)
        return UserMessage.model_validate(row)
