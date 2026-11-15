"""users / permissions / user_permissions 仓库。"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from psycopg import AsyncConnection

from exception import NotFoundError, ValidationError

from ..connection import Row
from ..models.enums import UserRole, UserStatus
from ..models.user import Permission, User, UserPermission
from .base import BaseRepository, _UNSET, _param, as_uuid, pagination, require_text


class UserRepository(BaseRepository):
    """用户账户。"""

    async def create(
        self,
        username: str,
        *,
        password_hash: str | None = None,
        email: str | None = None,
        phone: str | None = None,
        role: UserRole = UserRole.USER,
        status: UserStatus = UserStatus.ACTIVE,
        conn: AsyncConnection[Row] | None = None,
    ) -> User:
        """创建用户账户并返回持久化记录。"""
        username = require_text(username, "username")
        async with self._transaction(conn) as c:
            row = await self._fetch_one(
                c,
                """
                INSERT INTO users (username, email, phone, password_hash, role, status)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING *
                """,
                (username, email, phone, password_hash, _param(role), _param(status)),
            )
        assert row is not None
        return User.model_validate(row)

    async def get(
        self, user_id: UUID | str, *, conn: AsyncConnection[Row] | None = None
    ) -> User | None:
        """按 id 取用户。"""
        async with self._transaction(conn) as c:
            row = await self._fetch_one(
                c, "SELECT * FROM users WHERE id = %s", (as_uuid(user_id),)
            )
        return self._model(User, row)

    async def get_by_username(
        self, username: str, *, conn: AsyncConnection[Row] | None = None
    ) -> User | None:
        """按用户名取用户（登录用）。"""
        async with self._transaction(conn) as c:
            row = await self._fetch_one(
                c, "SELECT * FROM users WHERE username = %s", (username,)
            )
        return self._model(User, row)

    async def get_by_email(
        self, email: str, *, conn: AsyncConnection[Row] | None = None
    ) -> User | None:
        """按邮箱取用户。"""
        async with self._transaction(conn) as c:
            row = await self._fetch_one(
                c, "SELECT * FROM users WHERE email = %s", (email,)
            )
        return self._model(User, row)

    async def list(
        self,
        *,
        status: UserStatus | None = None,
        limit: int = 50,
        offset: int = 0,
        conn: AsyncConnection[Row] | None = None,
    ) -> list[User]:
        """分页列出用户，可按状态过滤。"""
        limit, offset = pagination(limit, offset)
        conditions: list[str] = []
        params: list[Any] = []
        if status is not None:
            conditions.append("status = %s")
            params.append(_param(status))
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.extend([limit, offset])
        async with self._transaction(conn) as c:
            rows = await self._fetch_all(
                c,
                f"SELECT * FROM users {where} ORDER BY created_at DESC LIMIT %s OFFSET %s",
                params,
            )
        return self._models(User, rows)

    async def list_deletion_ready(
        self,
        *,
        limit: int = 20,
        conn: AsyncConnection[Row] | None = None,
    ) -> list[User]:
        """列出已到重试时间的待删除用户；未设置重试时间的优先。"""
        limit, _ = pagination(limit, 0)
        async with self._transaction(conn) as c:
            rows = await self._fetch_all(
                c,
                """
                SELECT * FROM users
                WHERE status = %s
                  AND (delete_next_retry_at IS NULL OR delete_next_retry_at <= now())
                ORDER BY delete_next_retry_at NULLS FIRST, created_at
                LIMIT %s
                """,
                (_param(UserStatus.DELETING), limit),
            )
        return self._models(User, rows)

    async def update(
        self,
        user_id: UUID | str,
        *,
        email: Any = _UNSET,
        phone: Any = _UNSET,
        password_hash: Any = _UNSET,
        role: Any = _UNSET,
        status: Any = _UNSET,
        conn: AsyncConnection[Row] | None = None,
    ) -> User:
        """按非 _UNSET 字段局部更新用户。"""
        sets: list[str] = []
        params: list[Any] = []
        for column, value in (
            ("email", email),
            ("phone", phone),
            ("password_hash", password_hash),
            ("role", role),
            ("status", status),
        ):
            if value is not _UNSET:
                sets.append(f"{column} = %s")
                params.append(_param(value))
        if not sets:
            raise ValidationError("没有需要更新的字段")
        sets.append("updated_at = now()")
        params.append(as_uuid(user_id))

        async with self._transaction(conn) as c:
            row = await self._fetch_one(
                c,
                f"UPDATE users SET {', '.join(sets)} WHERE id = %s RETURNING *",
                params,
            )
        if row is None:
            raise NotFoundError("用户", user_id)
        return User.model_validate(row)

    async def mark_login(
        self,
        user_id: UUID | str,
        *,
        at: datetime | None = None,
        conn: AsyncConnection[Row] | None = None,
    ) -> User:
        """记录登录时间并返回最新用户记录。"""
        async with self._transaction(conn) as c:
            row = await self._fetch_one(
                c,
                """
                UPDATE users
                SET last_login_at = COALESCE(%s, now()), updated_at = now()
                WHERE id = %s
                RETURNING *
                """,
                (at, as_uuid(user_id)),
            )
        if row is None:
            raise NotFoundError("用户", user_id)
        return User.model_validate(row)

    async def soft_delete(
        self,
        user_id: UUID | str,
        *,
        conn: AsyncConnection[Row] | None = None,
    ) -> User:
        """发起用户删除：先软删 PG，worker 再清理 Weaviate UserFact/对象存储。"""
        async with self._transaction(conn) as c:
            row = await self._fetch_one(
                c,
                """
                UPDATE users
                SET status = 'deleting',
                    deleted_at = COALESCE(deleted_at, now()),
                    updated_at = now()
                WHERE id = %s AND status <> 'deleted'
                RETURNING *
                """,
                (as_uuid(user_id),),
            )
        if row is None:
            raise NotFoundError("用户", user_id)
        return User.model_validate(row)

    async def mark_deleted(
        self,
        user_id: UUID | str,
        *,
        conn: AsyncConnection[Row] | None = None,
    ) -> User:
        """跨存储清理完成后的终态；保留 tombstone 供审计。"""
        async with self._transaction(conn) as c:
            row = await self._fetch_one(
                c,
                """
                UPDATE users
                SET status = 'deleted',
                    deleted_at = COALESCE(deleted_at, now()),
                    updated_at = now()
                WHERE id = %s
                RETURNING *
                """,
                (as_uuid(user_id),),
            )
        if row is None:
            raise NotFoundError("用户", user_id)
        return User.model_validate(row)

    async def record_delete_failure(
        self,
        user_id: UUID | str,
        error: str,
        *,
        retry_at: datetime | None = None,
        conn: AsyncConnection[Row] | None = None,
    ) -> User:
        """记录用户跨存储删除失败并安排重试。"""
        async with self._transaction(conn) as c:
            row = await self._fetch_one(
                c,
                """
                UPDATE users
                SET delete_attempts = delete_attempts + 1,
                    delete_error = %s,
                    delete_next_retry_at = %s,
                    updated_at = now()
                WHERE id = %s
                RETURNING *
                """,
                (error, retry_at, as_uuid(user_id)),
            )
        if row is None:
            raise NotFoundError("用户", user_id)
        return User.model_validate(row)

    async def delete(
        self, user_id: UUID | str, *, conn: AsyncConnection[Row] | None = None
    ) -> bool:
        """物理删除用户记录，返回是否实际删除。"""
        async with self._transaction(conn) as c:
            affected = await self._execute(
                c, "DELETE FROM users WHERE id = %s", (as_uuid(user_id),)
            )
        return affected > 0


class PermissionRepository(BaseRepository):
    """权限码定义与用户授权。

    鉴权一律查 ``user_permissions.permission_code``，不解析 description，
    也不因为 ``users.role = admin`` 跳过校验。
    """

    async def create(
        self,
        permission_code: str,
        description: str = "",
        *,
        conn: AsyncConnection[Row] | None = None,
    ) -> Permission:
        """创建权限项；权限码全局唯一。"""
        permission_code = require_text(permission_code, "permission_code")
        async with self._transaction(conn) as c:
            row = await self._fetch_one(
                c,
                """
                INSERT INTO permissions (permission_code, description)
                VALUES (%s, %s)
                RETURNING *
                """,
                (permission_code, description),
            )
        assert row is not None
        return Permission.model_validate(row)

    async def get(
        self, permission_id: UUID | str, *, conn: AsyncConnection[Row] | None = None
    ) -> Permission | None:
        """按 id 取权限。"""
        async with self._transaction(conn) as c:
            row = await self._fetch_one(
                c, "SELECT * FROM permissions WHERE id = %s", (as_uuid(permission_id),)
            )
        return self._model(Permission, row)

    async def get_by_code(
        self, permission_code: str, *, conn: AsyncConnection[Row] | None = None
    ) -> Permission | None:
        """按权限码取权限。"""
        async with self._transaction(conn) as c:
            row = await self._fetch_one(
                c,
                "SELECT * FROM permissions WHERE permission_code = %s",
                (permission_code,),
            )
        return self._model(Permission, row)

    async def list(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        conn: AsyncConnection[Row] | None = None,
    ) -> list[Permission]:
        """分页列出权限项。"""
        limit, offset = pagination(limit, offset)
        async with self._transaction(conn) as c:
            rows = await self._fetch_all(
                c,
                "SELECT * FROM permissions ORDER BY permission_code LIMIT %s OFFSET %s",
                (limit, offset),
            )
        return self._models(Permission, rows)

    async def grant(
        self,
        user_id: UUID | str,
        permission_code: str,
        *,
        conn: AsyncConnection[Row] | None = None,
    ) -> UserPermission:
        """授予权限；重复授权幂等（刷新冗余的 permission_code）。"""
        async with self._transaction(conn) as c:
            row = await self._fetch_one(
                c,
                """
                INSERT INTO user_permissions (user_id, permission_id, permission_code)
                SELECT %s, p.id, p.permission_code
                FROM permissions p
                WHERE p.permission_code = %s
                ON CONFLICT (user_id, permission_id)
                DO UPDATE SET permission_code = EXCLUDED.permission_code
                RETURNING *
                """,
                (as_uuid(user_id), permission_code),
            )
        if row is None:
            raise NotFoundError("权限码", permission_code)
        return UserPermission.model_validate(row)

    async def revoke(
        self,
        user_id: UUID | str,
        permission_code: str,
        *,
        conn: AsyncConnection[Row] | None = None,
    ) -> bool:
        """撤销用户的某个权限，返回是否实际删除。"""
        async with self._transaction(conn) as c:
            affected = await self._execute(
                c,
                "DELETE FROM user_permissions WHERE user_id = %s AND permission_code = %s",
                (as_uuid(user_id), permission_code),
            )
        return affected > 0

    async def list_for_user(
        self, user_id: UUID | str, *, conn: AsyncConnection[Row] | None = None
    ) -> list[UserPermission]:
        """列出用户已授予的权限关系。"""
        async with self._transaction(conn) as c:
            rows = await self._fetch_all(
                c,
                """
                SELECT * FROM user_permissions
                WHERE user_id = %s
                ORDER BY permission_code
                """,
                (as_uuid(user_id),),
            )
        return self._models(UserPermission, rows)

    async def list_codes_for_user(
        self, user_id: UUID | str, *, conn: AsyncConnection[Row] | None = None
    ) -> set[str]:
        """登录时加载到内存用户态的权限码集合。"""
        async with self._transaction(conn) as c:
            rows = await self._fetch_all(
                c,
                "SELECT permission_code FROM user_permissions WHERE user_id = %s",
                (as_uuid(user_id),),
            )
        return {row["permission_code"] for row in rows}

    async def has_permission(
        self,
        user_id: UUID | str,
        permission_code: str,
        *,
        conn: AsyncConnection[Row] | None = None,
    ) -> bool:
        """判断用户是否拥有指定权限。"""
        async with self._transaction(conn) as c:
            value = await self._fetch_val(
                c,
                """
                SELECT 1 FROM user_permissions
                WHERE user_id = %s AND permission_code = %s
                LIMIT 1
                """,
                (as_uuid(user_id), permission_code),
            )
        return value is not None

    async def list_user_ids_with_permission(
        self,
        permission_code: str,
        *,
        limit: int = 1000,
        offset: int = 0,
        conn: AsyncConnection[Row] | None = None,
    ) -> list[UUID]:
        """分页列出拥有指定权限的用户 id。"""
        limit, offset = pagination(limit, offset)
        async with self._transaction(conn) as c:
            rows = await self._fetch_all(
                c,
                """
                SELECT user_id FROM user_permissions
                WHERE permission_code = %s
                ORDER BY user_id
                LIMIT %s OFFSET %s
                """,
                (permission_code, limit, offset),
            )
        return [row["user_id"] for row in rows]


__all__ = ["UserRepository", "PermissionRepository"]
