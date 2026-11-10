"""users / permissions / user_permissions 行模型。"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from .base import DBModel
from .enums import UserRole, UserStatus

__all__ = ["User", "Permission", "UserPermission"]


class User(DBModel):
    id: UUID
    username: str
    email: str | None = None
    phone: str | None = None
    password_hash: str | None = None
    role: UserRole
    status: UserStatus
    last_login_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    # 跨存储删除状态机（PG 软删 → Weaviate → MinIO → 完成）
    deleted_at: datetime | None = None
    delete_attempts: int = 0
    delete_error: str | None = None
    delete_next_retry_at: datetime | None = None


class Permission(DBModel):
    id: UUID
    permission_code: str
    description: str
    created_at: datetime


class UserPermission(DBModel):
    user_id: UUID
    permission_id: UUID
    # 冗余自 permissions.permission_code，鉴权直接查本表，不 join
    permission_code: str
