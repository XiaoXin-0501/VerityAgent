"""session_files 仓库。

会话文件只在所属会话范围内检索，不自动进公共知识库；过期由 MinIO
bucket TTL 回收，PG 不维护 expires_at。
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from psycopg import AsyncConnection

from exception import NotFoundError, ValidationError

from ..connection import Row
from ..models.enums import SESSION_FILE_TRANSITIONS, SessionFileStatus
from ..models.session_file import SessionFile
from .base import (
    BaseRepository,
    _UNSET,
    _param,
    as_uuid,
    ensure_transition,
    normalize_extension,
    normalize_sha256,
    pagination,
    require_text,
)


class SessionFileRepository(BaseRepository):
    """会话内上传文件的元数据与状态机。"""

    async def create(
        self,
        conversation_id: UUID | str,
        uploaded_by: UUID | str,
        original_filename: str,
        extension: str,
        size_bytes: int,
        source_object_key: str,
        *,
        message_id: UUID | str | None = None,
        sha256: str | None = None,
        parse_manifest_key: str | None = None,
        status: SessionFileStatus = SessionFileStatus.UPLOADING,
        conn: AsyncConnection[Row] | None = None,
    ) -> SessionFile:
        original_filename = require_text(original_filename, "original_filename")
        source_object_key = require_text(source_object_key, "source_object_key")
        if size_bytes < 0:
            raise ValidationError("size_bytes 不能为负数")
        async with self._transaction(conn) as c:
            row = await self._fetch_one(
                c,
                """
                INSERT INTO session_files
                    (conversation_id, message_id, uploaded_by, original_filename,
                     extension, size_bytes, sha256, source_object_key,
                     parse_manifest_key, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING *
                """,
                (
                    as_uuid(conversation_id),
                    as_uuid(message_id) if message_id is not None else None,
                    as_uuid(uploaded_by),
                    original_filename,
                    normalize_extension(extension),
                    size_bytes,
                    normalize_sha256(sha256),
                    source_object_key,
                    parse_manifest_key,
                    _param(status),
                ),
            )
        assert row is not None
        return SessionFile.model_validate(row)

    async def get(
        self,
        file_id: UUID | str,
        *,
        conn: AsyncConnection[Row] | None = None,
    ) -> SessionFile | None:
        async with self._transaction(conn) as c:
            row = await self._fetch_one(
                c, "SELECT * FROM session_files WHERE id = %s", (as_uuid(file_id),)
            )
        return self._model(SessionFile, row)

    async def list_by_conversation(
        self,
        conversation_id: UUID | str,
        *,
        status: SessionFileStatus | None = None,
        limit: int = 50,
        offset: int = 0,
        conn: AsyncConnection[Row] | None = None,
    ) -> list[SessionFile]:
        limit, offset = pagination(limit, offset)
        conditions = ["conversation_id = %s"]
        params: list[Any] = [as_uuid(conversation_id)]
        if status is not None:
            conditions.append("status = %s")
            params.append(_param(status))
        params.extend([limit, offset])
        async with self._transaction(conn) as c:
            rows = await self._fetch_all(
                c,
                f"""
                SELECT * FROM session_files
                WHERE {' AND '.join(conditions)}
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
                """,
                params,
            )
        return self._models(SessionFile, rows)

    async def set_status(
        self,
        file_id: UUID | str,
        status: SessionFileStatus,
        *,
        token_count: Any = _UNSET,
        parse_manifest_key: Any = _UNSET,
        conn: AsyncConnection[Row] | None = None,
    ) -> SessionFile:
        """按状态机迁移；``available`` 表示解析产物已可检索。"""
        try:
            status = SessionFileStatus(status)
        except ValueError as exc:
            raise ValidationError(f"非法会话文件状态: {status}") from exc

        async with self._transaction(conn) as c:
            current = await self._fetch_one(
                c,
                "SELECT * FROM session_files WHERE id = %s FOR UPDATE",
                (as_uuid(file_id),),
            )
            if current is None:
                raise NotFoundError("会话文件", file_id)
            ensure_transition(
                SessionFileStatus(current["status"]),
                status,
                SESSION_FILE_TRANSITIONS,
                resource="会话文件",
            )

            sets = ["status = %s"]
            params: list[Any] = [_param(status)]
            if token_count is not _UNSET:
                sets.append("token_count = %s")
                params.append(token_count)
            if parse_manifest_key is not _UNSET:
                sets.append("parse_manifest_key = %s")
                params.append(parse_manifest_key)
            params.append(as_uuid(file_id))

            row = await self._fetch_one(
                c,
                f"UPDATE session_files SET {', '.join(sets)} WHERE id = %s RETURNING *",
                params,
            )
        assert row is not None
        return SessionFile.model_validate(row)

    async def mark_processing(
        self, file_id: UUID | str, *, conn: AsyncConnection[Row] | None = None
    ) -> SessionFile:
        return await self.set_status(file_id, SessionFileStatus.PROCESSING, conn=conn)

    async def mark_available(
        self,
        file_id: UUID | str,
        *,
        token_count: int | None = None,
        parse_manifest_key: str | None = None,
        conn: AsyncConnection[Row] | None = None,
    ) -> SessionFile:
        kwargs: dict[str, Any] = {}
        if token_count is not None:
            kwargs["token_count"] = token_count
        if parse_manifest_key is not None:
            kwargs["parse_manifest_key"] = parse_manifest_key
        return await self.set_status(
            file_id, SessionFileStatus.AVAILABLE, conn=conn, **kwargs
        )

    async def mark_failed(
        self, file_id: UUID | str, *, conn: AsyncConnection[Row] | None = None
    ) -> SessionFile:
        return await self.set_status(file_id, SessionFileStatus.FAILED, conn=conn)

    async def mark_deleting(
        self, file_id: UUID | str, *, conn: AsyncConnection[Row] | None = None
    ) -> SessionFile:
        return await self.set_status(file_id, SessionFileStatus.DELETING, conn=conn)

    async def mark_deleted(
        self, file_id: UUID | str, *, conn: AsyncConnection[Row] | None = None
    ) -> SessionFile:
        return await self.set_status(file_id, SessionFileStatus.DELETED, conn=conn)


__all__ = ["SessionFileRepository"]
