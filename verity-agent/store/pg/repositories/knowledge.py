"""knowledge_bases / documents 仓库。"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from psycopg import AsyncConnection

from exception import NotFoundError, PermissionDeniedError, ValidationError

from ..connection import Row
from ..models.enums import (
    DOCUMENT_TRANSITIONS,
    DocumentStatus,
    KnowledgeBaseStatus,
)
from ..models.knowledge import Document, KnowledgeBase
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


class KnowledgeBaseRepository(BaseRepository):
    """知识库元数据与访问控制。

    访问判断必须是"当前用户真实拥有 ``required_permission_code``"，
    不能只检查码是否存在（见设计文档 §3 关键约束 1）。
    """

    async def create(
        self,
        name: str,
        required_permission_code: str,
        *,
        description: str | None = None,
        status: KnowledgeBaseStatus = KnowledgeBaseStatus.ACTIVE,
        conn: AsyncConnection[Row] | None = None,
    ) -> KnowledgeBase:
        name = require_text(name, "name")
        required_permission_code = require_text(
            required_permission_code, "required_permission_code"
        )
        async with self._transaction(conn) as c:
            row = await self._fetch_one(
                c,
                """
                INSERT INTO knowledge_bases
                    (name, description, required_permission_code, status)
                VALUES (%s, %s, %s, %s)
                RETURNING *
                """,
                (name, description, required_permission_code, _param(status)),
            )
        assert row is not None
        return KnowledgeBase.model_validate(row)

    async def get(
        self,
        kb_id: UUID | str,
        *,
        include_deleted: bool = False,
        conn: AsyncConnection[Row] | None = None,
    ) -> KnowledgeBase | None:
        sql = "SELECT * FROM knowledge_bases WHERE id = %s"
        if not include_deleted:
            sql += " AND deleted_at IS NULL"
        async with self._transaction(conn) as c:
            row = await self._fetch_one(c, sql, (as_uuid(kb_id),))
        return self._model(KnowledgeBase, row)

    async def list(
        self,
        *,
        status: KnowledgeBaseStatus | None = None,
        include_deleted: bool = False,
        limit: int = 50,
        offset: int = 0,
        conn: AsyncConnection[Row] | None = None,
    ) -> list[KnowledgeBase]:
        limit, offset = pagination(limit, offset)
        conditions: list[str] = []
        params: list[Any] = []
        if not include_deleted:
            conditions.append("deleted_at IS NULL")
        if status is not None:
            conditions.append("status = %s")
            params.append(_param(status))
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.extend([limit, offset])
        async with self._transaction(conn) as c:
            rows = await self._fetch_all(
                c,
                f"""
                SELECT * FROM knowledge_bases {where}
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
                """,
                params,
            )
        return self._models(KnowledgeBase, rows)

    async def update(
        self,
        kb_id: UUID | str,
        *,
        name: Any = _UNSET,
        description: Any = _UNSET,
        required_permission_code: Any = _UNSET,
        status: Any = _UNSET,
        conn: AsyncConnection[Row] | None = None,
    ) -> KnowledgeBase:
        sets: list[str] = []
        params: list[Any] = []
        for column, value in (
            ("name", name),
            ("description", description),
            ("required_permission_code", required_permission_code),
            ("status", status),
        ):
            if value is not _UNSET:
                sets.append(f"{column} = %s")
                params.append(_param(value))
        if not sets:
            raise ValidationError("没有需要更新的字段")
        sets.append("updated_at = now()")
        params.append(as_uuid(kb_id))

        async with self._transaction(conn) as c:
            row = await self._fetch_one(
                c,
                f"""
                UPDATE knowledge_bases SET {', '.join(sets)}
                WHERE id = %s AND deleted_at IS NULL
                RETURNING *
                """,
                params,
            )
        if row is None:
            raise NotFoundError("知识库", kb_id)
        return KnowledgeBase.model_validate(row)

    async def soft_delete(
        self,
        kb_id: UUID | str,
        *,
        conn: AsyncConnection[Row] | None = None,
    ) -> KnowledgeBase:
        """软删除：标记 deleting + deleted_at，等待后台清理 Weaviate/Neo4j/MinIO。"""
        async with self._transaction(conn) as c:
            row = await self._fetch_one(
                c,
                """
                UPDATE knowledge_bases
                SET status = 'deleting', deleted_at = now(), updated_at = now()
                WHERE id = %s AND deleted_at IS NULL
                RETURNING *
                """,
                (as_uuid(kb_id),),
            )
        if row is None:
            raise NotFoundError("知识库", kb_id)
        return KnowledgeBase.model_validate(row)

    async def restore(
        self,
        kb_id: UUID | str,
        *,
        conn: AsyncConnection[Row] | None = None,
    ) -> KnowledgeBase:
        async with self._transaction(conn) as c:
            row = await self._fetch_one(
                c,
                """
                UPDATE knowledge_bases
                SET status = 'active', deleted_at = NULL, updated_at = now()
                WHERE id = %s
                RETURNING *
                """,
                (as_uuid(kb_id),),
            )
        if row is None:
            raise NotFoundError("知识库", kb_id)
        return KnowledgeBase.model_validate(row)

    # ——————————————————— 访问控制 ———————————————————

    async def is_accessible(
        self,
        user_id: UUID | str,
        kb_id: UUID | str,
        *,
        conn: AsyncConnection[Row] | None = None,
    ) -> bool:
        async with self._transaction(conn) as c:
            value = await self._fetch_val(
                c,
                """
                SELECT 1
                FROM knowledge_bases kb
                JOIN user_permissions up
                  ON up.user_id = %s
                 AND up.permission_code = kb.required_permission_code
                WHERE kb.id = %s
                  AND kb.status = 'active'
                  AND kb.deleted_at IS NULL
                LIMIT 1
                """,
                (as_uuid(user_id), as_uuid(kb_id)),
            )
        return value is not None

    async def get_for_user(
        self,
        user_id: UUID | str,
        kb_id: UUID | str,
        *,
        conn: AsyncConnection[Row] | None = None,
    ) -> KnowledgeBase:
        """取知识库并强制校验访问权限。

        - 不存在 / 已删除 -> NotFoundError
        - 已停用 / 无 required_permission_code -> PermissionDeniedError
        """
        kb = await self.get(kb_id, conn=conn)
        if kb is None:
            raise NotFoundError("知识库", kb_id)
        if kb.status is not KnowledgeBaseStatus.ACTIVE:
            raise PermissionDeniedError(
                f"知识库当前不可用（status={kb.status}）",
                detail={"kb_id": str(kb.id), "status": kb.status.value},
            )
        if not await self.is_accessible(user_id, kb_id, conn=conn):
            raise PermissionDeniedError(
                "没有访问该知识库的权限（缺少 required_permission_code）",
                detail={
                    "kb_id": str(kb.id),
                    "required_permission_code": kb.required_permission_code,
                },
            )
        return kb

    async def list_accessible(
        self,
        user_id: UUID | str,
        *,
        limit: int = 50,
        offset: int = 0,
        conn: AsyncConnection[Row] | None = None,
    ) -> list[KnowledgeBase]:
        limit, offset = pagination(limit, offset)
        async with self._transaction(conn) as c:
            rows = await self._fetch_all(
                c,
                """
                SELECT kb.*
                FROM knowledge_bases kb
                JOIN user_permissions up
                  ON up.user_id = %s
                 AND up.permission_code = kb.required_permission_code
                WHERE kb.status = 'active'
                  AND kb.deleted_at IS NULL
                ORDER BY kb.created_at DESC
                LIMIT %s OFFSET %s
                """,
                (as_uuid(user_id), limit, offset),
            )
        return self._models(KnowledgeBase, rows)


class DocumentRepository(BaseRepository):
    """知识库文档。

    ``completed`` 表示"解析 + Weaviate 写入 + Neo4j 引用建完"全部成功，
    仓库层只负责状态机，跨存储的编排由业务层完成。
    """

    async def create(
        self,
        kb_id: UUID | str,
        original_filename: str,
        extension: str,
        size_bytes: int,
        source_object_key: str,
        uploaded_by: UUID | str,
        *,
        sha256: str | None = None,
        parse_manifest_key: str | None = None,
        status: DocumentStatus = DocumentStatus.PENDING,
        conn: AsyncConnection[Row] | None = None,
    ) -> Document:
        original_filename = require_text(original_filename, "original_filename")
        source_object_key = require_text(source_object_key, "source_object_key")
        if size_bytes < 0:
            raise ValidationError("size_bytes 不能为负数")
        async with self._transaction(conn) as c:
            row = await self._fetch_one(
                c,
                """
                INSERT INTO documents
                    (kb_id, original_filename, extension, size_bytes, sha256,
                     source_object_key, parse_manifest_key, uploaded_by, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING *
                """,
                (
                    as_uuid(kb_id),
                    original_filename,
                    normalize_extension(extension),
                    size_bytes,
                    normalize_sha256(sha256),
                    source_object_key,
                    parse_manifest_key,
                    as_uuid(uploaded_by),
                    _param(status),
                ),
            )
        assert row is not None
        return Document.model_validate(row)

    async def get(
        self,
        doc_id: UUID | str,
        *,
        include_deleted: bool = False,
        conn: AsyncConnection[Row] | None = None,
    ) -> Document | None:
        sql = "SELECT * FROM documents WHERE id = %s"
        if not include_deleted:
            sql += " AND deleted_at IS NULL"
        async with self._transaction(conn) as c:
            row = await self._fetch_one(c, sql, (as_uuid(doc_id),))
        return self._model(Document, row)

    async def list_by_kb(
        self,
        kb_id: UUID | str,
        *,
        status: DocumentStatus | None = None,
        include_deleted: bool = False,
        limit: int = 50,
        offset: int = 0,
        conn: AsyncConnection[Row] | None = None,
    ) -> list[Document]:
        limit, offset = pagination(limit, offset)
        conditions = ["kb_id = %s"]
        params: list[Any] = [as_uuid(kb_id)]
        if not include_deleted:
            conditions.append("deleted_at IS NULL")
        if status is not None:
            conditions.append("status = %s")
            params.append(_param(status))
        params.extend([limit, offset])
        async with self._transaction(conn) as c:
            rows = await self._fetch_all(
                c,
                f"""
                SELECT * FROM documents
                WHERE {' AND '.join(conditions)}
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
                """,
                params,
            )
        return self._models(Document, rows)

    async def find_by_sha256(
        self,
        kb_id: UUID | str,
        sha256: str,
        *,
        conn: AsyncConnection[Row] | None = None,
    ) -> Document | None:
        """按内容哈希查同库重复文档（上传去重）。"""
        async with self._transaction(conn) as c:
            row = await self._fetch_one(
                c,
                """
                SELECT * FROM documents
                WHERE kb_id = %s AND sha256 = %s AND deleted_at IS NULL
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (as_uuid(kb_id), normalize_sha256(sha256)),
            )
        return self._model(Document, row)

    async def count_by_kb(
        self,
        kb_id: UUID | str,
        *,
        conn: AsyncConnection[Row] | None = None,
    ) -> int:
        async with self._transaction(conn) as c:
            value = await self._fetch_val(
                c,
                """
                SELECT count(*) FROM documents
                WHERE kb_id = %s AND deleted_at IS NULL
                """,
                (as_uuid(kb_id),),
            )
        return int(value or 0)

    async def set_status(
        self,
        doc_id: UUID | str,
        status: DocumentStatus,
        *,
        token_count: Any = _UNSET,
        chunk_count: Any = _UNSET,
        parse_manifest_key: Any = _UNSET,
        conn: AsyncConnection[Row] | None = None,
    ) -> Document:
        """按状态机迁移文档状态。

        先 ``SELECT ... FOR UPDATE`` 锁行，避免并发写入把状态机踩乱。
        ``completed`` 时顺带写 ``parsed_at``，``deleted`` 时写 ``deleted_at``。
        """
        if not isinstance(status, DocumentStatus):
            status = DocumentStatus(status)
        async with self._transaction(conn) as c:
            current = await self._fetch_one(
                c,
                "SELECT * FROM documents WHERE id = %s AND deleted_at IS NULL FOR UPDATE",
                (as_uuid(doc_id),),
            )
            if current is None:
                raise NotFoundError("文档", doc_id)
            current_status = DocumentStatus(current["status"])
            ensure_transition(
                current_status, status, DOCUMENT_TRANSITIONS, resource="文档"
            )

            sets = ["status = %s", "updated_at = now()"]
            params: list[Any] = [_param(status)]
            if token_count is not _UNSET:
                sets.append("token_count = %s")
                params.append(token_count)
            if chunk_count is not _UNSET:
                sets.append("chunk_count = %s")
                params.append(chunk_count)
            if parse_manifest_key is not _UNSET:
                sets.append("parse_manifest_key = %s")
                params.append(parse_manifest_key)
            if status is DocumentStatus.COMPLETED:
                sets.append("parsed_at = now()")
            elif status is DocumentStatus.DELETED:
                sets.append("deleted_at = COALESCE(deleted_at, now())")
            params.append(as_uuid(doc_id))

            row = await self._fetch_one(
                c,
                f"UPDATE documents SET {', '.join(sets)} WHERE id = %s RETURNING *",
                params,
            )
        assert row is not None
        return Document.model_validate(row)

    async def mark_processing(
        self, doc_id: UUID | str, *, conn: AsyncConnection[Row] | None = None
    ) -> Document:
        return await self.set_status(doc_id, DocumentStatus.PROCESSING, conn=conn)

    async def mark_completed(
        self,
        doc_id: UUID | str,
        *,
        token_count: int | None = None,
        chunk_count: int | None = None,
        parse_manifest_key: str | None = None,
        conn: AsyncConnection[Row] | None = None,
    ) -> Document:
        kwargs: dict[str, Any] = {}
        if token_count is not None:
            kwargs["token_count"] = token_count
        if chunk_count is not None:
            kwargs["chunk_count"] = chunk_count
        if parse_manifest_key is not None:
            kwargs["parse_manifest_key"] = parse_manifest_key
        return await self.set_status(
            doc_id, DocumentStatus.COMPLETED, conn=conn, **kwargs
        )

    async def mark_failed(
        self, doc_id: UUID | str, *, conn: AsyncConnection[Row] | None = None
    ) -> Document:
        return await self.set_status(doc_id, DocumentStatus.FAILED, conn=conn)

    async def mark_deleting(
        self, doc_id: UUID | str, *, conn: AsyncConnection[Row] | None = None
    ) -> Document:
        return await self.set_status(doc_id, DocumentStatus.DELETING, conn=conn)

    async def mark_deleted(
        self, doc_id: UUID | str, *, conn: AsyncConnection[Row] | None = None
    ) -> Document:
        return await self.set_status(doc_id, DocumentStatus.DELETED, conn=conn)


__all__ = ["KnowledgeBaseRepository", "DocumentRepository"]
