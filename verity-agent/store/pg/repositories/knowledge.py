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
    IndexingStage,
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
        embedding_model_id: str,
        embedding_dimensions: int,
        embedding_version: str = "v1",
        description: str | None = None,
        status: KnowledgeBaseStatus = KnowledgeBaseStatus.ACTIVE,
        conn: AsyncConnection[Row] | None = None,
    ) -> KnowledgeBase:
        """创建知识库并记录其向量模型与权限要求。"""
        name = require_text(name, "name")
        required_permission_code = require_text(
            required_permission_code, "required_permission_code"
        )
        embedding_model_id = require_text(embedding_model_id, "embedding_model_id")
        embedding_version = require_text(embedding_version, "embedding_version")
        if embedding_dimensions <= 0:
            raise ValidationError("embedding_dimensions 必须 > 0")
        async with self._transaction(conn) as c:
            row = await self._fetch_one(
                c,
                """
                INSERT INTO knowledge_bases
                    (name, description, required_permission_code,
                     embedding_model_id, embedding_dimensions, embedding_version,
                     status)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING *
                """,
                (
                    name,
                    description,
                    required_permission_code,
                    embedding_model_id,
                    embedding_dimensions,
                    embedding_version,
                    _param(status),
                ),
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
        """按 id 取知识库；默认过滤软删除。"""
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
        """分页列出知识库，可按状态过滤。"""
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
        """按非 _UNSET 字段局部更新知识库。"""
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
                SET status = 'deleting',
                    deleted_at = COALESCE(deleted_at, now()),
                    index_version = index_version
                        + CASE WHEN status = 'deleting' THEN 0 ELSE 1 END,
                    updated_at = now()
                WHERE id = %s
                RETURNING *
                """,
                (as_uuid(kb_id),),
            )
        if row is None:
            raise NotFoundError("知识库", kb_id)
        return KnowledgeBase.model_validate(row)

    async def bump_index_version(
        self,
        kb_id: UUID | str,
        *,
        conn: AsyncConnection[Row] | None = None,
    ) -> int:
        """KB 级索引版本 +1；缓存 key 用它失效（D-54）。"""
        async with self._transaction(conn) as c:
            value = await self._fetch_val(
                c,
                """
                UPDATE knowledge_bases
                SET index_version = index_version + 1, updated_at = now()
                WHERE id = %s AND deleted_at IS NULL
                RETURNING index_version
                """,
                (as_uuid(kb_id),),
            )
        if value is None:
            raise NotFoundError("知识库", kb_id)
        return int(value)

    async def update_embedding(
        self,
        kb_id: UUID | str,
        *,
        embedding_model_id: str,
        embedding_dimensions: int,
        embedding_version: str,
        conn: AsyncConnection[Row] | None = None,
    ) -> KnowledgeBase:
        """切换 embedding 模型/维度；调用方负责整库重灌（D-57）。"""
        embedding_model_id = require_text(embedding_model_id, "embedding_model_id")
        embedding_version = require_text(embedding_version, "embedding_version")
        if embedding_dimensions <= 0:
            raise ValidationError("embedding_dimensions 必须 > 0")
        async with self._transaction(conn) as c:
            row = await self._fetch_one(
                c,
                """
                UPDATE knowledge_bases
                SET embedding_model_id = %s,
                    embedding_dimensions = %s,
                    embedding_version = %s,
                    index_version = index_version + 1,
                    updated_at = now()
                WHERE id = %s AND deleted_at IS NULL
                RETURNING *
                """,
                (
                    embedding_model_id,
                    embedding_dimensions,
                    embedding_version,
                    as_uuid(kb_id),
                ),
            )
        if row is None:
            raise NotFoundError("知识库", kb_id)
        return KnowledgeBase.model_validate(row)

    async def record_delete_failure(
        self,
        kb_id: UUID | str,
        error: str,
        *,
        retry_at: Any | None = None,
        conn: AsyncConnection[Row] | None = None,
    ) -> KnowledgeBase:
        """记录知识库跨存储删除失败并安排重试。"""
        async with self._transaction(conn) as c:
            row = await self._fetch_one(
                c,
                """
                UPDATE knowledge_bases
                SET delete_attempts = delete_attempts + 1,
                    delete_error = %s,
                    delete_next_retry_at = %s
                WHERE id = %s
                RETURNING *
                """,
                (error, retry_at, as_uuid(kb_id)),
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
        """恢复被软删除的知识库。"""
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
        """判断用户是否有权访问指定知识库。"""
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
                details={"kb_id": str(kb.id), "status": kb.status.value},
            )
        if not await self.is_accessible(user_id, kb_id, conn=conn):
            raise PermissionDeniedError(
                "没有访问该知识库的权限（缺少 required_permission_code）",
                details={
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
        """分页列出用户有权访问的知识库。"""
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
        description: str | None = None,
        parse_manifest_key: str | None = None,
        status: DocumentStatus = DocumentStatus.PENDING,
        conn: AsyncConnection[Row] | None = None,
    ) -> Document:
        """登记文档元数据；索引前状态为 pending。"""
        original_filename = require_text(original_filename, "original_filename")
        source_object_key = require_text(source_object_key, "source_object_key")
        if size_bytes < 0:
            raise ValidationError("size_bytes 不能为负数")
        async with self._transaction(conn) as c:
            row = await self._fetch_one(
                c,
                """
                INSERT INTO documents
                    (kb_id, original_filename, description, extension, size_bytes,
                     sha256, source_object_key, parse_manifest_key, uploaded_by, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING *
                """,
                (
                    as_uuid(kb_id),
                    original_filename,
                    description,
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
        """按 id 取文档；默认过滤软删除。"""
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
        """分页列出知识库下的文档。"""
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
        """统计知识库下的文档数量。"""
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
                "SELECT * FROM documents WHERE id = %s FOR UPDATE",
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
            elif status in (DocumentStatus.DELETING, DocumentStatus.DELETED):
                sets.append("deleted_at = COALESCE(deleted_at, now())")
            params.append(as_uuid(doc_id))

            row = await self._fetch_one(
                c,
                f"UPDATE documents SET {', '.join(sets)} WHERE id = %s RETURNING *",
                params,
            )
        assert row is not None
        return Document.model_validate(row)

    # ——————————————————— 入库任务（D-77） ———————————————————

    async def set_description(
        self,
        doc_id: UUID | str,
        description: str | None,
        *,
        conn: AsyncConnection[Row] | None = None,
    ) -> Document:
        """写入/清空文档描述；``None`` 表示待人工补充（P1-17）。"""
        async with self._transaction(conn) as c:
            row = await self._fetch_one(
                c,
                """
                UPDATE documents
                SET description = %s, updated_at = now()
                WHERE id = %s AND deleted_at IS NULL
                RETURNING *
                """,
                (description, as_uuid(doc_id)),
            )
        if row is None:
            raise NotFoundError("文档", doc_id)
        return Document.model_validate(row)

    async def claim_index_jobs(
        self,
        worker_id: str,
        *,
        limit: int = 1,
        lease_seconds: int = 300,
        doc_id: UUID | str | None = None,
        conn: AsyncConnection[Row] | None = None,
    ) -> list[Document]:
        """领取待入库/失败重试/租约过期的文档。

        - ``index_attempts`` 在领取时 +1；
        - 只有 ``index_locked_until`` 为空或已过期的文档可被领取；
        - ``FOR UPDATE SKIP LOCKED`` 支持多 worker。
        """
        limit, _ = pagination(limit, 0)
        if lease_seconds <= 0:
            raise ValidationError("lease_seconds 必须 > 0")
        worker = require_text(worker_id, "worker_id")
        scope = as_uuid(doc_id) if doc_id is not None else None
        async with self._transaction(conn) as c:
            rows = await self._fetch_all(
                c,
                """
                WITH claimed AS (
                    SELECT id
                    FROM documents
                    WHERE deleted_at IS NULL
                      AND status IN ('pending', 'processing', 'failed')
                      AND (index_next_retry_at IS NULL OR index_next_retry_at <= now())
                      AND (index_locked_until IS NULL OR index_locked_until < now())
                      -- failed + 有错误 + 无下次重试时间 = 超过上限的 dead，不再领取
                      AND NOT (status = 'failed'
                               AND index_error IS NOT NULL
                               AND index_next_retry_at IS NULL)
                      AND (%s::uuid IS NULL OR id = %s::uuid)
                    ORDER BY
                        CASE status WHEN 'pending' THEN 0
                                    WHEN 'failed' THEN 1
                                    ELSE 2 END,
                        created_at
                    LIMIT %s
                    FOR UPDATE SKIP LOCKED
                )
                UPDATE documents d
                SET status = 'processing',
                    index_attempts = index_attempts + 1,
                    index_locked_by = %s,
                    index_locked_until = now() + make_interval(secs => %s),
                    updated_at = now()
                FROM claimed
                WHERE d.id = claimed.id
                RETURNING d.*
                """,
                (scope, scope, limit, worker, lease_seconds),
            )
        return self._models(Document, rows)

    async def claim_index_job(
        self,
        worker_id: str,
        *,
        lease_seconds: int = 300,
        doc_id: UUID | str | None = None,
        conn: AsyncConnection[Row] | None = None,
    ) -> Document | None:
        """领取单个待索引文档（claim_index_jobs 的 limit=1 包装）。"""
        docs = await self.claim_index_jobs(
            worker_id,
            limit=1,
            lease_seconds=lease_seconds,
            doc_id=doc_id,
            conn=conn,
        )
        return docs[0] if docs else None

    async def acquire_index_lease(
        self,
        doc_id: UUID | str,
        *,
        worker_id: str,
        lease_seconds: int = 300,
        conn: AsyncConnection[Row] | None = None,
    ) -> Document:
        """直接执行入库时获取文档租约并置 processing。

        ``claim_index_jobs`` 是 worker 批量领取路径；``Indexer.index()`` 直接
        调用流水线时没有 claim，需要在这里补 `index_locked_by/until`，否则后续
        `update_index_progress/mark_index_completed` 的租约校验会查不到行。
        """
        if lease_seconds <= 0:
            raise ValidationError("lease_seconds 必须 > 0")
        worker = require_text(worker_id, "worker_id")
        async with self._transaction(conn) as c:
            row = await self._fetch_one(
                c,
                """
                UPDATE documents
                SET status = 'processing',
                    index_locked_by = %s,
                    index_locked_until = now() + make_interval(secs => %s),
                    updated_at = now()
                WHERE id = %s
                  AND deleted_at IS NULL
                  AND status IN ('pending', 'processing', 'failed', 'completed', 'deleting')
                  -- 不抢占其他 worker 未过期的租约；同一 worker 可续租。
                  AND (
                      index_locked_until IS NULL
                      OR index_locked_until < now()
                      OR index_locked_by = %s
                  )
                RETURNING *
                """,
                (worker, lease_seconds, as_uuid(doc_id), worker),
            )
        if row is None:
            raise NotFoundError("文档", doc_id)
        return Document.model_validate(row)

    async def update_index_progress(
        self,
        doc_id: UUID | str,
        *,
        stage: IndexingStage,
        progress: float,
        worker_id: str | None = None,
        include_deleted: bool = False,
        conn: AsyncConnection[Row] | None = None,
    ) -> Document:
        """更新入库阶段/进度；删除流程用 ``include_deleted=True``。"""
        if not 0.0 <= progress <= 1.0:
            raise ValidationError("index_progress 必须在 0~1 之间")
        conditions = ["id = %s"]
        if not include_deleted:
            conditions.append("deleted_at IS NULL")
        params: list[Any] = [_param(stage), progress, as_uuid(doc_id)]
        if worker_id is not None:
            conditions.append("index_locked_by = %s")
            params.append(worker_id)
        async with self._transaction(conn) as c:
            row = await self._fetch_one(
                c,
                f"""
                UPDATE documents
                SET index_stage = %s, index_progress = %s, updated_at = now()
                WHERE {' AND '.join(conditions)}
                RETURNING *
                """,
                params,
            )
        if row is None:
            raise NotFoundError("文档", doc_id)
        return Document.model_validate(row)

    async def mark_index_failed(
        self,
        doc_id: UUID | str,
        *,
        error: str,
        worker_id: str | None = None,
        retry_backoff_s: float = 30.0,
        max_attempts: int = 5,
        conn: AsyncConnection[Row] | None = None,
    ) -> Document:
        """入库失败：写错误、清租约、按指数退避；超限不再排期（dead）。

        ``documents.status`` 仍为 failed；job 的 dead 语义由
        ``index_error`` + ``index_next_retry_at IS NULL`` 投影（D-77）。
        """
        if retry_backoff_s <= 0:
            raise ValidationError("retry_backoff_s 必须 > 0")
        if max_attempts < 1:
            raise ValidationError("max_attempts 必须 >= 1")
        conditions = ["id = %s", "deleted_at IS NULL"]
        params: list[Any] = [error, max_attempts, retry_backoff_s, as_uuid(doc_id)]
        if worker_id is not None:
            conditions.append("index_locked_by = %s")
            params.append(worker_id)
        async with self._transaction(conn) as c:
            row = await self._fetch_one(
                c,
                f"""
                UPDATE documents
                SET status = 'failed',
                    index_error = %s,
                    index_next_retry_at = CASE
                        WHEN index_attempts >= %s THEN NULL
                        ELSE now() + make_interval(secs => %s * power(2, index_attempts))
                    END,
                    index_locked_until = NULL,
                    index_locked_by = NULL,
                    updated_at = now()
                WHERE {' AND '.join(conditions)}
                RETURNING *
                """,
                params,
            )
        if row is None:
            raise NotFoundError("文档", doc_id)
        return Document.model_validate(row)

    async def mark_index_completed(
        self,
        doc_id: UUID | str,
        *,
        token_count: int | None = None,
        chunk_count: int | None = None,
        parse_manifest_key: str | None = None,
        index_version: int | None = None,
        worker_id: str | None = None,
        conn: AsyncConnection[Row] | None = None,
    ) -> Document:
        """三者全部成功后才置 completed（D-56）。"""
        conditions = ["id = %s", "deleted_at IS NULL", "status = 'processing'"]
        params: list[Any] = [
            token_count,
            chunk_count,
            parse_manifest_key,
            index_version,
            as_uuid(doc_id),
        ]
        if worker_id is not None:
            conditions.append("index_locked_by = %s")
            params.append(worker_id)
        async with self._transaction(conn) as c:
            row = await self._fetch_one(
                c,
                f"""
                UPDATE documents
                SET status = 'completed',
                    token_count = COALESCE(%s, token_count),
                    chunk_count = COALESCE(%s, chunk_count),
                    parse_manifest_key = COALESCE(%s, parse_manifest_key),
                    index_version = COALESCE(%s, index_version),
                    index_stage = 'DONE',
                    index_progress = 1,
                    index_error = NULL,
                    index_next_retry_at = NULL,
                    index_locked_until = NULL,
                    index_locked_by = NULL,
                    parsed_at = now(),
                    updated_at = now()
                WHERE {' AND '.join(conditions)}
                RETURNING *
                """,
                params,
            )
        if row is None:
            raise NotFoundError("文档", doc_id)
        return Document.model_validate(row)

    async def release_index_lease(
        self,
        doc_id: UUID | str,
        *,
        worker_id: str,
        conn: AsyncConnection[Row] | None = None,
    ) -> bool:
        """释放文档索引租约；仅持有者可释放。"""
        async with self._transaction(conn) as c:
            affected = await self._execute(
                c,
                """
                UPDATE documents
                SET index_locked_until = NULL, index_locked_by = NULL, updated_at = now()
                WHERE id = %s AND index_locked_by = %s
                """,
                (as_uuid(doc_id), worker_id),
            )
        return affected > 0

    async def mark_processing(
        self, doc_id: UUID | str, *, conn: AsyncConnection[Row] | None = None
    ) -> Document:
        """把文档置为处理中。"""
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
        """把文档置为已完成并记录 token/chunk 统计。"""
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
        """把文档置为失败。"""
        return await self.set_status(doc_id, DocumentStatus.FAILED, conn=conn)

    async def mark_deleting(
        self, doc_id: UUID | str, *, conn: AsyncConnection[Row] | None = None
    ) -> Document:
        """把文档置为删除中（跨存储删除补偿）。"""
        return await self.set_status(doc_id, DocumentStatus.DELETING, conn=conn)

    async def mark_deleted(
        self, doc_id: UUID | str, *, conn: AsyncConnection[Row] | None = None
    ) -> Document:
        """把文档置为已删除。"""
        return await self.set_status(doc_id, DocumentStatus.DELETED, conn=conn)


__all__ = ["KnowledgeBaseRepository", "DocumentRepository"]
