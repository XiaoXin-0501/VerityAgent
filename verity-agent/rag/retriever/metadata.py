"""检索阶段的 PG 元数据接口。

Retriever 需要两类信息：

- 候选 chunk 对应的文档状态：只返回 ``completed`` 文档；
- 最终 top-k 的文档描述/文件名：按 D-103 按 ``doc_id`` 批量补文档级描述；
- KB 索引版本：用于检索缓存 key（D-84）。

这里只依赖窄协议，生产用 ``PgRetrievalMetadataStore``，测试可注入 fake。
"""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from rag.shared.models import DocumentMeta
from store.pg import get_pg


class RetrievalMetadataStore(Protocol):
    """检索所需 PG 元数据的只读协议。"""

    async def get_document_statuses(
        self,
        doc_ids: list[UUID],
    ) -> dict[UUID, str]:
        """批量返回 doc_id → status；不存在/已删除的文档不返回。"""
        ...

    async def get_document_meta(
        self,
        doc_ids: list[UUID],
    ) -> dict[UUID, DocumentMeta]:
        """批量返回文档级元数据/描述；不存在的文档不返回。"""
        ...

    async def get_kb_index_versions(
        self,
        kb_ids: list[UUID],
    ) -> dict[UUID, int]:
        """批量返回知识库索引版本，用于缓存 key。"""
        ...

    async def get_kb_permission_codes(
        self,
        kb_ids: list[UUID],
    ) -> dict[UUID, str]:
        """批量返回知识库所需的权限码（PG ``required_permission_code``）。"""
        ...

    async def get_kb_domain_profiles(
        self,
        kb_ids: list[UUID],
    ) -> dict[UUID, str | None]:
        """批量返回知识库绑定的领域画像名；NULL 表示使用全局默认。"""
        ...


class PgRetrievalMetadataStore:
    """基于 PostgreSQL 的检索元数据适配器；直接走连接池，不拼业务 SQL 到 Retriever。"""

    async def get_document_statuses(
        self,
        doc_ids: list[UUID],
    ) -> dict[UUID, str]:
        """批量读取文档状态；只查未删除行。"""
        if not doc_ids:
            return {}
        async with get_pg().transaction() as conn:
            cursor = await conn.execute(
                """
                SELECT id, status
                FROM documents
                WHERE id = ANY(%s) AND deleted_at IS NULL
                """,
                (doc_ids,),
            )
            rows = await cursor.fetchall()
        return {UUID(str(row["id"])): str(row["status"]) for row in rows}

    async def get_document_meta(
        self,
        doc_ids: list[UUID],
    ) -> dict[UUID, DocumentMeta]:
        """批量读取文档名、描述与更新时间；description 为空时标记待人工。"""
        if not doc_ids:
            return {}
        async with get_pg().transaction() as conn:
            cursor = await conn.execute(
                """
                SELECT id, kb_id, original_filename, description,
                       created_at, updated_at
                FROM documents
                WHERE id = ANY(%s) AND deleted_at IS NULL
                """,
                (doc_ids,),
            )
            rows = await cursor.fetchall()
        result: dict[UUID, DocumentMeta] = {}
        for row in rows:
            doc_id = UUID(str(row["id"]))
            description = row["description"]
            result[doc_id] = DocumentMeta(
                doc_id=doc_id,
                kb_id=UUID(str(row["kb_id"])),
                doc_name=str(row["original_filename"]),
                description=description,
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                description_missing=not bool(description),
            )
        return result

    async def get_kb_index_versions(
        self,
        kb_ids: list[UUID],
    ) -> dict[UUID, int]:
        """批量读取 knowledge_bases.index_version。"""
        if not kb_ids:
            return {}
        async with get_pg().transaction() as conn:
            cursor = await conn.execute(
                """
                SELECT id, index_version
                FROM knowledge_bases
                WHERE id = ANY(%s) AND deleted_at IS NULL
                """,
                (kb_ids,),
            )
            rows = await cursor.fetchall()
        return {
            UUID(str(row["id"])): int(row["index_version"] or 0)
            for row in rows
        }

    async def get_kb_permission_codes(
        self,
        kb_ids: list[UUID],
    ) -> dict[UUID, str]:
        """批量读取 knowledge_bases.required_permission_code。"""
        if not kb_ids:
            return {}
        async with get_pg().transaction() as conn:
            cursor = await conn.execute(
                """
                SELECT id, required_permission_code
                FROM knowledge_bases
                WHERE id = ANY(%s) AND deleted_at IS NULL
                """,
                (kb_ids,),
            )
            rows = await cursor.fetchall()
        return {
            UUID(str(row["id"])): str(row["required_permission_code"])
            for row in rows
            if row["required_permission_code"]
        }

    async def get_kb_domain_profiles(
        self,
        kb_ids: list[UUID],
    ) -> dict[UUID, str | None]:
        """批量读取 knowledge_bases.domain_profile；NULL 原样返回。"""
        if not kb_ids:
            return {}
        async with get_pg().transaction() as conn:
            cursor = await conn.execute(
                """
                SELECT id, domain_profile
                FROM knowledge_bases
                WHERE id = ANY(%s) AND deleted_at IS NULL
                """,
                (kb_ids,),
            )
            rows = await cursor.fetchall()
        return {
            UUID(str(row["id"])): (
                str(row["domain_profile"]) if row["domain_profile"] else None
            )
            for row in rows
        }
