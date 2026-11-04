"""knowledge_bases / documents 行模型。"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from .base import DBModel
from .enums import DocumentStatus, KnowledgeBaseStatus


class KnowledgeBase(DBModel):
    id: UUID
    name: str
    description: str | None = None
    # 访问该库所需权限码，查询时必须真实校验用户是否拥有
    required_permission_code: str
    status: KnowledgeBaseStatus
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None


class Document(DBModel):
    id: UUID
    kb_id: UUID
    original_filename: str
    extension: str
    size_bytes: int
    sha256: str | None = None
    source_object_key: str
    parse_manifest_key: str | None = None
    uploaded_by: UUID
    status: DocumentStatus
    token_count: int | None = None
    chunk_count: int | None = None
    created_at: datetime
    updated_at: datetime
    parsed_at: datetime | None = None
    deleted_at: datetime | None = None
