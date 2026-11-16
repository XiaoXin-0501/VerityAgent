"""knowledge_bases / documents 行模型。"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from .base import DBModel
from .enums import DocumentStatus, IndexingStage, KnowledgeBaseStatus

__all__ = ["KnowledgeBase", "Document"]


class KnowledgeBase(DBModel):
    id: UUID
    name: str
    description: str | None = None
    # 访问该库所需权限码，查询时必须真实校验用户是否拥有
    required_permission_code: str
    # 该库绑定的领域画像名；NULL 回退到 IndexerConfig.domain_profile
    domain_profile: str | None = None
    # KB 级缓存令牌；入库/重切片/删除时递增（D-54）
    index_version: int = 0
    # 整库重灌依据（D-57）
    embedding_model_id: str
    embedding_dimensions: int
    embedding_version: str = "v1"
    status: KnowledgeBaseStatus
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None
    delete_attempts: int = 0
    delete_error: str | None = None
    delete_next_retry_at: datetime | None = None


class Document(DBModel):
    id: UUID
    kb_id: UUID
    original_filename: str
    # 文档级描述；NULL 表示待人工补充（P1-17）
    description: str | None = None
    extension: str
    size_bytes: int
    sha256: str | None = None
    source_object_key: str
    parse_manifest_key: str | None = None
    uploaded_by: UUID
    status: DocumentStatus
    token_count: int | None = None
    chunk_count: int | None = None
    # 入库任务（D-77）：复用 documents，不新增任务表
    index_job_id: UUID | None = None
    index_stage: IndexingStage | None = None
    index_progress: float | None = None
    index_attempts: int = 0
    index_error: str | None = None
    index_next_retry_at: datetime | None = None
    index_locked_until: datetime | None = None
    index_locked_by: str | None = None
    index_version: int | None = None
    created_at: datetime
    updated_at: datetime
    parsed_at: datetime | None = None
    deleted_at: datetime | None = None
    delete_attempts: int = 0
    delete_error: str | None = None
    delete_next_retry_at: datetime | None = None
