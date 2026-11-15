"""知识库管理 API：仅管理员可用。

- 创建/列出知识库；
- 上传文档：MinIO 原件 -> PG documents -> Indexer 后台入库；
- 删除文档：交给 Indexer 做 PG/Weaviate/Neo4j/MinIO 跨存储删除；
- 查询入库任务进度。

对话侧仍通过 ``search_kb`` 工具使用知识库，本路由只做管理面。
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from pydantic import BaseModel, Field

from api.routers.auth import require_admin
from config import settings
from exception import NotFoundError, UnavailableError, ValidationError
from rag.public import DocumentSource, IndexDocumentRequest
from store.minio import MinioStorage
from store.minio.paths import kb_source_key, sanitize_extension
from store.pg.models.enums import DocumentStatus, KnowledgeBaseStatus
from store.pg.models.knowledge import Document, KnowledgeBase
from store.pg.repositories import DocumentRepository, KnowledgeBaseRepository
from store.pg.models.user import User

router = APIRouter(prefix="/knowledge-bases", tags=["知识库管理"])
index_jobs_router = APIRouter(prefix="/index-jobs", tags=["知识库管理"])

# 扩展名 -> IndexerConfig.enabled_doc_types 的文档类型。
_DOC_TYPE_BY_EXTENSION = {
    "pdf": "pdf",
    "docx": "docx",
    "md": "markdown",
    "markdown": "markdown",
    "txt": "text",
    "text": "text",
    "xlsx": "spreadsheet",
    "xls": "spreadsheet",
    "csv": "spreadsheet",
    "pptx": "pptx",
    "html": "html",
    "htm": "html",
}


class CreateKnowledgeBaseRequest(BaseModel):
    """创建知识库入参。"""

    name: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=2000)
    # 访问该库所需权限码；对话检索时按真实用户权限校验
    required_permission_code: str = Field(default="kb:read", min_length=1, max_length=100)


def _indexer(request: Request):
    """取 lifespan 装配的 Indexer；缺失时返回 503。"""
    indexer = getattr(request.app.state, "indexer", None)
    if indexer is None:
        raise UnavailableError("RAG Indexer 尚未初始化")
    return indexer


def _kb_payload(kb: KnowledgeBase) -> dict[str, Any]:
    """知识库行 -> 管理面 DTO。"""
    return {
        "id": str(kb.id),
        "name": kb.name,
        "description": kb.description,
        "required_permission_code": kb.required_permission_code,
        "embedding_model_id": kb.embedding_model_id,
        "embedding_dimensions": kb.embedding_dimensions,
        "embedding_version": kb.embedding_version,
        "index_version": kb.index_version,
        "status": kb.status.value,
        "created_at": kb.created_at.isoformat(),
        "updated_at": kb.updated_at.isoformat(),
    }


def _document_payload(doc: Document) -> dict[str, Any]:
    """文档行 -> 管理面 DTO。"""
    return {
        "id": str(doc.id),
        "kb_id": str(doc.kb_id),
        "original_filename": doc.original_filename,
        "description": doc.description,
        "extension": doc.extension,
        "size_bytes": doc.size_bytes,
        "sha256": doc.sha256,
        "status": doc.status.value,
        "token_count": doc.token_count,
        "chunk_count": doc.chunk_count,
        "index_job_id": str(doc.index_job_id) if doc.index_job_id else None,
        "index_stage": doc.index_stage.value if doc.index_stage else None,
        "index_progress": doc.index_progress,
        "index_error": doc.index_error,
        "created_at": doc.created_at.isoformat(),
        "updated_at": doc.updated_at.isoformat(),
        "parsed_at": doc.parsed_at.isoformat() if doc.parsed_at else None,
    }


async def _require_kb(kb_id: UUID) -> KnowledgeBase:
    """校验知识库存在且未删除。"""
    kb = await KnowledgeBaseRepository().get(kb_id)
    if kb is None:
        raise NotFoundError("知识库", kb_id)
    return kb


def _validate_extension(extension: str) -> str:
    """校验扩展名在 Indexer 启用的文档类型内。"""
    doc_type = _DOC_TYPE_BY_EXTENSION.get(extension)
    if doc_type is None:
        raise ValidationError(
            "暂不支持该文件类型",
            details={"extension": extension},
        )
    enabled = set(settings.rag.indexer.enabled_doc_types or [])
    if enabled and doc_type not in enabled:
        raise ValidationError(
            "该文档类型未启用",
            details={"doc_type": doc_type, "enabled": sorted(enabled)},
        )
    return doc_type


@router.get("/options")
async def get_knowledge_options(
    user: User = Depends(require_admin),
) -> dict[str, Any]:
    """返回上传限制与启用类型，供前端设置 accept/大小提示。"""
    cfg = settings.rag.indexer
    return {
        "enabled_doc_types": list(cfg.enabled_doc_types),
        "max_file_bytes": settings.message.max_file_bytes,
        "default_permission_code": "kb:read",
        "embedding_model_id": cfg.embedding_model_id,
        "embedding_dimensions": cfg.embedding_dimensions,
        "embedding_version": cfg.embedding_version,
    }


@router.post("", status_code=201)
async def create_knowledge_base(
    payload: CreateKnowledgeBaseRequest,
    user: User = Depends(require_admin),
) -> dict[str, Any]:
    """创建知识库；向量模型/维度来自服务端 Indexer 配置。"""
    cfg = settings.rag.indexer
    if not cfg.embedding_model_id:
        raise UnavailableError("未配置 RAG embedding 模型，无法创建知识库")
    kb = await KnowledgeBaseRepository().create(
        payload.name,
        payload.required_permission_code,
        description=payload.description,
        embedding_model_id=cfg.embedding_model_id,
        embedding_dimensions=cfg.embedding_dimensions,
        embedding_version=cfg.embedding_version,
        status=KnowledgeBaseStatus.ACTIVE,
    )
    return _kb_payload(kb)


@router.get("")
async def list_knowledge_bases(
    limit: int = 50,
    offset: int = 0,
    user: User = Depends(require_admin),
) -> dict[str, Any]:
    """管理员查看全部未删除知识库。"""
    items = await KnowledgeBaseRepository().list(limit=limit, offset=offset)
    return {"items": [_kb_payload(item) for item in items], "next_cursor": None}


@router.get("/{kb_id}/documents")
async def list_documents(
    kb_id: UUID,
    limit: int = 50,
    offset: int = 0,
    user: User = Depends(require_admin),
) -> dict[str, Any]:
    """列出知识库下的文档与入库状态。"""
    await _require_kb(kb_id)
    items = await DocumentRepository().list_by_kb(
        kb_id,
        limit=limit,
        offset=offset,
    )
    return {"items": [_document_payload(item) for item in items], "next_cursor": None}


@router.post("/{kb_id}/documents", status_code=201)
async def upload_document(
    kb_id: UUID,
    request: Request,
    file: UploadFile = File(...),
    description: str | None = Form(default=None),
    user: User = Depends(require_admin),
) -> dict[str, Any]:
    """上传原件到 MinIO、登记文档并提交后台入库。"""
    kb = await _require_kb(kb_id)
    indexer = _indexer(request)

    filename = (file.filename or "").strip()
    if not filename:
        raise ValidationError("文件名不能为空")
    extension = sanitize_extension(Path(filename).suffix)
    _validate_extension(extension)

    max_bytes = settings.message.max_file_bytes
    if file.size is not None and file.size > max_bytes:
        raise ValidationError(
            "文件超过大小上限",
            details={"size_bytes": file.size, "max_file_bytes": max_bytes},
        )
    data = await file.read()
    if len(data) > max_bytes:
        raise ValidationError(
            "文件超过大小上限",
            details={"size_bytes": len(data), "max_file_bytes": max_bytes},
        )
    if not data:
        raise ValidationError("文件内容为空")

    doc_id = uuid4()
    object_key = kb_source_key(kb.id, doc_id, extension)
    mime_type = file.content_type or "application/octet-stream"
    digest = sha256(data).hexdigest()

    storage = MinioStorage()
    await storage.put_kb_source(
        kb.id,
        doc_id,
        extension,
        data,
        content_type=mime_type,
        metadata={"uploaded_by": str(user.id), "sha256": digest},
    )

    documents = DocumentRepository()
    document = await documents.create(
        kb.id,
        filename,
        extension,
        len(data),
        object_key,
        user.id,
        sha256=digest,
        description=(description or "").strip() or None,
        status=DocumentStatus.PENDING,
    )

    index_request = IndexDocumentRequest(
        kb_id=kb.id,
        doc_id=document.id,
        source=DocumentSource(
            object_key=object_key,
            filename=filename,
            extension=extension,
            mime_type=mime_type,
            size_bytes=len(data),
            sha256=digest,
            uploaded_by=user.id,
        ),
    )
    try:
        job = await indexer.index(index_request)
    except Exception:
        # 入库提交失败时把文档置 failed，避免长期停在 pending。
        await documents.mark_failed(document.id)
        raise
    return {
        "document": _document_payload(document),
        "job": job.model_dump(mode="json"),
    }


@router.delete("/{kb_id}/documents/{doc_id}", status_code=202)
async def delete_document(
    kb_id: UUID,
    doc_id: UUID,
    request: Request,
    user: User = Depends(require_admin),
) -> dict[str, Any]:
    """提交跨存储删除任务；实际状态通过 index-jobs 轮询。"""
    await _require_kb(kb_id)
    document = await DocumentRepository().get(doc_id)
    if document is None or document.kb_id != kb_id:
        raise NotFoundError("文档", doc_id)
    job = await _indexer(request).delete(doc_id)
    return {"job": job.model_dump(mode="json")}


@index_jobs_router.get("/{job_id}")
async def get_index_job(
    job_id: str,
    request: Request,
    user: User = Depends(require_admin),
) -> dict[str, Any]:
    """查询入库/删除任务进度。"""
    job = await _indexer(request).get_job(job_id)
    if job is None:
        raise NotFoundError("索引任务", job_id)
    return job.model_dump(mode="json")
