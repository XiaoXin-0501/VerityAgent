"""知识库管理 API：仅管理员可用。

- 创建/列出知识库；
- 上传文档：MinIO 原件 -> PG documents -> Indexer 后台入库；
- 删除文档：交给 Indexer 做 PG/Weaviate/Neo4j/MinIO 跨存储删除；
- 查询入库任务进度。

对话侧仍通过 ``search_kb`` 工具使用知识库，本路由只做管理面。
"""

from __future__ import annotations

import logging
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from pydantic import BaseModel, Field

from api.routers.auth import require_admin, verify_password
from config import settings
from exception import (
    NotFoundError,
    PermissionDeniedError,
    UnavailableError,
    ValidationError,
)
from rag.indexer.enrichment.domain import BUILTIN_DOMAIN_DIR, load_domain_profile
from rag.public import DocumentSource, IndexDocumentRequest
from store.minio import MinioStorage
from store.minio.paths import kb_source_key, sanitize_extension
from store.pg.models.enums import DocumentStatus, KnowledgeBaseStatus
from store.pg.models.knowledge import Document, KnowledgeBase
from store.pg.models.user import User
from store.pg.repositories import DocumentRepository, KnowledgeBaseRepository

logger = logging.getLogger(__name__)

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
    # 访问该库所需权限码；为空使用 RAG__DEFAULT_KB_PERMISSION_CODE
    required_permission_code: str | None = Field(default=None, max_length=100)
    # 该库绑定的领域画像（如 medical / pet_medical）；为空使用全局默认
    domain_profile: str | None = Field(default=None, max_length=100)


class UpdateKnowledgeBaseStatusRequest(BaseModel):
    """启用/停用知识库；可选切换领域画像（切换后需 REINDEX）。"""

    status: Literal["active", "disabled"]
    domain_profile: str | None = Field(default=None, max_length=100)


class DeleteKnowledgeBaseRequest(BaseModel):
    """删除知识库前二次确认管理员密码。"""

    password: str = Field(min_length=1, max_length=128)


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
        "domain_profile": kb.domain_profile,
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


def _available_domain_profiles() -> list[str]:
    """列出内置/自定义目录中的领域画像名，供管理端选择。"""
    names = {item.stem for item in BUILTIN_DOMAIN_DIR.glob("*.json")}
    custom_dir = settings.rag.indexer.domain_profile_dir
    if custom_dir:
        names.update(item.stem for item in Path(custom_dir).expanduser().glob("*.json"))
    custom_path = settings.rag.indexer.domain_profile_path
    if custom_path:
        names.add(Path(custom_path).stem)
    return sorted(names)


def _validate_domain_profile(name: str | None) -> str | None:
    """校验并归一领域画像名；为空返回 None（使用全局默认）。"""
    value = (name or "").strip()
    if not value:
        return None
    load_domain_profile(
        value,
        directory=settings.rag.indexer.domain_profile_dir,
    )
    return value


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
        "default_permission_code": settings.rag.default_kb_permission_code,
        "domain_profiles": _available_domain_profiles(),
        "default_domain_profile": cfg.domain_profile,
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
    permission_code = (
        payload.required_permission_code
        or settings.rag.default_kb_permission_code
    ).strip()
    if not permission_code:
        raise ValidationError("required_permission_code 不能为空")
    kb = await KnowledgeBaseRepository().create(
        payload.name,
        permission_code,
        description=payload.description,
        domain_profile=_validate_domain_profile(payload.domain_profile),
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


@router.patch("/{kb_id}")
async def update_knowledge_base_status(
    kb_id: UUID,
    payload: UpdateKnowledgeBaseStatusRequest,
    user: User = Depends(require_admin),
) -> dict[str, Any]:
    """启用/停用知识库；停用后检索不可见，文档与索引保留。"""
    await _require_kb(kb_id)
    updates: dict[str, Any] = {"status": KnowledgeBaseStatus(payload.status)}
    if "domain_profile" in payload.model_fields_set:
        updates["domain_profile"] = _validate_domain_profile(
            payload.domain_profile
        )
    kb = await KnowledgeBaseRepository().update(kb_id, **updates)
    return _kb_payload(kb)


@router.delete("/{kb_id}", status_code=202)
async def delete_knowledge_base(
    kb_id: UUID,
    payload: DeleteKnowledgeBaseRequest,
    request: Request,
    user: User = Depends(require_admin),
) -> dict[str, Any]:
    """删除知识库：校验管理员密码后软删并提交文档清理任务。

    知识库先被标记 deleting/deleted_at，立即从管理列表与检索中消失；
    Weaviate/Neo4j/MinIO/PG 文档清理由 Indexer 后台任务完成。
    """
    if not verify_password(payload.password, user.password_hash):
        logger.warning(
            "管理员删除知识库密码校验失败: user=%s kb=%s", user.id, kb_id
        )
        raise PermissionDeniedError(
            "管理员密码错误",
            details={"kb_id": str(kb_id)},
        )
    await _require_kb(kb_id)
    documents = DocumentRepository()
    rows: list[Document] = []
    offset = 0
    while True:
        page = await documents.list_by_kb(kb_id, limit=200, offset=offset)
        rows.extend(page)
        if len(page) < 200:
            break
        offset += 200
    pending = [
        item
        for item in rows
        if item.status not in (DocumentStatus.DELETING, DocumentStatus.DELETED)
    ]
    # 有文档时必须先确认 Indexer 可用，避免只删元数据、留下索引孤儿。
    indexer = _indexer(request) if pending else None
    if pending:
        assert indexer is not None

    kb = await KnowledgeBaseRepository().soft_delete(kb_id)
    submitted = 0
    failed = 0
    if pending:
        assert indexer is not None
        for item in pending:
            try:
                await indexer.delete(item.id)
                submitted += 1
            except Exception:  # noqa: BLE001 - 单文档失败不影响知识库隐藏
                failed += 1
                logger.warning(
                    "知识库文档删除任务提交失败: kb=%s doc=%s",
                    kb_id,
                    item.id,
                    exc_info=True,
                )
    return {
        "knowledge_base": _kb_payload(kb),
        "document_count": len(pending),
        "submitted": submitted,
        "failed": failed,
    }


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
    document: Document | None = None
    try:
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
        job = await indexer.index(index_request)
    except Exception:
        # PG 登记失败时清理刚上传的对象，避免 MinIO 孤儿文件；
        # 入库提交失败时保留原件并置 failed，便于后续重试/删除。
        if document is None:
            try:
                await storage.delete_kb_document(
                    kb.id, doc_id, extension=extension
                )
            except Exception:  # noqa: BLE001 - 清理失败不覆盖原始异常
                logger.warning("清理上传孤儿对象失败: %s", object_key)
        else:
            try:
                await documents.mark_failed(document.id)
            except Exception:  # noqa: BLE001 - 状态回写失败不覆盖原始异常
                logger.warning("文档置 failed 失败: %s", document.id)
        raise
    assert document is not None
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
