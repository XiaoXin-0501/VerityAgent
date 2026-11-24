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
from api.upload_tracker import kb_uploads
from config import settings
from exception import (
    ConflictError,
    NotFoundError,
    PermissionDeniedError,
    UnavailableError,
    ValidationError,
)
from config.domain import list_domain_profile_names, load_domain_profile
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

# 上传分块大小：UploadFile 本身由 Starlette 落到临时文件，这里按块读取，
# 避免几百 MB 的文件一次性进入进程内存。
_UPLOAD_CHUNK_BYTES = 1024 * 1024

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


async def _list_all_documents(kb_id: UUID) -> list[Document]:
    """分页拉取知识库下全部未删除文档，供删除/统计使用。"""
    documents = DocumentRepository()
    rows: list[Document] = []
    offset = 0
    while True:
        page = await documents.list_by_kb(kb_id, limit=200, offset=offset)
        rows.extend(page)
        if len(page) < 200:
            break
        offset += 200
    return rows


def _domain_profile_directory() -> str | None:
    """解析画像目录；``path`` 指向单个文件时使用其父目录。"""
    directory = settings.rag.indexer.domain_profile_dir
    configured_path = settings.rag.indexer.domain_profile_path
    if configured_path:
        path_obj = Path(configured_path).expanduser()
        if path_obj.is_file() and not directory:
            return str(path_obj.parent)
    return directory


def _available_domain_profiles() -> list[str]:
    """列出内置/自定义目录中的领域画像名，供管理端选择。"""
    return list_domain_profile_names(
        path=settings.rag.indexer.domain_profile_path,
        directory=_domain_profile_directory(),
    )


def _domain_profile_details() -> list[dict[str, Any]]:
    """返回画像的展示信息；主画像标注 ``is_master``，便于前端分组。"""
    details: list[dict[str, Any]] = []
    for name in _available_domain_profiles():
        try:
            profile = load_domain_profile(
                name,
                path=(
                    settings.rag.indexer.domain_profile_path
                    if settings.rag.indexer.domain_profile_path
                    and Path(settings.rag.indexer.domain_profile_path).stem == name
                    else None
                ),
                directory=_domain_profile_directory(),
            )
        except Exception:  # noqa: BLE001 - 单个坏画像不影响选项列表
            continue
        details.append(
            {
                "name": profile.name,
                "display_name": profile.display_name or profile.name,
                "description": profile.description,
                "master": profile.master or None,
                "is_master": profile.is_master,
            }
        )
    return details


def _validate_domain_profile(name: str | None) -> str | None:
    """校验并归一领域画像名；为空返回 None（使用全局默认）。"""
    value = (name or "").strip()
    if not value:
        return None
    load_domain_profile(
        value,
        directory=_domain_profile_directory(),
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
        # 知识库文档上限；与聊天附件上限解耦，支持较大的专业文档
        "max_file_bytes": settings.rag.indexer.max_source_bytes,
        "default_permission_code": settings.rag.default_kb_permission_code,
        "domain_profiles": _available_domain_profiles(),
        "domain_profile_details": _domain_profile_details(),
        "default_domain_profile": cfg.domain_profile,
        "master_domain_profile": cfg.master_profile,
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
    """删除知识库：校验管理员密码后停止上传、软删并提交文档清理任务。

    顺序很重要：
    1. 预检 Indexer，避免只删元数据、留下索引孤儿；
    2. 向该库的在途上传发取消信号并等待退出，避免删除过程中又登记新文档；
    3. 软删知识库，立即从管理列表与检索中消失；
    4. 重新拉一次文档列表（兜底跨实例晚到登记），逐文档提交跨存储删除。
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

    rows = await _list_all_documents(kb_id)
    pending = [
        item
        for item in rows
        if item.status not in (DocumentStatus.DELETING, DocumentStatus.DELETED)
    ]
    # 有文档时必须先确认 Indexer 可用，避免只删元数据、留下索引孤儿；
    # 这个检查放在取消上传之前，避免删除最终失败却先打断了上传。
    indexer = _indexer(request) if pending else getattr(
        request.app.state, "indexer", None
    )

    # 停止本实例的在途上传；跨实例上传由“写后原子校验 KB 状态”兜底。
    cancelled = kb_uploads.cancel_kb(kb_id)
    settled = await kb_uploads.wait_settled(
        kb_id,
        settings.rag.indexer.cancel_wait_s,
    )
    if cancelled and not settled:
        logger.warning(
            "删除知识库时仍有上传未退出，继续删除并依赖登记校验: kb=%s active=%d",
            kb_id,
            kb_uploads.active_count(kb_id),
        )

    kb = await KnowledgeBaseRepository().soft_delete(kb_id)

    # 软删后补拉一次：跨实例上传可能在第一步之后、软删之前完成登记。
    seen = {item.id: item for item in pending}
    for item in await _list_all_documents(kb_id):
        if item.status in (DocumentStatus.DELETING, DocumentStatus.DELETED):
            continue
        seen.setdefault(item.id, item)
    pending = list(seen.values())
    failed = 0
    if pending and indexer is None:
        # 空库删除后仍出现文档，说明跨实例上传抢在软删前登记；此时没有
        # Indexer 只能记录，等待运维补跑清理。
        failed = len(pending)
        logger.error(
            "知识库删除后仍有待清理文档但 Indexer 不可用: kb=%s docs=%d",
            kb_id,
            len(pending),
        )
        pending = []

    submitted = 0
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
        "document_count": submitted + failed,
        "submitted": submitted,
        "failed": failed,
        "cancelled_uploads": cancelled,
        "uploads_settled": settled,
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


def _upload_cancelled(kb_id: UUID) -> ConflictError:
    """构造“知识库删除导致上传取消”的统一错误。"""
    return ConflictError(
        "知识库正在删除或已删除，上传已取消",
        details={"kb_id": str(kb_id)},
    )


async def _delete_uploaded_object(
    storage: MinioStorage,
    kb_id: UUID,
    doc_id: UUID,
    extension: str,
    object_key: str,
) -> None:
    """清理失败/取消的上传对象；清理失败只记日志，不覆盖原始异常。"""
    try:
        await storage.delete_kb_document(kb_id, doc_id, extension=extension)
    except Exception:  # noqa: BLE001 - 清理失败不覆盖原始异常
        logger.warning("清理上传孤儿对象失败: %s", object_key)


@router.post("/{kb_id}/documents", status_code=201)
async def upload_document(
    kb_id: UUID,
    request: Request,
    file: UploadFile = File(...),
    description: str | None = Form(default=None),
    user: User = Depends(require_admin),
) -> dict[str, Any]:
    """上传原件到 MinIO（分块流式）、登记文档并提交后台入库。

    上传期间登记取消令牌；删除知识库会先置位令牌，读文件循环在分块边界
    退出；写 MinIO 后再检查一次，并用 ``require_active_kb`` 原子登记兜底
    跨实例的并发删除。
    """
    kb = await _require_kb(kb_id)
    if kb.status is not KnowledgeBaseStatus.ACTIVE:
        raise ConflictError(
            "知识库未启用，不能上传文档",
            details={"kb_id": str(kb_id), "status": kb.status.value},
        )
    indexer = _indexer(request)

    filename = (file.filename or "").strip()
    if not filename:
        raise ValidationError("文件名不能为空")
    extension = sanitize_extension(Path(filename).suffix)
    _validate_extension(extension)

    max_bytes = settings.rag.indexer.max_source_bytes
    if file.size is not None and file.size > max_bytes:
        raise ValidationError(
            "文件超过大小上限",
            details={"size_bytes": file.size, "max_file_bytes": max_bytes},
        )

    doc_id = uuid4()
    object_key = kb_source_key(kb.id, doc_id, extension)
    mime_type = file.content_type or "application/octet-stream"
    digest = sha256()
    size = 0
    sha = ""
    storage = MinioStorage()
    token = kb_uploads.begin(kb.id)
    document: Document | None = None

    try:
        # 1) 分块读取：每块都响应取消信号，避免删除知识库后继续吞大文件。
        while True:
            if token.is_cancelled:
                raise _upload_cancelled(kb.id)
            chunk = await file.read(_UPLOAD_CHUNK_BYTES)
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                raise ValidationError(
                    "文件超过大小上限",
                    details={
                        "size_bytes": size,
                        "max_file_bytes": max_bytes,
                    },
                )
            digest.update(chunk)
        if size == 0:
            raise ValidationError("文件内容为空")
        sha = digest.hexdigest()
        if token.is_cancelled:
            raise _upload_cancelled(kb.id)

        # 2) 关键：不把 UploadFile 一次性 read() 进内存。seek(0) 后把同一个
        # 临时文件句柄交给 MinIO put_object 分片流式上传。
        await file.seek(0)
        await storage.put_kb_source_stream(
            kb.id,
            doc_id,
            extension,
            file.file,
            size,
            content_type=mime_type,
            metadata={
                "uploaded_by": str(user.id),
                "sha256": sha,
            },
        )
        if token.is_cancelled:
            raise _upload_cancelled(kb.id)
        # 跨实例删除兜底：上传完成后重新确认知识库仍可写。
        latest = await KnowledgeBaseRepository().get(kb.id)
        if latest is None or latest.status is not KnowledgeBaseStatus.ACTIVE:
            raise _upload_cancelled(kb.id)

        # 3) 原子登记：KB 已在并发删除时 INSERT 不会产生行，避免孤儿文档。
        # 主键沿用前面生成 doc_id：MinIO source key 已按它写入，复用同一个
        # id 才能保证删除/清理时按 PG doc_id 找到原件。
        document = await DocumentRepository().create(
            kb.id,
            filename,
            extension,
            size,
            object_key,
            user.id,
            sha256=sha,
            description=(description or "").strip() or None,
            status=DocumentStatus.PENDING,
            require_active_kb=True,
            doc_id=doc_id,
        )
        logger.info(
            "知识库文档上传完成: kb=%s doc=%s size_bytes=%d",
            kb.id,
            document.id,
            size,
        )
    except Exception:
        # 文档尚未登记成功：清理已写对象，避免 MinIO 孤儿文件。
        await _delete_uploaded_object(
            storage, kb.id, doc_id, extension, object_key
        )
        raise
    finally:
        kb_uploads.end(token)
        await file.close()

    assert document is not None
    try:
        index_request = IndexDocumentRequest(
            kb_id=kb.id,
            doc_id=document.id,
            source=DocumentSource(
                object_key=object_key,
                filename=filename,
                extension=extension,
                mime_type=mime_type,
                size_bytes=size,
                sha256=sha,
                uploaded_by=user.id,
            ),
        )
        job = await indexer.index(index_request)
    except Exception:
        # 登记成功但入库提交失败：保留原件并置 failed，便于后续重试/删除。
        try:
            await DocumentRepository().mark_failed(document.id)
        except Exception:  # noqa: BLE001 - 状态回写失败不覆盖原始异常
            logger.warning("文档置 failed 失败: %s", document.id)
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
