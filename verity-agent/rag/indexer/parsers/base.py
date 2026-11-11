"""文档解析器协议与公共辅助。"""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from rag.shared.enums import DocumentType
from rag.shared.models import DocumentProfile, DocumentSource, ParsedDocument

__all__ = ["DocumentParser", "detect_profile"]

_EXTENSION_MAP: dict[str, DocumentType] = {
    # v1 只启用这五种格式；其他格式走 UNKNOWN，由 pipeline 拒绝。
    ".pdf": DocumentType.PDF,
    ".docx": DocumentType.DOCX,
    ".md": DocumentType.MARKDOWN,
    ".markdown": DocumentType.MARKDOWN,
    ".txt": DocumentType.TEXT,
    ".xlsx": DocumentType.SPREADSHEET,
    ".xlsm": DocumentType.SPREADSHEET,
}

# 扩展名缺失/错误时，用 MIME 兜底；仍只覆盖 v1 支持的格式。
_MIME_MAP: dict[str, DocumentType] = {
    "application/pdf": DocumentType.PDF,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": DocumentType.DOCX,
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": DocumentType.SPREADSHEET,
    "text/markdown": DocumentType.MARKDOWN,
    "text/x-markdown": DocumentType.MARKDOWN,
    "text/plain": DocumentType.TEXT,
}



def detect_profile(source: DocumentSource) -> DocumentProfile:
    """按扩展名/MIME 做轻量画像；真正能力探活由 parser 更新。"""
    extension = source.extension.lower()
    if not extension.startswith("."):
        extension = f".{extension}"
    doc_type = _EXTENSION_MAP.get(extension)
    if doc_type is None:
        doc_type = _MIME_MAP.get(source.mime_type.lower(), DocumentType.UNKNOWN)
    return DocumentProfile(
        doc_type=doc_type,
        mime_type=source.mime_type,
        extension=extension,
        languages=[source.language] if source.language else [],
    )


class DocumentParser(Protocol):
    async def parse(
        self,
        source: DocumentSource,
        profile: DocumentProfile,
        *,
        doc_id: UUID,
        kb_id: UUID,
        data: bytes,
    ) -> ParsedDocument:
        """解析原文件字节为 ParsedDocument；失败抛 RagError。"""
        ...
