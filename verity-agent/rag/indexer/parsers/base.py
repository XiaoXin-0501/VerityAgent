"""文档解析器协议与公共辅助。"""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from rag.shared.models import DocumentProfile, DocumentSource, ParsedDocument
from rag.shared.registry import ParserRegistry

__all__ = ["DocumentParser", "detect_profile"]


def detect_profile(
    source: DocumentSource,
    registry: ParserRegistry,
) -> DocumentProfile:
    """按扩展名/MIME 做轻量画像。

    文档类型从 **parser 注册表** 推断，避免在解析器之外再维护一套
    `_EXTENSION_MAP / _MIME_MAP`；真正能力探活仍由 parser 更新。
    """
    extension = source.extension.lower()
    if extension and not extension.startswith("."):
        extension = f".{extension}"
    doc_type = registry.detect_doc_type(extension, source.mime_type)
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
