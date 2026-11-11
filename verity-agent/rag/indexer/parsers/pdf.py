"""PDF parser（PyMuPDF 主，pdfplumber fallback，v1 只抽文本）。

- 只读取文本层；PDF 内嵌图片不 OCR、不建 FigureBlock、不进入 chunk；
- 每页一个文本块，保留页码；切片由后续 TextChunker 完成；
- 文本覆盖率为“有文本页 / 总页数”，纯图片 PDF 会标记 `is_scanned=True`，
  由质量闸门决定是否拒绝；
- 未安装依赖时抛 `RagError(INVALID_CONFIG)`，不阻塞其他 parser 注册。
"""

from __future__ import annotations

import io
from uuid import UUID

from exception import RagError, RagErrorCode
from rag.shared.enums import BlockType, DocumentType
from rag.shared.models import DocumentProfile, DocumentSource, ParsedDocument

from ._common import make_block, make_parsed_document

__all__ = ["PdfParser"]


class PdfParser:
    parser_id = "pdf_pymupdf"
    parser_version = "1"

    async def parse(
        self,
        source: DocumentSource,
        profile: DocumentProfile,
        *,
        doc_id: UUID,
        kb_id: UUID,
        data: bytes,
    ) -> ParsedDocument:
        try:
            import pymupdf as fitz  # type: ignore[import-not-found]
        except ImportError:
            try:
                import fitz  # type: ignore[import-not-found]
            except ImportError:
                return await self._parse_pdfplumber(
                    source, profile, doc_id=doc_id, kb_id=kb_id, data=data
                )

        try:
            document = fitz.open(stream=data, filetype="pdf")
        except Exception as exc:
            raise RagError(
                RagErrorCode.INVALID_REQUEST,
                "PDF 无法打开",
                details={"filename": source.filename, "cause": str(exc)},
            ) from exc

        blocks = []
        page_count = 0
        pages_with_text = 0
        try:
            page_count = document.page_count
            for page_index in range(page_count):
                page = document.load_page(page_index)
                text = str(page.get_text("text") or "").strip()
                if not text:
                    continue
                pages_with_text += 1
                blocks.append(
                    make_block(
                        doc_id=doc_id,
                        page_no=page_index + 1,
                        order=len(blocks),
                        block_type=BlockType.PARAGRAPH,
                        text=text,
                        parser_id=self.parser_id,
                    )
                )
        except Exception as exc:
            raise RagError(
                RagErrorCode.INVALID_REQUEST,
                "PDF 解析失败",
                details={"filename": source.filename, "cause": str(exc)},
            ) from exc
        finally:
            document.close()

        coverage = pages_with_text / page_count if page_count else 0.0
        has_text = pages_with_text > 0
        return make_parsed_document(
            source=source,
            doc_id=doc_id,
            kb_id=kb_id,
            parser_id=self.parser_id,
            parser_version=self.parser_version,
            profile=profile.model_copy(
                update={
                    "doc_type": DocumentType.PDF,
                    "page_count": page_count,
                    "has_text_layer": has_text,
                    "has_images": False,
                    "is_scanned": not has_text,
                    "text_coverage": coverage,
                }
            ),
            blocks=blocks,
            warnings=[] if has_text else ["PDF 未检测到文本层，图片/扫描件暂不处理"],
        )

    async def _parse_pdfplumber(
        self,
        source: DocumentSource,
        profile: DocumentProfile,
        *,
        doc_id: UUID,
        kb_id: UUID,
        data: bytes,
    ) -> ParsedDocument:
        try:
            import pdfplumber  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RagError(
                RagErrorCode.INVALID_CONFIG,
                "未安装 PDF parser 依赖（PyMuPDF 或 pdfplumber）",
                details={"parser_id": self.parser_id},
            ) from exc

        blocks = []
        page_count = 0
        pages_with_text = 0
        try:
            with pdfplumber.open(io.BytesIO(data)) as document:
                page_count = len(document.pages)
                for page_index, page in enumerate(document.pages):
                    text = (page.extract_text() or "").strip()
                    if not text:
                        continue
                    pages_with_text += 1
                    blocks.append(
                        make_block(
                            doc_id=doc_id,
                            page_no=page_index + 1,
                            order=len(blocks),
                            block_type=BlockType.PARAGRAPH,
                            text=text,
                            parser_id=self.parser_id,
                        )
                    )
        except Exception as exc:
            raise RagError(
                RagErrorCode.INVALID_REQUEST,
                "PDF 解析失败",
                details={"filename": source.filename, "cause": str(exc)},
            ) from exc

        coverage = pages_with_text / page_count if page_count else 0.0
        has_text = pages_with_text > 0
        return make_parsed_document(
            source=source,
            doc_id=doc_id,
            kb_id=kb_id,
            parser_id=self.parser_id,
            parser_version=self.parser_version,
            profile=profile.model_copy(
                update={
                    "doc_type": DocumentType.PDF,
                    "page_count": page_count,
                    "has_text_layer": has_text,
                    "has_images": False,
                    "is_scanned": not has_text,
                    "text_coverage": coverage,
                }
            ),
            blocks=blocks,
            warnings=[] if has_text else ["PDF 未检测到文本层，图片/扫描件暂不处理"],
        )
