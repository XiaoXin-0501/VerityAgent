"""默认 parser 注册表（v1：PDF / DOCX / Markdown / TXT / XLSX）。

未启用的格式（PPTX、HTML、CSV、代码、图片 OCR）不在注册表内；pipeline 会
在 `SUPPORTED_DOC_TYPES` 层拒绝，避免误走 text fallback。

具体 parser 的第三方依赖按需懒加载；注册本身不需要安装依赖。
"""

from __future__ import annotations

from config.rag import IndexerConfig
from rag.shared.enums import DocumentType
from rag.shared.models import ParserSpec
from rag.shared.registry import ParserRegistry

from .base import DocumentParser, detect_profile
from .docx import DocxParser
from .markdown import MarkdownParser
from .ocr_provider import OcrProvider
from .pdf import PdfParser
from .spreadsheet import SpreadsheetParser
from .text import TextParser

__all__ = [
    "DocumentParser",
    "detect_profile",
    "build_parser_registry",
    "TextParser",
    "MarkdownParser",
    "PdfParser",
    "DocxParser",
    "SpreadsheetParser",
]


def build_parser_registry(
    config: IndexerConfig | None = None,
    *,
    pdf_ocr_provider: OcrProvider | None = None,
) -> ParserRegistry:
    """注册 v1 支持的五个 parser；其他格式不注册。

    ``pdf_ocr_provider`` 用于测试或自定义 OCR 引擎注入；不传时按
    ``IndexerConfig.ocr`` 自动选择。
    """
    registry = ParserRegistry()
    entries = [
        (
            ParserSpec(
                parser_id="text",
                version=TextParser.parser_version,
                doc_types={DocumentType.TEXT},
                extensions={".txt"},
                mime_types={"text/plain"},
                priority=10,
            ),
            TextParser(),
        ),
        (
            ParserSpec(
                parser_id="markdown",
                version=MarkdownParser.parser_version,
                doc_types={DocumentType.MARKDOWN},
                extensions={".md", ".markdown"},
                mime_types={"text/markdown", "text/x-markdown"},
                supports_code=True,
                supports_tables=True,
                priority=20,
            ),
            MarkdownParser(),
        ),
        (
            ParserSpec(
                parser_id="pdf_pymupdf",
                version=PdfParser.parser_version,
                doc_types={DocumentType.PDF},
                extensions={".pdf"},
                mime_types={"application/pdf"},
                # v1 只取文本层 + 带线表格，不做 OCR
                supports_scanned=False,
                supports_tables=True,
                priority=50,
            ),
            PdfParser(
                ocr_config=config.ocr if config else None,
                ocr_provider=pdf_ocr_provider,
                image_pdf_config=config.image_pdf if config else None,
            ),
        ),
        (
            ParserSpec(
                parser_id="docx_python_docx",
                version=DocxParser.parser_version,
                doc_types={DocumentType.DOCX},
                extensions={".docx"},
                mime_types={
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                },
                supports_tables=True,
                priority=50,
            ),
            DocxParser(),
        ),
        (
            ParserSpec(
                parser_id="xlsx_openpyxl",
                version=SpreadsheetParser.parser_version,
                doc_types={DocumentType.SPREADSHEET},
                extensions={".xlsx", ".xlsm"},
                mime_types={
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                },
                supports_tables=True,
                priority=50,
            ),
            SpreadsheetParser(),
        ),
    ]
    for spec, parser in entries:
        registry.register(spec, parser)
    return registry
