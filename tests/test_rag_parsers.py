"""RAG parser / chunker / 文本流水线测试。"""

from __future__ import annotations

import asyncio
import io
import sys

import pytest
from pathlib import Path
from uuid import uuid4

_ROOT = Path(__file__).resolve().parents[1]
_AGENT_ROOT = _ROOT / "verity-agent"
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

from config.rag import IndexerConfig  # noqa: E402
from rag.indexer import TextIngestionPipeline  # noqa: E402
from rag.indexer.chunkers import build_chunker_registry  # noqa: E402
from rag.indexer.chunkers.text import TextChunker  # noqa: E402
from rag.indexer.parsers import build_parser_registry, detect_profile  # noqa: E402
from rag.indexer.parsers.docx import DocxParser  # noqa: E402
from rag.indexer.parsers.markdown import MarkdownParser  # noqa: E402
from rag.indexer.parsers.pdf import PdfParser  # noqa: E402
from rag.indexer.parsers.spreadsheet import SpreadsheetParser  # noqa: E402
from rag.indexer.parsers.text import TextParser  # noqa: E402
from rag.shared.enums import (  # noqa: E402
    BlockType,
    DocumentType,
    SectionType,
)
from rag.shared.models import (  # noqa: E402
    ContentBlock,
    DocumentSource,
    ProcessedBlock,
    TableBlock,
)
from utils.tokens import estimate_tokens  # noqa: E402


def _source(filename: str, extension: str) -> DocumentSource:
    return DocumentSource(
        object_key=f"source/kb/{filename}",
        filename=filename,
        extension=extension,
        size_bytes=10,
        sha256="a" * 64,
    )


def test_text_parser_and_chunker_deterministic() -> None:
    async def run() -> None:
        doc_id, kb_id = uuid4(), uuid4()
        source = _source("a.txt", ".txt")
        parsed = await TextParser().parse(
            source,
            detect_profile(source),
            doc_id=doc_id,
            kb_id=kb_id,
            data=("第一段。" * 60 + "\n\n" + "第二段。" * 60).encode(),
        )
        assert parsed.blocks
        assert parsed.profile.doc_type is DocumentType.TEXT
        registry = build_chunker_registry(IndexerConfig())
        chunker = registry.resolve(DocumentType.TEXT, BlockType.PARAGRAPH)[1]
        processed = ProcessedBlock(block=parsed.blocks[0], text=parsed.blocks[0].text)
        first = await chunker.chunk(
            parsed.blocks[0],
            processed,
            kb_id=kb_id,
            doc_type=DocumentType.TEXT,
            chunking_version="v1",
        )
        second = await chunker.chunk(
            parsed.blocks[0],
            processed,
            kb_id=kb_id,
            doc_type=DocumentType.TEXT,
            chunking_version="v1",
        )
        assert first
        assert [item.chunk_id for item in first] == [item.chunk_id for item in second]
        assert first[0].section_type is SectionType.PARAGRAPH

    asyncio.run(run())


def test_markdown_parser_blocks_and_heading_not_chunked() -> None:
    async def run() -> None:
        doc_id, kb_id = uuid4(), uuid4()
        source = _source("a.md", ".md")
        parsed = await MarkdownParser().parse(
            source,
            detect_profile(source),
            doc_id=doc_id,
            kb_id=kb_id,
            data=(
                b"# Title\n\nHello.\n\n```python\nprint(1)\n```\n\n"
                b"| a | b |\n|---|---|\n| 1 | 2 |\n"
            ),
        )
        types = [block.block_type for block in parsed.blocks]
        assert BlockType.HEADING in types
        assert BlockType.PARAGRAPH in types
        assert BlockType.CODE in types
        assert BlockType.TABLE in types
        registry = build_chunker_registry(IndexerConfig())
        assert registry.resolve(DocumentType.MARKDOWN, BlockType.HEADING) is None

    asyncio.run(run())


def test_docx_parser_preserves_body_order_and_ignores_images() -> None:
    async def run() -> None:
        pytest.importorskip("docx")
        from docx import Document

        document = Document()
        document.add_heading("Title", level=1)
        document.add_paragraph("Hello DOCX")
        table = document.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "a"
        table.cell(0, 1).text = "b"
        table.cell(1, 0).text = "1"
        table.cell(1, 1).text = "2"
        buffer = io.BytesIO()
        document.save(buffer)

        source = _source("a.docx", ".docx")
        parsed = await DocxParser().parse(
            source,
            detect_profile(source),
            doc_id=uuid4(),
            kb_id=uuid4(),
            data=buffer.getvalue(),
        )
        types = [block.block_type for block in parsed.blocks]
        assert types[:3] == [
            BlockType.HEADING,
            BlockType.PARAGRAPH,
            BlockType.TABLE,
        ]
        assert parsed.profile.has_images is False

    asyncio.run(run())


def test_excel_parser_creates_table_block() -> None:
    async def run() -> None:
        pytest.importorskip("openpyxl")
        from openpyxl import Workbook

        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Sheet1"
        sheet.append(["id", "value"])
        sheet.append([1, "one"])
        buffer = io.BytesIO()
        workbook.save(buffer)
        workbook.close()

        source = _source("a.xlsx", ".xlsx")
        parsed = await SpreadsheetParser().parse(
            source,
            detect_profile(source),
            doc_id=uuid4(),
            kb_id=uuid4(),
            data=buffer.getvalue(),
        )
        assert len(parsed.blocks) == 1
        assert parsed.blocks[0].block_type is BlockType.TABLE
        assert parsed.blocks[0].table is not None
        assert parsed.blocks[0].table.headers == ["id", "value"]

    asyncio.run(run())


def test_pdf_parser_extracts_text_only() -> None:
    async def run() -> None:
        pytest.importorskip("pymupdf")
        import pymupdf as fitz  # type: ignore[import-not-found]

        document = fitz.open()
        page = document.new_page()
        page.insert_text((72, 72), "Hello PDF")
        data = document.tobytes()
        document.close()

        source = _source("a.pdf", ".pdf")
        parsed = await PdfParser().parse(
            source,
            detect_profile(source),
            doc_id=uuid4(),
            kb_id=uuid4(),
            data=data,
        )
        assert parsed.profile.has_text_layer is True
        assert parsed.profile.has_images is False
        assert any("Hello PDF" in block.text for block in parsed.blocks)

    asyncio.run(run())


def test_table_chunker_repeats_header_for_large_table() -> None:
    async def run() -> None:
        doc_id, kb_id = uuid4(), uuid4()
        rows = [[str(i), f"value-{i}"] for i in range(120)]
        block = ContentBlock(
            block_id=uuid4(),
            doc_id=doc_id,
            order=0,
            block_type=BlockType.TABLE,
            text="\n".join(
                " | ".join(row) for row in [["id", "value"]] + rows
            ),
            table=TableBlock(headers=["id", "value"], rows=rows),
        )
        registry = build_chunker_registry(
            IndexerConfig(max_chunk_tokens=120, chunk_size_tokens=80)
        )
        chunker = registry.resolve(DocumentType.SPREADSHEET, BlockType.TABLE)[1]
        chunks = await chunker.chunk(
            block,
            ProcessedBlock(block=block, text=block.text),
            kb_id=kb_id,
            doc_type=DocumentType.SPREADSHEET,
            chunking_version="v1",
        )
        assert len(chunks) > 1
        assert "id | value" in chunks[0].text
        assert "id | value" in chunks[1].text

    asyncio.run(run())


def test_parser_registry_selects_extension() -> None:
    registry = build_parser_registry()
    source = _source("a.md", ".md")
    spec, parser = registry.resolve(source, detect_profile(source))
    assert spec.parser_id == "markdown"
    assert parser is not None


def test_text_chunker_splits_long_unit_within_max_tokens() -> None:
    async def run() -> None:
        doc_id, kb_id = uuid4(), uuid4()
        block = ContentBlock(
            block_id=uuid4(),
            doc_id=doc_id,
            order=0,
            block_type=BlockType.PARAGRAPH,
            text="a " * 5000,
        )
        processed = ProcessedBlock(block=block, text=block.text)
        chunker = TextChunker(
            target_tokens=100,
            max_tokens=100,
            min_tokens=0,
            overlap_tokens=0,
        )
        chunks = await chunker.chunk(
            block,
            processed,
            kb_id=kb_id,
            doc_type=DocumentType.TEXT,
            chunking_version="v1",
        )
        assert len(chunks) > 1
        assert all(estimate_tokens(item.embedding_text) <= 100 for item in chunks)

    asyncio.run(run())


def test_text_ingestion_pipeline_markdown_ignores_image_syntax() -> None:
    async def run() -> None:
        pipeline = TextIngestionPipeline(
            IndexerConfig(
                chunk_size_tokens=80,
                max_chunk_tokens=100,
                min_chunk_tokens=0,
            )
        )
        source = _source("a.md", ".md")
        data = (
            b"# Title\n\n![logo](https://example.com/logo.png)\n\n"
            + ("Hello text. " * 20).encode()
        )
        result = await pipeline.process(
            source,
            data,
            doc_id=uuid4(),
            kb_id=uuid4(),
        )
        assert result.chunks
        assert result.parsed.profile.has_images is True
        assert all("example.com" not in chunk.text for chunk in result.chunks)
        assert all("![" not in chunk.text for chunk in result.chunks)

    asyncio.run(run())
