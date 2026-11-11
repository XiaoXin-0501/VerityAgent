"""CSV parser：首行表头 + 数据行。"""

from __future__ import annotations

import csv
import io
from uuid import UUID

from rag.shared.enums import BlockType, DocumentType
from rag.shared.models import (
    DocumentProfile,
    DocumentSource,
    ParsedDocument,
    TableBlock,
)

from ._common import decode_text, make_block, make_parsed_document

__all__ = ["CsvParser"]


class CsvParser:
    parser_id = "csv"
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
        text = decode_text(data)
        sample = text[:4096]
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        rows = list(csv.reader(io.StringIO(text), dialect))
        rows = [row for row in rows if any(cell.strip() for cell in row)]
        headers = [cell.strip() for cell in rows[0]] if rows else []
        body = [[cell.strip() for cell in row] for row in rows[1:]]
        table = TableBlock(headers=headers, rows=body)
        block = make_block(
            doc_id=doc_id,
            page_no=0,
            order=0,
            block_type=BlockType.TABLE,
            text=text,
            parser_id=self.parser_id,
            table=table,
        )
        return make_parsed_document(
            source=source,
            doc_id=doc_id,
            kb_id=kb_id,
            parser_id=self.parser_id,
            parser_version=self.parser_version,
            profile=profile.model_copy(
                update={
                    "doc_type": DocumentType.CSV,
                    "has_text_layer": True,
                    "has_tables": True,
                    "page_count": 1,
                    "text_coverage": 1.0,
                }
            ),
            blocks=[block],
        )
