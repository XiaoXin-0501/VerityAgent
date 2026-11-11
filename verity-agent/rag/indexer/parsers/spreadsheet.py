"""XLSX parser（openpyxl，v1 把每个工作表转成文本表格）。

- 不读取图片/图表/批注，只读取单元格值；
- 每个非空工作表一个 `TableBlock`，`text` 为 ``a | b`` 行文本，供 TableChunker
  生成纯文本 chunk；
- 第一行作为表头；空行跳过；旧版 `.xls` 不处理。
"""

from __future__ import annotations

import io
from uuid import UUID

from exception import RagError, RagErrorCode
from rag.shared.enums import BlockType, DocumentType
from rag.shared.models import (
    DocumentProfile,
    DocumentSource,
    ParsedDocument,
    TableBlock,
)

from ._common import make_block, make_parsed_document

__all__ = ["SpreadsheetParser"]


class SpreadsheetParser:
    parser_id = "xlsx_openpyxl"
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
            from openpyxl import load_workbook  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RagError(
                RagErrorCode.INVALID_CONFIG,
                "未安装 openpyxl，无法解析 XLSX",
                details={"parser_id": self.parser_id},
            ) from exc

        try:
            workbook = load_workbook(
                io.BytesIO(data),
                read_only=True,
                data_only=True,
            )
        except Exception as exc:
            raise RagError(
                RagErrorCode.INVALID_REQUEST,
                "XLSX 无法打开",
                details={"filename": source.filename, "cause": str(exc)},
            ) from exc

        blocks = []
        order = 0
        try:
            for sheet_index, sheet in enumerate(workbook.worksheets, start=1):
                rows = [
                    ["" if cell is None else str(cell) for cell in row]
                    for row in sheet.iter_rows(values_only=True)
                ]
                rows = [row for row in rows if any(cell.strip() for cell in row)]
                if not rows:
                    continue
                blocks.append(
                    make_block(
                        doc_id=doc_id,
                        page_no=sheet_index,
                        order=order,
                        block_type=BlockType.TABLE,
                        text="\n".join(" | ".join(row) for row in rows),
                        parser_id=self.parser_id,
                        table=TableBlock(headers=rows[0], rows=rows[1:]),
                        metadata={"sheet": sheet.title},
                    )
                )
                order += 1
            page_count = len(workbook.worksheets)
        finally:
            workbook.close()

        has_text = bool(blocks)
        return make_parsed_document(
            source=source,
            doc_id=doc_id,
            kb_id=kb_id,
            parser_id=self.parser_id,
            parser_version=self.parser_version,
            profile=profile.model_copy(
                update={
                    "doc_type": DocumentType.SPREADSHEET,
                    "has_text_layer": has_text,
                    "has_tables": has_text,
                    "has_images": False,
                    "page_count": page_count,
                    "text_coverage": 1.0 if has_text else 0.0,
                }
            ),
            blocks=blocks,
            warnings=[] if has_text else ["Excel 没有可索引单元格"],
        )
