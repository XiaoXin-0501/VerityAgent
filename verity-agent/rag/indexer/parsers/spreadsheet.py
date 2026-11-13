"""XLSX parser（openpyxl，v1 把每个工作表转成文本表格）。

改进点：
- 每个工作表生成一个 TABLE 块，工作表名作为 ``caption``，进入 chunk 正文与
  embedding，便于“按 sheet 检索”；
- 去掉尾部全空列/全空行，避免 Excel 的格式空白污染表头；
- 第一非空行作为表头；空表跳过；
- 旧版 ``.xls`` 不处理，只读单元格值，不读取图片/图表/批注。
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
)

from ._common import make_block, make_parsed_document, make_table
from utils.text import (
    filename_to_title,
    is_meaningful_filename,
    truncate,
)

__all__ = ["SpreadsheetParser"]

# Excel 描述在 heading 中的最大长度，避免路径过长。
_DESCRIPTION_MAX_CHARS = 120


def _workbook_description(workbook) -> str | None:
    """读取工作簿描述：优先 description，其次 subject/title。"""
    props = getattr(workbook, "properties", None)
    if props is None:
        return None
    for name in ("description", "subject", "title"):
        value = getattr(props, name, None)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return truncate(text, _DESCRIPTION_MAX_CHARS)
    return None


def _cell_value(value: object) -> str:
    """把 openpyxl 单元格值转成可读字符串。"""
    if value is None:
        return ""
    return str(value).strip()


def _read_sheet_rows(sheet) -> list[list[str]]:
    """读取工作表所有行，去掉尾部空列与空行。"""
    rows = [
        [_cell_value(cell) for cell in row]
        for row in sheet.iter_rows(values_only=True)
    ]
    rows = [row for row in rows if any(cell for cell in row)]
    if not rows:
        return []
    # 只裁剪尾部空白列；中间空列可能承载对齐语义，保留。
    width = max(len(row) for row in rows)
    while width > 0 and all(
        len(row) < width or not row[width - 1] for row in rows
    ):
        width -= 1
    return [row[:width] for row in rows]


def _load_workbook(data: bytes):
    """优先只读模式打开；缺少 dimension 的文件回退到普通模式。

    部分 Excel 工具写出的工作表没有正确的 ``<dimension>``，openpyxl 的
    read_only 模式会错误地只看到 A1 一个单元格。这里检测到所有工作表都只有
    1x1 时重新以普通模式加载，保证表格内容不被静默截断。
    """
    from openpyxl import load_workbook

    workbook = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    only_cells = all(
        (sheet.max_row or 0) <= 1 and (sheet.max_column or 0) <= 1
        for sheet in workbook.worksheets
    )
    if only_cells:
        workbook.close()
        workbook = load_workbook(io.BytesIO(data), read_only=False, data_only=True)
    return workbook


class SpreadsheetParser:
    """v1 XLSX 解析器。"""

    parser_id = "xlsx_openpyxl"
    parser_version = "2"

    async def parse(
        self,
        source: DocumentSource,
        profile: DocumentProfile,
        *,
        doc_id: UUID,
        kb_id: UUID,
        data: bytes,
    ) -> ParsedDocument:
        """解析 XLSX 字节为 ``ParsedDocument``。"""
        try:
            import openpyxl  # noqa: F401  # 仅用于提前检测依赖是否安装
        except ImportError as exc:
            raise RagError(
                RagErrorCode.INVALID_CONFIG,
                "未安装 openpyxl，无法解析 XLSX",
                details={"parser_id": self.parser_id},
            ) from exc

        try:
            workbook = _load_workbook(data)
        except Exception as exc:
            raise RagError(
                RagErrorCode.INVALID_REQUEST,
                "XLSX 无法打开",
                details={"filename": source.filename, "cause": str(exc)},
            ) from exc

        blocks = []
        order = 0
        try:
            description = _workbook_description(workbook)
            if not description:
                metadata_description = str(
                    source.metadata.get("description") or ""
                ).strip()
                if metadata_description:
                    description = truncate(
                        metadata_description,
                        _DESCRIPTION_MAX_CHARS,
                    )
            # 文件名只有“可读”时才进入 heading；UUID/哈希/乱码会污染检索。
            filename_meaningful = is_meaningful_filename(source.filename)
            filename_label = (
                filename_to_title(source.filename)
                if filename_meaningful
                else None
            )
            for sheet_index, sheet in enumerate(workbook.worksheets, start=1):
                rows = _read_sheet_rows(sheet)
                if not rows:
                    continue
                heading_path = (
                    [filename_label, sheet.title]
                    if filename_label
                    else [sheet.title]
                )
                if description:
                    heading_path.append(description)
                caption_parts = (
                    [filename_label, sheet.title]
                    if filename_label
                    else [sheet.title]
                )
                caption = " / ".join(item for item in caption_parts if item)
                if description:
                    caption = f"{caption} - {description}"
                table = make_table(
                    rows[0],
                    rows[1:],
                    caption=caption,
                    summary=description,
                )
                blocks.append(
                    make_block(
                        doc_id=doc_id,
                        page_no=sheet_index,
                        order=order,
                        block_type=BlockType.TABLE,
                        text=table.markdown or "",
                        parser_id=self.parser_id,
                        table=table,
                        heading_level=1,
                        heading_text=" > ".join(heading_path),
                        heading_path=heading_path,
                        metadata={
                            "sheet": sheet.title,
                            "filename": source.filename,
                            "filename_meaningful": filename_meaningful,
                            "display_filename": filename_label,
                            "workbook_description": description,
                        },
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
