"""LayoutDocument → ParsedDocument 映射。"""

from __future__ import annotations

from difflib import SequenceMatcher
from uuid import UUID

from config.rag import ImagePdfConfig
from rag.shared.enums import BlockType, CodeParseStatus, DocumentType, SymbolType
from rag.shared.models import (
    CodeBlock,
    ContentBlock,
    DocumentProfile,
    DocumentSource,
    Page,
    ParsedDocument,
)

from .._common import HeadingStack, make_block, make_page, make_parsed_document
from .models import LayoutBlock, LayoutDocument
from .utils import normalize_for_dedup

__all__ = ["layout_document_to_parsed", "merge_tile_blocks"]


def merge_tile_blocks(layout: LayoutDocument) -> list[LayoutBlock]:
    """合并 tile 重叠区域的重复块，并按页/阅读顺序排序。"""
    grouped: dict[int, list[LayoutBlock]] = {}
    for block in layout.blocks:
        grouped.setdefault(block.page_no, []).append(block)

    merged: list[LayoutBlock] = []
    for page_no, blocks in grouped.items():
        kept: list[LayoutBlock] = []
        for block in blocks:
            duplicate_index = _find_duplicate(block, kept)
            if duplicate_index is None:
                kept.append(block)
                continue
            # 重叠区域两份 OCR 可能略有差异；保留信息更完整的一份。
            if _is_better(block, kept[duplicate_index]):
                kept[duplicate_index] = block
        kept.sort(
            key=lambda item: (
                item.metadata.get("tile_index", 0),
                item.reading_order,
            )
        )
        merged.extend(kept)
    return merged


def _find_duplicate(
    block: LayoutBlock,
    kept: list[LayoutBlock],
) -> int | None:
    """查找与当前块重复的已保留块索引。"""
    for index, existing in enumerate(kept):
        if _is_duplicate(existing, block):
            return index
    return None


def _is_duplicate(left: LayoutBlock, right: LayoutBlock) -> bool:
    """判断两个块是否来自 tile 重叠区域的同一内容。"""
    if left.block_type is not right.block_type:
        return False
    overlap = _overlap_ratio(left, right)
    if overlap is None:
        # 没有 bbox 时，只按完全相同文本去重。
        return normalize_for_dedup(left.text) == normalize_for_dedup(right.text)
    if overlap <= 0:
        return False

    # 表格结构在重叠区可能 OCR 出细微差异；bbox 大面积重叠即可视为同一表。
    if left.block_type is BlockType.TABLE:
        return overlap >= 0.35

    left_text = normalize_for_dedup(left.text)
    right_text = normalize_for_dedup(right.text)
    if not left_text or not right_text:
        return False
    if left_text == right_text:
        return overlap >= 0.2
    if overlap < 0.25:
        return False
    return _similarity(left_text, right_text) >= 0.85


def _similarity(left: str, right: str) -> float:
    """限制长度后做文本相似度，避免超长表格拖慢。"""
    return SequenceMatcher(None, left[:800], right[:800]).ratio()


def _overlap_ratio(left: LayoutBlock, right: LayoutBlock) -> float | None:
    """返回较小块被重叠覆盖的面积比例；无 bbox 返回 None。"""
    if left.bbox is None or right.bbox is None:
        return None
    lx0, ly0, lx1, ly1 = left.bbox
    rx0, ry0, rx1, ry1 = right.bbox
    x_overlap = min(lx1, rx1) - max(lx0, rx0)
    y_overlap = min(ly1, ry1) - max(ly0, ry0)
    if x_overlap <= 0 or y_overlap <= 0:
        return 0.0
    left_area = max(1.0, (lx1 - lx0) * (ly1 - ly0))
    right_area = max(1.0, (rx1 - rx0) * (ry1 - ry0))
    return (x_overlap * y_overlap) / min(left_area, right_area)


def _is_better(candidate: LayoutBlock, existing: LayoutBlock) -> bool:
    """判断候选块是否比已保留块更完整。"""
    if candidate.block_type is BlockType.TABLE and existing.block_type is BlockType.TABLE:
        candidate_rows = len(candidate.table.rows) if candidate.table else 0
        existing_rows = len(existing.table.rows) if existing.table else 0
        if candidate_rows != existing_rows:
            return candidate_rows > existing_rows
        return len(candidate.text) > len(existing.text)
    if candidate.confidence is not None and existing.confidence is not None:
        if abs(candidate.confidence - existing.confidence) > 0.02:
            return candidate.confidence > existing.confidence
    return len(candidate.text) > len(existing.text)


def layout_document_to_parsed(
    layout: LayoutDocument,
    *,
    source: DocumentSource,
    doc_id: UUID,
    kb_id: UUID,
    profile: DocumentProfile,
    config: ImagePdfConfig,
    parser_id: str = "image_pdf",
    parser_version: str = "1",
    page_count: int | None = None,
) -> ParsedDocument:
    """把版式结果映射为 ``ParsedDocument``，并维护完整标题路径。"""
    blocks = merge_tile_blocks(layout)
    if not blocks:
        return make_parsed_document(
            source=source,
            doc_id=doc_id,
            kb_id=kb_id,
            parser_id=parser_id,
            parser_version=parser_version,
            profile=profile.model_copy(
                update={
                    "doc_type": DocumentType.PDF,
                    "page_count": page_count,
                    "has_text_layer": False,
                    "is_scanned": True,
                    "text_coverage": 0.0,
                    "ocr_ratio": 0.0,
                }
            ),
            blocks=[],
            warnings=[*layout.warnings, "图片 PDF 未识别到可索引文本"],
        )

    headings = HeadingStack()
    content_blocks: list[ContentBlock] = []
    order = 0
    has_tables = False
    has_formulas = False
    has_images = False

    for block in blocks:
        if block.block_type is BlockType.HEADING:
            level = max(1, min(6, block.level or 1))
            path_text, path = headings.apply(level, block.text)
            content_blocks.append(
                make_block(
                    doc_id=doc_id,
                    page_no=block.page_no,
                    order=order,
                    block_type=BlockType.HEADING,
                    text=block.text,
                    parser_id="image_pdf",
                    heading_level=level,
                    heading_text=path_text,
                    heading_path=path,
                    bbox=list(block.bbox) if block.bbox else None,
                    confidence=block.confidence,
                    metadata=_metadata(block),
                )
            )
            order += 1
            continue

        current_heading, current_path = headings.current()
        block_type = block.block_type
        kwargs: dict = {
            "bbox": list(block.bbox) if block.bbox else None,
            "confidence": block.confidence,
            "metadata": _metadata(block),
            "heading_text": current_heading,
            "heading_path": current_path,
        }
        if block_type is BlockType.TABLE:
            if block.table is None:
                block_type = BlockType.PARAGRAPH
            else:
                has_tables = True
                kwargs["table"] = block.table
        elif block_type is BlockType.CODE:
            kwargs["code"] = CodeBlock(
                body=block.text,
                symbol_type=SymbolType.BLOCK,
                parse_status=CodeParseStatus.PARTIAL,
            )
        elif block_type is BlockType.FIGURE:
            has_images = True
            kwargs["figure"] = block.figure
        elif block_type is BlockType.FORMULA:
            has_formulas = True
            kwargs["formula"] = block.formula

        content_blocks.append(
            make_block(
                doc_id=doc_id,
                page_no=block.page_no,
                order=order,
                block_type=block_type,
                text=block.text,
                parser_id="image_pdf",
                **kwargs,
            )
        )
        order += 1

    page_numbers = sorted(
        set(layout.page_sizes) | {item.page_no for item in content_blocks}
    )
    resolved_page_count = page_count or max(page_numbers, default=0)
    pages: list[Page] = []
    pages_with_text = 0
    for page_no in page_numbers:
        page_blocks = [item for item in content_blocks if item.page_no == page_no]
        if any(item.block_type is not BlockType.FIGURE for item in page_blocks):
            pages_with_text += 1
        width, height = layout.page_sizes.get(page_no, (None, None))
        pages.append(
            make_page(
                page_no=page_no,
                blocks=page_blocks,
                width=width or None,
                height=height or None,
            )
        )
    coverage = (
        pages_with_text / resolved_page_count if resolved_page_count else 0.0
    )
    ocr_ratio = (
        len(layout.ocr_page_numbers) / resolved_page_count
        if resolved_page_count
        else 1.0
    )
    warnings = list(layout.warnings)
    low_confidence = sum(
        1
        for item in content_blocks
        if item.confidence is not None
        and item.confidence < config.min_confidence
    )
    if low_confidence:
        warnings.append(f"{low_confidence} 个块 OCR 置信度较低")

    return make_parsed_document(
        source=source,
        doc_id=doc_id,
        kb_id=kb_id,
        parser_id=parser_id,
        parser_version=parser_version,
        profile=profile.model_copy(
            update={
                "doc_type": DocumentType.PDF,
                "page_count": resolved_page_count,
                "has_text_layer": False,
                "is_scanned": True,
                "has_tables": has_tables,
                "has_images": has_images,
                "has_formulas": has_formulas,
                "text_coverage": coverage,
                "ocr_ratio": ocr_ratio,
                "quality_score": coverage,
            }
        ),
        blocks=content_blocks,
        pages=pages,
        warnings=warnings,
    )


def _metadata(block: LayoutBlock) -> dict:
    """合并引擎元数据并标记 OCR 来源。"""
    metadata = dict(block.metadata)
    metadata.setdefault("engine", "image_pdf")
    metadata.setdefault("ocr", True)
    return metadata
