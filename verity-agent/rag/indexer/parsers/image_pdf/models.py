"""图片 PDF 版式解析的内部数据模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rag.shared.enums import BlockType
from rag.shared.models import (
    CodeBlock,
    FigureBlock,
    FormulaBlock,
    TableBlock,
)

__all__ = ["RenderedTile", "LayoutBlock", "LayoutDocument"]


@dataclass(slots=True)
class RenderedTile:
    """渲染后的一张页面或页面纵向切片。"""

    page_no: int
    tile_index: int
    image_path: str
    width_px: int
    height_px: int
    # 该 tile 在整页渲染图中的纵向偏移
    offset_y_px: int
    # 整页渲染图尺寸与 DPI，用于把 bbox 换算回 PDF 点
    page_width_px: int
    page_height_px: int
    dpi: int
    is_full_page: bool = True

    @property
    def scale_to_points(self) -> float:
        """渲染像素 → PDF 点的换算系数。"""
        return 72.0 / self.dpi


@dataclass(slots=True)
class LayoutBlock:
    """版式引擎输出的结构化块。"""

    block_type: BlockType
    text: str = ""
    page_no: int = 1
    # PDF 点坐标系；来自 tile 时已经加回全局偏移
    bbox: tuple[float, float, float, float] | None = None
    reading_order: int = 0
    confidence: float | None = None
    level: int | None = None
    table: TableBlock | None = None
    code: CodeBlock | None = None
    formula: FormulaBlock | None = None
    figure: FigureBlock | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LayoutDocument:
    """一个文档（或若干页）的版式解析结果。"""

    blocks: list[LayoutBlock] = field(default_factory=list)
    # page_no -> (width_pt, height_pt)
    page_sizes: dict[int, tuple[float, float]] = field(default_factory=dict)
    engine: str = "unknown"
    engine_version: str | None = None
    warnings: list[str] = field(default_factory=list)
    # 原始内容统计，用于 profile
    ocr_page_numbers: set[int] = field(default_factory=set)
