"""RapidOCR 轻量版式引擎。

RapidOCR 只提供文本行，不提供版面/表格结构；本引擎用 bbox 做近似段落
分组与标题推断，作为 MinerU 不可用时的降级方案。表格会退化为文本行，
并写 warning。
"""

from __future__ import annotations

import importlib.util
import statistics
from pathlib import Path

from config.rag import ImagePdfConfig
from rag.shared.enums import BlockType

from ..models import LayoutBlock, LayoutDocument, RenderedTile
from ..utils import guess_heading_level, looks_like_list

__all__ = ["RapidLayoutEngine"]


class RapidLayoutEngine:
    """基于 RapidOCR 的轻量版式重建。"""

    name = "rapidocr"

    def __init__(self, config: ImagePdfConfig) -> None:
        """保存配置；OCR 引擎懒加载。"""
        self.config = config
        self._engine = None

    def available(self) -> bool:
        """RapidOCR 是否可导入。"""
        try:
            return importlib.util.find_spec("rapidocr") is not None
        except (ImportError, ValueError):
            return False

    def _get_engine(self):
        """懒加载 RapidOCR。"""
        if self._engine is None:
            from rapidocr import RapidOCR  # type: ignore[import-not-found]

            self._engine = RapidOCR()
        return self._engine

    def parse(
        self,
        tiles: list[RenderedTile],
        *,
        config: ImagePdfConfig,
        work_dir: str | Path,
    ) -> LayoutDocument:
        """逐 tile OCR，并按行框聚类成段落/标题。"""
        engine = self._get_engine()
        blocks: list[LayoutBlock] = []
        page_sizes: dict[int, tuple[float, float]] = {}
        warnings: list[str] = ["RapidOCR 降级模式：表格结构可能丢失"]
        ocr_pages: set[int] = set()

        for tile in tiles:
            page_sizes[tile.page_no] = (
                tile.page_width_px * tile.scale_to_points,
                tile.page_height_px * tile.scale_to_points,
            )
            result = engine(tile.image_path)
            lines = _extract_lines(result, tile)
            if not lines:
                continue
            ocr_pages.add(tile.page_no)
            blocks.extend(_group_lines(lines, tile))
        return LayoutDocument(
            blocks=blocks,
            page_sizes=page_sizes,
            engine=self.name,
            engine_version=None,
            warnings=warnings,
            ocr_page_numbers=ocr_pages,
        )


def _extract_lines(result, tile: RenderedTile) -> list[dict]:
    """把 RapidOCR 输出转成内部行结构。"""
    boxes = getattr(result, "boxes", None)
    texts = getattr(result, "txts", None)
    scores = getattr(result, "scores", None)
    if boxes is None or texts is None:
        return []
    lines: list[dict] = []
    for index, text in enumerate(texts):
        value = str(text or "").strip()
        if not value:
            continue
        try:
            box = boxes[index]
            xs = [float(point[0]) for point in box]
            ys = [float(point[1]) for point in box]
        except Exception:
            continue
        confidence = None
        try:
            confidence = float(scores[index]) if scores is not None else None
        except (IndexError, TypeError, ValueError):
            confidence = None
        lines.append(
            {
                "text": value,
                "x0": min(xs),
                "y0": min(ys) + tile.offset_y_px,
                "x1": max(xs),
                "y1": max(ys) + tile.offset_y_px,
                "confidence": confidence,
                "tile": tile,
            }
        )
    lines.sort(key=lambda item: (round(item["y0"] / 6), item["x0"]))
    return lines


def _to_points(value: float, tile: RenderedTile) -> float:
    """像素 → PDF 点。"""
    return value * tile.scale_to_points


def _group_lines(lines: list[dict], tile: RenderedTile) -> list[LayoutBlock]:
    """按行距和 x 重叠把行聚成段落，短/大字号行尝试识别为标题。"""
    heights = [max(1.0, line["y1"] - line["y0"]) for line in lines]
    median_height = statistics.median(heights) if heights else 12.0
    blocks: list[LayoutBlock] = []
    group: list[dict] = []

    def flush() -> None:
        """输出当前段落。"""
        if not group:
            return
        value = "\n".join(line["text"] for line in group).strip()
        if not value:
            group.clear()
            return
        is_list = len(group) > 1 and all(
            looks_like_list(line["text"]) for line in group
        )
        blocks.append(
            LayoutBlock(
                block_type=BlockType.LIST if is_list else BlockType.PARAGRAPH,
                text=value,
                page_no=tile.page_no,
                bbox=(
                    _to_points(min(line["x0"] for line in group), tile),
                    _to_points(min(line["y0"] for line in group), tile),
                    _to_points(max(line["x1"] for line in group), tile),
                    _to_points(max(line["y1"] for line in group), tile),
                ),
                reading_order=len(blocks),
                confidence=_min_confidence(group),
                metadata={"engine": "rapidocr", "tile_index": tile.tile_index},
            )
        )
        group.clear()

    for line in lines:
        text = line["text"]
        line_height = max(1.0, line["y1"] - line["y0"])
        size_level = None
        if len(text) <= 40 and not text.endswith(("。", "！", ".", "!")):
            if line_height >= median_height * 1.45:
                size_level = 1
            elif line_height >= median_height * 1.2:
                size_level = 2
        level = guess_heading_level(None, text) or size_level
        if level is not None:
            flush()
            blocks.append(
                LayoutBlock(
                    block_type=BlockType.HEADING,
                    text=text,
                    page_no=tile.page_no,
                    bbox=(
                        _to_points(line["x0"], tile),
                        _to_points(line["y0"], tile),
                        _to_points(line["x1"], tile),
                        _to_points(line["y1"], tile),
                    ),
                    reading_order=len(blocks),
                    confidence=line.get("confidence"),
                    level=level,
                    metadata={"engine": "rapidocr", "tile_index": tile.tile_index},
                )
            )
            continue

        if group:
            previous = group[-1]
            gap = line["y0"] - previous["y1"]
            x_overlap = min(line["x1"], previous["x1"]) - max(
                line["x0"], previous["x0"]
            )
            if gap > median_height * 1.8 or x_overlap <= 0:
                flush()
        group.append(line)
    flush()
    return blocks


def _min_confidence(group: list[dict]) -> float | None:
    """取组内最低置信度。"""
    values = [
        line["confidence"]
        for line in group
        if line.get("confidence") is not None
    ]
    return min(values) if values else None
