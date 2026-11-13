"""图片 PDF 渲染与超长页切片。"""

from __future__ import annotations

import math
from pathlib import Path

from config.rag import ImagePdfConfig
from exception import RagError, RagErrorCode

from .models import RenderedTile

__all__ = ["render_pdf_tiles"]


def render_pdf_tiles(
    pdf_bytes: bytes,
    *,
    page_numbers: list[int],
    config: ImagePdfConfig,
    work_dir: str | Path,
) -> tuple[list[RenderedTile], dict[int, tuple[float, float]]]:
    """把指定页渲染成 PNG；超长页按纵向 tile 切分。

    返回 tile 列表和 ``page_no -> (width_pt, height_pt)``。
    """
    try:
        import pymupdf as fitz  # type: ignore[import-not-found]
    except ImportError:
        try:
            import fitz  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RagError(
                RagErrorCode.INVALID_CONFIG,
                "图片 PDF 解析需要 PyMuPDF",
                details={"engine": "image_pdf"},
            ) from exc

    output_dir = Path(work_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise RagError(
            RagErrorCode.INVALID_REQUEST,
            "图片 PDF 无法打开",
            details={"cause": str(exc)},
        ) from exc

    tiles: list[RenderedTile] = []
    page_sizes: dict[int, tuple[float, float]] = {}
    try:
        for page_no in page_numbers:
            if page_no < 1 or page_no > document.page_count:
                continue
            page = document.load_page(page_no - 1)
            width_pt = float(page.rect.width)
            height_pt = float(page.rect.height)
            page_sizes[page_no] = (width_pt, height_pt)

            page_width_px = max(1, round(width_pt * config.dpi / 72))
            page_height_px = max(1, round(height_pt * config.dpi / 72))
            is_long = (
                height_pt / width_pt >= config.long_page_ratio
                or page_height_px > config.tile_height * 2
            )
            if not is_long:
                tiles.append(
                    _render_clip(
                        page,
                        fitz,
                        page_no=page_no,
                        tile_index=0,
                        offset_y_px=0,
                        clip_top_pt=0.0,
                        clip_bottom_pt=height_pt,
                        page_width_px=page_width_px,
                        page_height_px=page_height_px,
                        dpi=config.dpi,
                        output_dir=output_dir,
                        is_full_page=True,
                    )
                )
                continue

            tile_height, overlap = _resolve_tile_geometry(
                page_height_px,
                config.tile_height,
                config.tile_overlap,
                config.max_tiles_per_page,
            )
            step = max(1, tile_height - overlap)
            tile_index = 0
            y = 0
            while y < page_height_px:
                y2 = min(page_height_px, y + tile_height)
                if y2 - y < 200 and tile_index > 0:
                    break
                tiles.append(
                    _render_clip(
                        page,
                        fitz,
                        page_no=page_no,
                        tile_index=tile_index,
                        offset_y_px=y,
                        clip_top_pt=y * 72.0 / config.dpi,
                        clip_bottom_pt=y2 * 72.0 / config.dpi,
                        page_width_px=page_width_px,
                        page_height_px=page_height_px,
                        dpi=config.dpi,
                        output_dir=output_dir,
                        is_full_page=False,
                    )
                )
                tile_index += 1
                if y2 >= page_height_px:
                    break
                y += step
    finally:
        document.close()

    return tiles, page_sizes


def _resolve_tile_geometry(
    page_height_px: int,
    tile_height: int,
    overlap: int,
    max_tiles: int,
) -> tuple[int, int]:
    """在 tile 数上限内计算实际 tile 高度与重叠。"""
    if max_tiles <= 0:
        return page_height_px, 0
    needed = max(1, math.ceil(page_height_px / max(1, tile_height)))
    if needed <= max_tiles:
        return tile_height, min(overlap, max(0, tile_height // 2))
    resolved_height = max(400, math.ceil(page_height_px / max_tiles))
    return resolved_height, min(overlap, max(0, resolved_height // 4))


def _render_clip(
    page,
    fitz,
    *,
    page_no: int,
    tile_index: int,
    offset_y_px: int,
    clip_top_pt: float,
    clip_bottom_pt: float,
    page_width_px: int,
    page_height_px: int,
    dpi: int,
    output_dir: Path,
    is_full_page: bool,
) -> RenderedTile:
    """渲染一个页面/切片并保存为 PNG。"""
    clip = fitz.Rect(0, clip_top_pt, page.rect.width, clip_bottom_pt)
    pixmap = page.get_pixmap(dpi=dpi, clip=clip, alpha=False)
    name = (
        f"p{page_no:04d}_full.png"
        if is_full_page
        else f"p{page_no:04d}_t{tile_index:02d}_{offset_y_px}.png"
    )
    path = output_dir / name
    pixmap.save(str(path))
    return RenderedTile(
        page_no=page_no,
        tile_index=tile_index,
        image_path=str(path),
        width_px=pixmap.width,
        height_px=pixmap.height,
        offset_y_px=offset_y_px,
        page_width_px=page_width_px,
        page_height_px=page_height_px,
        dpi=dpi,
        is_full_page=is_full_page,
    )
