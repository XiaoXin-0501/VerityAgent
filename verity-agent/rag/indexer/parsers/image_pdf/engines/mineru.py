"""MinerU Pipeline 版式引擎（非 LLM）。

通过 MinerU CLI 的 ``pipeline`` backend 解析渲染后的页面/切片，读取
``*_content_list.json`` 并转换为内部 ``LayoutBlock``。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from config.rag import ImagePdfConfig
from exception import RagError, RagErrorCode
from rag.shared.enums import BlockType
from rag.shared.models import FigureBlock, FormulaBlock

from ..html_table import html_table_to_table_block
from ..models import LayoutBlock, LayoutDocument, RenderedTile
from ..utils import (
    as_caption,
    guess_heading_level,
    html_to_text,
    looks_like_list,
)

__all__ = ["MineruLayoutEngine"]


class MineruLayoutEngine:
    """MinerU pipeline backend 适配器。"""

    name = "mineru"

    def __init__(self, config: ImagePdfConfig) -> None:
        """保存配置；不在此处启动模型。"""
        self.config = config

    def available(self) -> bool:
        """检查 MinerU CLI 是否存在。"""
        return shutil.which(self.config.mineru_command) is not None

    def parse(
        self,
        tiles: list[RenderedTile],
        *,
        config: ImagePdfConfig,
        work_dir: str | Path,
    ) -> LayoutDocument:
        """调用 MinerU CLI 并解析 content_list。"""
        if not tiles:
            return LayoutDocument(engine=self.name)
        if not self.available():
            raise RagError(
                RagErrorCode.INVALID_CONFIG,
                "未找到 MinerU CLI，请安装 mineru[core] 或配置 mineru_command",
                details={"engine": self.name},
            )

        output_root = Path(work_dir) / "mineru_out"
        output_root.mkdir(parents=True, exist_ok=True)
        command = [
            self.config.mineru_command,
            "-p",
            str(Path(work_dir) / "tiles"),
            "-o",
            str(output_root),
            "-b",
            config.mineru_backend,
            "-m",
            config.mineru_method,
            "-l",
            config.mineru_language,
            "-f",
            "True" if config.mineru_formula_enabled else "False",
            "-t",
            "True" if config.mineru_table_enabled else "False",
        ]
        if config.mineru_api_url:
            command.extend(["--api-url", config.mineru_api_url])
        env = dict(os.environ)
        env.setdefault("MINERU_MODEL_SOURCE", config.mineru_model_source)
        print(
            f"[mineru] 解析 {len(tiles)} 个 tile；首次运行加载模型可能需要 1~3 分钟",
            flush=True,
        )
        try:
            completed = subprocess.run(
                command,
                cwd=str(work_dir),
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=config.mineru_timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RagError(
                RagErrorCode.DEPENDENCY_UNAVAILABLE,
                "MinerU 解析超时",
                details={"engine": self.name, "timeout_s": config.mineru_timeout_s},
                retryable=True,
            ) from exc
        if completed.returncode != 0:
            stderr = (completed.stderr or completed.stdout or "")[-2000:]
            raise RagError(
                RagErrorCode.DEPENDENCY_UNAVAILABLE,
                "MinerU 解析失败",
                details={"engine": self.name, "stderr": stderr},
                retryable=True,
            )
        print("[mineru] 解析完成，读取 content_list...", flush=True)

        blocks: list[LayoutBlock] = []
        page_sizes: dict[int, tuple[float, float]] = {}
        warnings: list[str] = []
        ocr_pages: set[int] = set()

        for tile in tiles:
            stem = Path(tile.image_path).stem
            matches = list(output_root.glob(f"**/{stem}/ocr/*_content_list.json"))
            if not matches:
                matches = list(output_root.glob(f"**/{stem}/*_content_list.json"))
            if not matches:
                warnings.append(f"MinerU 未找到 {stem} 的输出")
                continue
            content_list = matches[0]
            try:
                items = json.loads(content_list.read_text(encoding="utf-8"))
            except Exception as exc:
                warnings.append(f"MinerU 输出解析失败 {stem}: {exc}")
                continue
            page_sizes[tile.page_no] = _page_size_points(tile)
            ocr_pages.add(tile.page_no)
            for order, item in enumerate(items):
                mapped = _map_item(
                    item,
                    tile=tile,
                    tile_order=order,
                )
                if mapped is None:
                    continue
                blocks.append(mapped)

        return LayoutDocument(
            blocks=blocks,
            page_sizes=page_sizes,
            engine=self.name,
            engine_version=_mineru_version(self.config),
            warnings=warnings,
            ocr_page_numbers=ocr_pages,
        )


def _page_size_points(tile: RenderedTile) -> tuple[float, float]:
    """把整页像素尺寸换算为 PDF 点。"""
    scale = tile.scale_to_points
    return (
        tile.page_width_px * scale,
        tile.page_height_px * scale,
    )


def _mineru_version(config: ImagePdfConfig) -> str | None:
    """读取 CLI 版本；失败返回 None。"""
    try:
        completed = subprocess.run(
            [config.mineru_command, "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            check=False,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    match = re.search(r"(\d+\.\d+\.\d+)", completed.stdout or "")
    return match.group(1) if match else None


def _item_bbox(
    item: dict,
    tile: RenderedTile,
) -> tuple[float, float, float, float] | None:
    """把 MinerU 的 0~1000 归一化 bbox 换算为 PDF 点坐标。"""
    bbox = item.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    scale_x = tile.width_px / 1000.0 * tile.scale_to_points
    scale_y = tile.height_px / 1000.0 * tile.scale_to_points
    return (
        x0 * scale_x,
        y0 * scale_y + tile.offset_y_px * tile.scale_to_points,
        x1 * scale_x,
        y1 * scale_y + tile.offset_y_px * tile.scale_to_points,
    )


def _map_item(
    item: dict,
    *,
    tile: RenderedTile,
    tile_order: int,
) -> LayoutBlock | None:
    """把 MinerU content_list 的一项映射为 LayoutBlock。"""
    raw_type = str(item.get("type") or "").strip().lower()
    text = str(item.get("text") or "").strip()
    bbox = _item_bbox(item, tile)
    level = guess_heading_level(item.get("text_level"), text)
    metadata = {
        "engine": "mineru",
        "raw_type": raw_type,
        "tile_index": tile.tile_index,
        "img_path": item.get("img_path"),
        "text_level": item.get("text_level"),
    }

    if raw_type in {"header", "footer"}:
        return None

    if raw_type == "text":
        if level is not None:
            return LayoutBlock(
                block_type=BlockType.HEADING,
                text=text,
                page_no=tile.page_no,
                bbox=bbox,
                reading_order=tile_order,
                level=level,
                metadata=metadata,
            )
        block_type = BlockType.LIST if looks_like_list(text) else BlockType.PARAGRAPH
        return LayoutBlock(
            block_type=block_type,
            text=text,
            page_no=tile.page_no,
            bbox=bbox,
            reading_order=tile_order,
            metadata=metadata,
        )

    if raw_type == "table":
        caption = as_caption(item.get("table_caption"))
        footnote = as_caption(item.get("table_footnote"))
        if footnote:
            caption = f"{caption} {footnote}".strip() if caption else footnote
        table = html_table_to_table_block(
            str(item.get("table_body") or ""),
            caption=caption,
        )
        if table is None:
            # 表格结构失败：保留纯文本，避免丢内容。
            fallback = html_to_text(str(item.get("table_body") or "")) or text
            if not fallback:
                return None
            metadata["table_parse_failed"] = True
            return LayoutBlock(
                block_type=BlockType.PARAGRAPH,
                text=fallback,
                page_no=tile.page_no,
                bbox=bbox,
                reading_order=tile_order,
                metadata=metadata,
            )
        return LayoutBlock(
            block_type=BlockType.TABLE,
            text=table.markdown or "",
            page_no=tile.page_no,
            bbox=bbox,
            reading_order=tile_order,
            table=table,
            metadata=metadata,
        )

    if raw_type in {"image", "figure", "chart"}:
        caption = as_caption(item.get("image_caption")) or as_caption(
            item.get("chart_caption")
        )
        img_path = item.get("img_path")
        return LayoutBlock(
            block_type=BlockType.FIGURE,
            text=caption or "",
            page_no=tile.page_no,
            bbox=bbox,
            reading_order=tile_order,
            figure=FigureBlock(
                image_ref=str(img_path or ""),
                caption=caption,
                alt_text=as_caption(item.get("image_footnote")),
            ),
            metadata=metadata,
        )

    if raw_type in {"equation", "interline_equation"}:
        latex = text or str(item.get("latex") or "")
        if not latex:
            return None
        return LayoutBlock(
            block_type=BlockType.FORMULA,
            text=latex,
            page_no=tile.page_no,
            bbox=bbox,
            reading_order=tile_order,
            formula=FormulaBlock(latex=latex, text=text or None),
            metadata=metadata,
        )

    if raw_type == "list":
        return LayoutBlock(
            block_type=BlockType.LIST,
            text=text,
            page_no=tile.page_no,
            bbox=bbox,
            reading_order=tile_order,
            metadata=metadata,
        )

    if raw_type == "code":
        return LayoutBlock(
            block_type=BlockType.CODE,
            text=text,
            page_no=tile.page_no,
            bbox=bbox,
            reading_order=tile_order,
            metadata=metadata,
        )

    if text:
        return LayoutBlock(
            block_type=BlockType.PARAGRAPH,
            text=text,
            page_no=tile.page_no,
            bbox=bbox,
            reading_order=tile_order,
            metadata=metadata,
        )
    return None
