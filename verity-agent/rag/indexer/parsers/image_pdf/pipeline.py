"""纯图片/扫描 PDF 解析流水线。

职责：
- 渲染页面，超长页切 tile；
- 按配置选择 MinerU / RapidOCR 版式引擎；
- 引擎失败时按策略降级；
- 把 LayoutDocument 映射为 ParsedDocument，交给现有切片器。
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from uuid import UUID

from config.rag import ImagePdfConfig
from exception import RagError, RagErrorCode
from rag.shared.models import DocumentProfile, DocumentSource, ParsedDocument

from .engines.base import LayoutEngine
from .engines.mineru import MineruLayoutEngine
from .engines.rapid import RapidLayoutEngine
from .mapper import layout_document_to_parsed
from .models import LayoutDocument
from .renderer import render_pdf_tiles

__all__ = ["ImagePdfPipeline"]


class ImagePdfPipeline:
    """图片 PDF 版式解析门面。"""

    def __init__(self, config: ImagePdfConfig) -> None:
        """保存配置。"""
        self.config = config

    def parse(
        self,
        pdf_bytes: bytes,
        *,
        page_numbers: list[int],
        total_pages: int,
        source: DocumentSource,
        doc_id: UUID,
        kb_id: UUID,
        profile: DocumentProfile,
    ) -> ParsedDocument:
        """解析指定图片页；返回仅包含这些页的 ParsedDocument。"""
        if not self.config.enabled or not page_numbers:
            raise RagError(
                RagErrorCode.INVALID_REQUEST,
                "未启用图片 PDF 解析或没有可处理页面",
                details={"engine": self.config.engine},
            )
        page_numbers = sorted(set(page_numbers))
        warnings: list[str] = []
        if len(page_numbers) > self.config.max_pages:
            warnings.append(
                f"图片页数超过上限 {self.config.max_pages}，仅处理前 "
                f"{self.config.max_pages} 页"
            )
            page_numbers = page_numbers[: self.config.max_pages]
        print(
            f"[image_pdf] 开始解析 {len(page_numbers)} 页，engine={self.config.engine}",
            flush=True,
        )

        with tempfile.TemporaryDirectory(prefix="verity-image-pdf-") as tmp:
            work_dir = Path(tmp)
            tiles_dir = work_dir / "tiles"
            tiles, page_sizes = render_pdf_tiles(
                pdf_bytes,
                page_numbers=page_numbers,
                config=self.config,
                work_dir=tiles_dir,
            )
            if not tiles:
                raise RagError(
                    RagErrorCode.INVALID_REQUEST,
                    "图片 PDF 没有可渲染页面",
                )
            print(
                f"[image_pdf] 渲染完成：{len(tiles)} 个 tile，开始版式识别",
                flush=True,
            )

            layout, engine_warnings = self._run_engines(tiles, work_dir)
            print(
                f"[image_pdf] 版式识别完成：engine={layout.engine}，"
                f"blocks={len(layout.blocks)}",
                flush=True,
            )
            layout.page_sizes.update(page_sizes)
            layout.warnings = [*warnings, *engine_warnings, *layout.warnings]

            parsed = layout_document_to_parsed(
                layout,
                source=source,
                doc_id=doc_id,
                kb_id=kb_id,
                profile=profile,
                config=self.config,
                parser_id=f"image_pdf_{layout.engine}",
                parser_version=layout.engine_version or "1",
                page_count=total_pages,
            )
        return parsed

    def _run_engines(
        self,
        tiles,
        work_dir: Path,
    ) -> tuple[LayoutDocument, list[str]]:
        """按配置运行主引擎；auto 模式下失败可降级。"""
        engine_name = (self.config.engine or "auto").strip().lower()
        warnings: list[str] = []
        if engine_name == "none":
            raise RagError(
                RagErrorCode.INVALID_CONFIG,
                "图片 PDF 解析引擎为 none",
            )

        candidates: list[LayoutEngine] = []
        if engine_name in {"auto", "mineru"}:
            candidates.append(MineruLayoutEngine(self.config))
        if engine_name in {"auto", "rapid"}:
            candidates.append(RapidLayoutEngine(self.config))
        if engine_name not in {"auto", "mineru", "rapid"}:
            raise RagError(
                RagErrorCode.INVALID_CONFIG,
                f"未知的图片 PDF 引擎: {self.config.engine}",
                details={"engine": self.config.engine},
            )

        available = [engine for engine in candidates if engine.available()]
        if not available:
            raise RagError(
                RagErrorCode.INVALID_CONFIG,
                "没有可用的图片 PDF 版式引擎，请安装 mineru[core] 或 rapidocr",
                details={"engine": self.config.engine},
            )

        last_error: Exception | None = None
        for engine in available:
            print(f"[image_pdf] 使用 {engine.name} 引擎解析...", flush=True)
            try:
                layout = engine.parse(
                    tiles,
                    config=self.config,
                    work_dir=work_dir,
                )
                if _has_meaningful_content(layout):
                    return layout, warnings
                warnings.append(f"{engine.name} 未识别到有效内容，尝试降级")
            except RagError as exc:
                if engine_name != "auto":
                    raise
                warnings.append(f"{engine.name} 解析失败，尝试降级: {exc.message}")
                last_error = exc
            except Exception as exc:
                if engine_name != "auto":
                    raise RagError(
                        RagErrorCode.INTERNAL,
                        "图片 PDF 解析失败",
                        details={"engine": engine.name, "cause": str(exc)},
                    ) from exc
                warnings.append(f"{engine.name} 解析异常，尝试降级: {exc}")
                last_error = exc

        if last_error is not None:
            raise RagError(
                RagErrorCode.DEPENDENCY_UNAVAILABLE,
                "所有图片 PDF 引擎均失败",
                details={"cause": str(last_error)},
                retryable=True,
            ) from last_error
        raise RagError(
            RagErrorCode.INVALID_REQUEST,
            "图片 PDF 未识别到有效内容",
        )


def _has_meaningful_content(layout: LayoutDocument) -> bool:
    """判断引擎输出是否至少有文本或表格，避免把空结果当成功。"""
    for block in layout.blocks:
        if block.table is not None:
            return True
        if block.text.strip():
            return True
    return False
