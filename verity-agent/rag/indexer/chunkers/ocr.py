"""OCR 文本切片：按段落打包，低置信度由 quality_flags 标记。"""

from __future__ import annotations

from .text import OcrChunker

__all__ = ["OcrChunker"]
