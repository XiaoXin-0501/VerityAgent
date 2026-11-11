"""幻灯片切片：一页一 chunk；沿用文本打包，但默认不切得过碎。"""

from __future__ import annotations

from .text import TextChunker

__all__ = ["SlideChunker"]


class SlideChunker(TextChunker):
    """幻灯片通常一页一个 block；文本打包仍受 max_tokens 保护。"""
