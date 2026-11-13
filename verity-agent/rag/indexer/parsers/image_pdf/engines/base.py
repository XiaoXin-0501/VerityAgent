"""版式引擎协议。"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from config.rag import ImagePdfConfig

from ..models import LayoutDocument, RenderedTile

__all__ = ["LayoutEngine"]


class LayoutEngine(Protocol):
    """图片/扫描页版式解析引擎协议。"""

    name: str

    def available(self) -> bool:
        """引擎依赖是否可用。"""
        ...

    def parse(
        self,
        tiles: list[RenderedTile],
        *,
        config: ImagePdfConfig,
        work_dir: str | Path,
    ) -> LayoutDocument:
        """解析渲染后的 tile，返回结构化块。"""
        ...
