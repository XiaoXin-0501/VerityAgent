"""内容处理器协议。"""

from __future__ import annotations

from typing import Protocol

from ...shared.models import ContentBlock, ProcessedBlock

__all__ = ["BlockProcessor"]


class BlockProcessor(Protocol):
    async def process(self, block: ContentBlock) -> ProcessedBlock:
        """清洗/结构化一个块，返回处理后文本和 embedding 文本。"""
        ...
