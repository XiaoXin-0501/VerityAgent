"""工作表切片：复用表格策略（表头重复、按行组切）。"""

from __future__ import annotations

from .table import TableChunker

__all__ = ["SpreadsheetChunker"]


class SpreadsheetChunker(TableChunker):
    """工作表与表格的切片规则一致。"""
