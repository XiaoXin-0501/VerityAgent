"""默认 chunker 注册表（v1：纯文本、列表、表格均输出文本 chunk）。

- 标题不作为独立 chunk，只作为后续块的 `heading_text` 上下文；
- Markdown 代码围栏按普通文本切分，不启用代码 AST 切片；
- 表格（Markdown/Excel/DOCX 表格）用 `TableChunker` 输出带表头的文本行；
- 图片、图注、公式、OCR、幻灯片等策略暂不注册。
"""

from __future__ import annotations

from config.rag import IndexerConfig
from rag.shared.enums import BlockType
from rag.shared.models import ChunkingSpec
from rag.shared.registry import ChunkerRegistry

from .base import BaseChunker, Chunker, section_type_for
from .list import ListChunker
from .table import TableChunker
from .text import TextChunker

__all__ = [
    "build_chunker_registry",
    "BaseChunker",
    "Chunker",
    "section_type_for",
    "TextChunker",
    "ListChunker",
    "TableChunker",
]


def build_chunker_registry(config: IndexerConfig | None = None) -> ChunkerRegistry:
    config = config or IndexerConfig()
    common = {
        "target_tokens": config.chunk_size_tokens,
        "max_tokens": config.max_chunk_tokens,
        "min_tokens": config.min_chunk_tokens,
        "overlap_tokens": config.chunk_overlap_tokens,
        "chunking_version": config.chunking_version,
        "chunk_id_namespace": config.chunk_id_namespace,
    }
    registry = ChunkerRegistry()
    entries = [
        (
            ChunkingSpec(
                strategy_id="text",
                version=config.chunking_version,
                block_types={
                    BlockType.PARAGRAPH,
                    BlockType.FOOTNOTE,
                    # Markdown 代码围栏在 v1 按纯文本处理
                    BlockType.CODE,
                },
                target_tokens=config.chunk_size_tokens,
                max_tokens=config.max_chunk_tokens,
                min_tokens=config.min_chunk_tokens,
                overlap_tokens=config.chunk_overlap_tokens,
                priority=100,
            ),
            TextChunker(**common),
        ),
        (
            ChunkingSpec(
                strategy_id="list",
                version=config.chunking_version,
                block_types={BlockType.LIST},
                target_tokens=config.chunk_size_tokens,
                max_tokens=config.max_chunk_tokens,
                min_tokens=config.min_chunk_tokens,
                overlap_tokens=config.chunk_overlap_tokens,
                priority=100,
            ),
            ListChunker(**common),
        ),
        (
            ChunkingSpec(
                strategy_id="table",
                version=config.chunking_version,
                block_types={BlockType.TABLE},
                target_tokens=config.chunk_size_tokens,
                max_tokens=config.max_chunk_tokens,
                min_tokens=config.min_chunk_tokens,
                overlap_tokens=0,
                priority=100,
            ),
            TableChunker(**{**common, "overlap_tokens": 0}),
        ),
    ]
    for spec, chunker in entries:
        registry.register(spec, chunker)
    return registry
