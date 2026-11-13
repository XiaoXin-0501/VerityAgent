"""默认 chunker 注册表（v1：纯文本、列表、表格均输出文本 chunk）。

- 标题不作为独立 chunk，只作为后续块的完整 `heading_path` 上下文；
- 实际入库由 `SectionAwareChunker` 按标题层级组织：默认从二级标题开始，
  内容超限才下钻三级/四级，避免一开始就切得过碎；
- 紧贴表格/代码的短描述会吸附到特殊块，避免代码/表格脱离语义；
- Markdown 代码围栏按普通文本切分，不启用代码 AST 切片；
- 表格（Markdown/Excel/DOCX 表格）用 `TableChunker` 输出带表头的文本行；
- 图片、图注、公式、OCR、幻灯片等策略暂不注册。
"""

from __future__ import annotations

from config.rag import IndexerConfig
from rag.shared.enums import BlockType
from rag.shared.models import ChunkingSpec
from rag.shared.registry import ChunkerRegistry

from .base import (
    BaseChunker,
    Chunker,
    coalesce_text_chunks,
    merge_adjacent_chunks,
    section_type_for,
)
from .code import CodeChunker
from .list import ListChunker
from .section import SectionAwareChunker, SectionNode, build_section_tree
from .table import TableChunker
from .text import TextChunker

__all__ = [
    "build_chunker_registry",
    "BaseChunker",
    "Chunker",
    "section_type_for",
    "merge_adjacent_chunks",
    "coalesce_text_chunks",
    "SectionAwareChunker",
    "SectionNode",
    "build_section_tree",
    "TextChunker",
    "ListChunker",
    "TableChunker",
    "CodeChunker",
]


def build_chunker_registry(config: IndexerConfig | None = None) -> ChunkerRegistry:
    """按配置注册 v1 的文本/列表/表格/代码切片策略。"""
    config = config or IndexerConfig()
    # 唯一硬上限来源：IndexerConfig.max_chunk_tokens。
    max_tokens = config.max_chunk_tokens
    target_tokens = min(config.chunk_size_tokens, max_tokens)
    common = {
        "target_tokens": target_tokens,
        "max_tokens": max_tokens,
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
                },
                target_tokens=target_tokens,
                max_tokens=max_tokens,
                min_tokens=config.min_chunk_tokens,
                overlap_tokens=config.chunk_overlap_tokens,
                priority=100,
            ),
            TextChunker(**common),
        ),
        (
            ChunkingSpec(
                strategy_id="code",
                version=config.chunking_version,
                block_types={BlockType.CODE},
                target_tokens=target_tokens,
                max_tokens=max_tokens,
                min_tokens=config.min_chunk_tokens,
                # 代码不 overlap
                overlap_tokens=0,
                priority=100,
            ),
            CodeChunker(**{**common, "overlap_tokens": 0}),
        ),
        (
            ChunkingSpec(
                strategy_id="list",
                version=config.chunking_version,
                block_types={BlockType.LIST},
                target_tokens=target_tokens,
                max_tokens=max_tokens,
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
                target_tokens=target_tokens,
                max_tokens=max_tokens,
                min_tokens=config.min_chunk_tokens,
                overlap_tokens=0,
                priority=100,
            ),
            TableChunker(
                **{
                    **common,
                    "overlap_tokens": 0,
                    "row_group_size": config.table_row_group_size,
                }
            ),
        ),
    ]
    for spec, chunker in entries:
        registry.register(spec, chunker)
    return registry
