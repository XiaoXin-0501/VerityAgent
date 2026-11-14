"""默认 chunker 注册表（v1：文本、列表、表格、代码均输出文本 chunk）。

- 标题不作为独立 chunk，只作为后续块的完整 `heading_path` 上下文；
- 实际入库由 `SectionAwareChunker` 按标题层级组织：默认从二级标题开始，
  内容超限才下钻三级/四级，避免一开始就切得过碎；
- 紧贴表格/代码的短描述会吸附到特殊块，避免代码/表格脱离语义；
- Markdown 代码围栏交 `CodeChunker` 按行/逻辑块切分，不启用代码 AST；
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
from .describer import (
    ChatCodeDescriber,
    ChatTableDescriber,
    CodeDescriber,
    TableDescriber,
)
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
    "ChatTableDescriber",
    "ChatCodeDescriber",
    "TableDescriber",
    "CodeDescriber",
    "ListChunker",
    "TableChunker",
    "CodeChunker",
]


def build_chunker_registry(
    config: IndexerConfig | None = None,
    *,
    table_describer: TableDescriber | None = None,
    code_describer: CodeDescriber | None = None,
) -> ChunkerRegistry:
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
            CodeChunker(
                **{
                    **common,
                    "overlap_tokens": 0,
                    "code_describer": code_describer,
                    "semantic_embedding_enabled": (
                        config.semantic_embedding_enabled
                    ),
                    "semantic_heading_min_tokens": (
                        config.semantic_heading_min_tokens
                    ),
                    "semantic_description_max_chars": (
                        config.semantic_description_max_chars
                    ),
                    "semantic_code_max_input_chars": (
                        config.semantic_code_max_input_chars
                    ),
                }
            ),
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
                    "min_table_confidence": (
                        config.quality.min_table_confidence
                    ),
                    "table_describer": table_describer,
                    "semantic_embedding_enabled": (
                        config.semantic_embedding_enabled
                    ),
                    "semantic_heading_min_tokens": (
                        config.semantic_heading_min_tokens
                    ),
                    "semantic_preview_rows": config.semantic_preview_rows,
                    "semantic_description_max_chars": (
                        config.semantic_description_max_chars
                    ),
                }
            ),
        ),
    ]
    for spec, chunker in entries:
        registry.register(spec, chunker)
    return registry
