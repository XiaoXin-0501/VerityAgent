"""按标题层级组织的章节切片策略。

核心思路（由粗到细、按需下钻）：
1. 用解析器产出的 ``HEADING`` 块构建章节树，每个节点带完整标题路径；
2. 取文档中“第二浅”的标题层级作为默认切片层级（通常就是二级标题；
   如果文档只有一级标题，则用一级标题）；
3. 默认层级节点的整棵子树不超过 ``max_tokens`` 时，在该标题下打包正文，
   不继续下钻到三级/四级标题；**整节组装只适用于纯正文/列表**；
4. 只有超过 ``max_tokens`` 才进入下一级标题，递归处理；
5. 表格/代码保持独立切片（不变量：不跨 block_type），并携带其所在的最深
   标题路径；章节里只要出现特殊块，该章节不做整节组装；
6. 紧贴表格/代码的短描述会被吸附到该特殊块，避免代码/表格脱离语义。

这样既不会一上来就按最细标题切成一堆小 chunk，也不会让单个 chunk 超过
硬上限；标题路径只进入 embedding，不写入正文。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from config.rag import IndexerConfig
from rag.shared.enums import (
    BlockType,
    DocumentType,
    QualityFlag,
    SectionType,
)
from rag.shared.models import ChunkDraft, ContentBlock, ProcessedBlock
from rag.shared.registry import ChunkerRegistry
from utils.text import smart_join, split_sentences
from utils.tokens import estimate_tokens

from .association import SpecialContext, associate_special_contexts
from .base import BaseChunker, coalesce_text_chunks, section_type_for
from .packing import Unit, build_group_chunk, pack_units
from ..policy import IndexerPolicy, get_indexer_policy

__all__ = ["SectionNode", "SectionAwareChunker", "build_section_tree"]

# 参与“章节内正文打包”的块类型。
_TEXT_BLOCK_TYPES = frozenset(
    {BlockType.PARAGRAPH, BlockType.LIST, BlockType.FOOTNOTE}
)
# 独立切片、不与其他正文合并的块类型。
_SPECIAL_BLOCK_TYPES = frozenset({BlockType.TABLE, BlockType.CODE})


@dataclass(slots=True)
class SectionNode:
    """章节树节点：标题、完整路径、按文档顺序排列的内容/子节点。"""

    level: int
    title: str | None
    path: list[str]
    items: list[object] = field(default_factory=list)
    total_tokens: int = 0


def build_section_tree(blocks: list[ContentBlock]) -> SectionNode:
    """根据 HEADING 块构建章节树；内容块挂到当前最深标题下。"""
    root = SectionNode(level=0, title=None, path=[])
    stack: list[SectionNode] = [root]
    for block in blocks:
        if block.block_type is BlockType.HEADING:
            level = max(1, int(block.heading_level or 1))
            while len(stack) > 1 and stack[-1].level >= level:
                stack.pop()
            parent = stack[-1]
            node = SectionNode(
                level=level,
                title=block.text,
                path=[*parent.path, block.text],
            )
            parent.items.append(node)
            stack.append(node)
            continue
        stack[-1].items.append(block)
    return root


def _is_chunkable(block: ContentBlock) -> bool:
    """判断块是否会产生 chunk；图/图注/目录等不参与 token 统计。"""
    return (
        block.block_type in _TEXT_BLOCK_TYPES
        or block.block_type in _SPECIAL_BLOCK_TYPES
    )


def _compute_totals(
    node: SectionNode,
    contexts: dict[UUID, SpecialContext],
) -> int:
    """后序计算每个节点子树的可切片 token 总量（含吸附描述）。"""
    total = 0
    for item in node.items:
        if isinstance(item, SectionNode):
            total += _compute_totals(item, contexts)
        elif isinstance(item, ContentBlock) and _is_chunkable(item):
            total += estimate_tokens(item.text)
            context = contexts.get(item.block_id)
            if context is not None:
                total += context.token_count
    node.total_tokens = total
    return total


def _collect_content_blocks(node: SectionNode) -> list[ContentBlock]:
    """按阅读顺序收集节点子树中的可切片内容块。"""
    blocks: list[ContentBlock] = []
    for item in node.items:
        if isinstance(item, SectionNode):
            blocks.extend(_collect_content_blocks(item))
        elif isinstance(item, ContentBlock) and _is_chunkable(item):
            blocks.append(item)
    return blocks


def _base_heading_level(blocks: list[ContentBlock]) -> int:
    """选择默认切片层级：文档中“第二浅”的标题级别。

    例如 ``[1,2,3]`` → 2，``[1,5]`` → 5，``[1]`` → 1。默认从二级标题
    （或等价层级）开始，只有内容超限时才继续下钻。
    """
    levels = sorted(
        {
            max(1, int(block.heading_level or 1))
            for block in blocks
            if block.block_type is BlockType.HEADING
        }
    )
    if not levels:
        return 1
    return levels[1] if len(levels) > 1 else levels[0]


class SectionAwareChunker:
    """按章节层级进行由粗到细的切片。"""

    def __init__(
        self,
        config: IndexerConfig,
        chunkers: ChunkerRegistry,
        *,
        policy: IndexerPolicy | None = None,
    ) -> None:
        """保存配置、特殊块切片器注册表与索引策略。"""
        self.config = config
        self.chunkers = chunkers
        self.policy = policy or get_indexer_policy()
        # 唯一硬上限来源：IndexerConfig.max_chunk_tokens。
        self.max_tokens = config.max_chunk_tokens
        self.target_tokens = min(config.chunk_size_tokens, self.max_tokens)
        self.packer = BaseChunker(
            target_tokens=self.target_tokens,
            max_tokens=self.max_tokens,
            min_tokens=config.min_chunk_tokens,
            overlap_tokens=config.chunk_overlap_tokens,
            chunking_version=config.chunking_version,
            chunk_id_namespace=config.chunk_id_namespace,
        )

    async def chunk(
        self,
        blocks: list[ContentBlock],
        *,
        kb_id: UUID,
        doc_type: DocumentType,
        document_description: str | None = None,
    ) -> list[ChunkDraft]:
        """把整篇文档的块切成 chunk 列表。"""
        contexts = {}
        if self.config.special_context_enabled:
            # 描述最多占 chunk 预算的一半，避免代码/表格本体没有空间。
            context_max = min(
                self.config.special_context_max_tokens,
                max(1, self.max_tokens // 2),
            )
            context_short = min(
                self.config.special_context_short_tokens,
                context_max,
            )
            blocks, contexts = associate_special_contexts(
                blocks,
                max_tokens=context_max,
                short_tokens=context_short,
                policy=self.policy,
            )
        root = build_section_tree(blocks)
        _compute_totals(root, contexts)
        base_level = _base_heading_level(blocks)

        chunks: list[ChunkDraft] = []
        index = 0

        def flush_text(buffer: list[ContentBlock], path: list[str]) -> None:
            """把缓存中的同类正文块打包成 chunk。"""
            nonlocal index
            if not buffer:
                return
            doc_id = buffer[0].doc_id
            block_type = buffer[0].block_type
            section_type = section_type_for(block_type)
            # 低置信度 OCR 文本保留，但打标供检索降权/过滤。
            low_ocr = any(
                block.metadata.get("ocr")
                and block.confidence is not None
                and block.confidence < self.config.quality.min_ocr_confidence
                for block in buffer
            )
            unit_flags = {QualityFlag.LOW_OCR} if low_ocr else set()
            joiner = (
                (lambda parts: "\n".join(parts))
                if section_type is SectionType.LIST
                else smart_join
            )
            units: list[Unit] = []
            for block in buffer:
                text = (block.text or "").strip()
                if not text:
                    continue
                pieces = (
                    [line.strip() for line in text.split("\n") if line.strip()]
                    if section_type is SectionType.LIST
                    else split_sentences(text) or [text]
                )
                units.extend(
                    Unit(
                        text=piece,
                        block_id=block.block_id,
                        page_no=block.page_no or None,
                        language=block.language,
                        quality_flags=set(unit_flags),
                    )
                    for piece in pieces
                    if piece
                )
            buffer.clear()
            if not units:
                return
            heading_text = " > ".join(path) if path else None
            groups = pack_units(
                units,
                target_tokens=self.target_tokens,
                max_tokens=self.max_tokens,
                overlap_tokens=self.config.chunk_overlap_tokens,
            )
            for group in groups:
                chunks.append(
                    build_group_chunk(
                        self.packer,
                        group,
                        doc_id=doc_id,
                        kb_id=kb_id,
                        index=index,
                        block_type=block_type,
                        section_type=section_type,
                        heading_text=heading_text,
                        heading_path=path,
                        joiner=joiner,
                        document_description=document_description,
                    )
                )
                index += 1

        async def emit_special(block: ContentBlock) -> None:
            """表格/代码走各自切片器，保留最深标题路径与吸附的描述。"""
            nonlocal index
            resolved = self.chunkers.resolve(doc_type, block.block_type)
            if resolved is None:
                return
            _spec, chunker = resolved
            context = contexts.get(block.block_id)
            metadata = {}
            if context is not None:
                metadata = {
                    "context_text": context.text,
                    "context_block_ids": context.block_ids,
                }
            if document_description:
                metadata["document_description"] = document_description
            produced = await chunker.chunk(
                block,
                ProcessedBlock(block=block, text=block.text, metadata=metadata),
                kb_id=kb_id,
                doc_type=doc_type,
                chunking_version=self.config.chunking_version,
                start_index=index,
            )
            chunks.extend(produced)
            index += len(produced)

        async def assemble(node: SectionNode, path: list[str]) -> bool:
            """把“纯正文/列表”章节整体组装成一个语义完整的 chunk。

            硬规则（《RAG 模块数据结构设计》§5.4 / §15）：表格、代码等特殊块
            必须各自独立切片，不能与正文混进同一个 chunk。因此本函数只处理
            ``_TEXT_BLOCK_TYPES``；章节里只要出现特殊块，调用方会回退到
            ``process_items``，由 TableChunker / CodeChunker 保留结构、表头
            和语义描述。

            组装后超过 ``max_tokens`` 时返回 ``False``，由调用方回退到按块处理。
            """
            nonlocal index
            content = _collect_content_blocks(node)
            if not content or any(
                block.block_type not in _TEXT_BLOCK_TYPES
                for block in content
            ):
                return False

            parts: list[str] = []
            block_ids: list[UUID] = []
            flags: set[QualityFlag] = set()
            block_types: set[BlockType] = set()
            page_no: int | None = None
            language: str | None = None

            def add_part(value: str | None) -> None:
                """追加非空文本片段。"""
                if value and value.strip():
                    parts.append(value.strip())

            def add_ids(ids: list[UUID]) -> None:
                """按首次出现顺序合并来源 block id。"""
                for block_id in ids:
                    if block_id not in block_ids:
                        block_ids.append(block_id)

            for block in content:
                block_types.add(block.block_type)
                add_part(block.text)
                add_ids([block.block_id])
                if (
                    block.metadata.get("ocr")
                    and block.confidence is not None
                    and block.confidence < self.config.quality.min_ocr_confidence
                ):
                    flags.add(QualityFlag.LOW_OCR)
                if page_no is None and block.page_no:
                    page_no = block.page_no
                if language is None and block.language:
                    language = block.language

            text = (
                smart_join(parts)
                if block_types <= {BlockType.PARAGRAPH, BlockType.FOOTNOTE}
                else "\n".join(parts)
            ).strip()
            if not text or estimate_tokens(text) > self.max_tokens:
                return False

            if block_types == {BlockType.LIST}:
                section_type, block_type = SectionType.LIST, BlockType.LIST
            else:
                section_type, block_type = (
                    SectionType.PARAGRAPH,
                    BlockType.PARAGRAPH,
                )

            heading_text = " > ".join(path) if path else None
            # 文档描述只作为 chunk metadata 供展示/审计，不进入 embedding：
            # 同一文档的所有 chunk 共享描述会让向量趋同，稀释检索信号（D-103）。
            embedding_text = (
                f"{heading_text}\n{text}" if heading_text else text
            )
            metadata = {
                "mixed": len(block_types) > 1,
                "block_types": sorted(item.value for item in block_types),
            }
            if document_description:
                metadata["document_description"] = document_description
            chunks.append(
                self.packer.build_chunk_from_text(
                    doc_id=content[0].doc_id,
                    kb_id=kb_id,
                    block_type=block_type,
                    section_type=section_type,
                    index=index,
                    text=text,
                    embedding_text=embedding_text,
                    block_ids=block_ids,
                    page_no=page_no,
                    heading=heading_text,
                    heading_path=path,
                    language=language,
                    quality_flags=flags,
                    metadata=metadata,
                )
            )
            index += 1
            return True

        async def process_items(node: SectionNode) -> None:
            """按文档顺序处理节点自身内容，并递归子标题。"""
            buffer: list[ContentBlock] = []
            for item in node.items:
                if isinstance(item, SectionNode):
                    flush_text(buffer, node.path)
                    await process(item)
                    continue
                if not isinstance(item, ContentBlock):
                    continue
                if item.block_type in _TEXT_BLOCK_TYPES:
                    if buffer and buffer[-1].block_type is not item.block_type:
                        flush_text(buffer, node.path)
                    buffer.append(item)
                elif item.block_type in _SPECIAL_BLOCK_TYPES:
                    flush_text(buffer, node.path)
                    await emit_special(item)
            flush_text(buffer, node.path)

        async def process(node: SectionNode) -> None:
            """按“默认层级优先、超限再下钻”的规则处理章节节点。

            - 比默认层级更浅的节点（如一级标题）：按文档顺序处理自身正文和子标题；
            - 默认层级及更深的节点：整棵子树不超过 max_tokens 时，只有“纯正文/
              列表”章节才整体组装成一个 chunk；含表格/代码的章节必须回退到按
              块处理，保证特殊块独立切片（§5.4 不变量）；
            - 超限时按文档顺序处理自身正文，并逐个递归子标题。
            """
            if (
                node.level >= base_level
                and node.total_tokens <= self.max_tokens
            ):
                content = _collect_content_blocks(node)
                # 只有一个特殊块时仍走专门切片器，保留表格行组/代码元数据。
                if (
                    len(content) == 1
                    and content[0].block_type in _SPECIAL_BLOCK_TYPES
                ):
                    await emit_special(content[0])
                    return
                # 只有纯正文/列表才能整节组装；混入表格/代码时必须保持结构边界。
                if await assemble(node, node.path):
                    return
            await process_items(node)

        await process(root)
        # 同一标题下被表格/代码隔开的正文片段再合并一次，减少碎片 chunk。
        return coalesce_text_chunks(
            chunks,
            max_tokens=self.max_tokens,
            chunking_version=self.config.chunking_version,
            chunk_id_namespace=self.config.chunk_id_namespace,
        )
