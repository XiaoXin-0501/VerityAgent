"""切片器协议与公共辅助。

``BaseChunker`` 负责确定性 chunk id / 正文与 embedding 文本分离；
``merge_short_chunks`` 在切片完成后合并相邻的过短文本块，避免短问答/列表项
被质量闸门当成噪音丢弃。
"""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from rag.shared.enums import BlockType, DocumentType, QualityFlag, SectionType
from rag.shared.ids import chunk_hash, deterministic_chunk_id
from rag.shared.models import ChunkDraft, ContentBlock, ProcessedBlock
from utils.tokens import estimate_tokens

__all__ = [
    "Chunker",
    "BaseChunker",
    "section_type_for",
    "merge_adjacent_chunks",
    "merge_small_sibling_chunks",
    "coalesce_text_chunks",
    "semantic_embedding_text",
]

_SECTION_TYPES = {
    BlockType.TABLE: SectionType.TABLE,
    BlockType.CODE: SectionType.CODE,
    BlockType.FIGURE: SectionType.FIGURE,
    BlockType.LIST: SectionType.LIST,
    BlockType.FORMULA: SectionType.FORMULA,
    BlockType.REFERENCE: SectionType.REFERENCE,
}

# 只有纯文本类切片允许跨块合并；表格/代码保持结构边界。
_MERGEABLE_SECTION_TYPES = {SectionType.PARAGRAPH, SectionType.LIST}


def section_type_for(block_type: BlockType) -> SectionType:
    """把精细 ``BlockType`` 映射为落库用 ``SectionType``。"""
    return _SECTION_TYPES.get(block_type, SectionType.PARAGRAPH)


def semantic_embedding_text(
    heading: str | None,
    description: str | None,
) -> str:
    """特殊块（表格/代码）的向量文本：只拼 heading + 描述。

    表格/代码正文语义密度低，整段向量化会稀释检索信号；正文仍落 Weaviate
    供 BM25/引用使用。没有 heading/描述时返回空串，由调用方回退旧策略。
    """
    parts = [item.strip() for item in (heading, description) if item and item.strip()]
    return "\n".join(dict.fromkeys(parts))


class BaseChunker:
    """所有切片器的基类：统一 token 预算、overlap 与确定性 id。"""

    def __init__(
        self,
        *,
        target_tokens: int = 300,
        max_tokens: int = 600,
        min_tokens: int = 40,
        overlap_tokens: int = 80,
        chunking_version: str = "v1",
        chunk_id_namespace: str = "verity:chunk",
    ) -> None:
        """保存切片预算与确定性 id 参数。"""
        self.max_tokens = max(1, int(max_tokens))
        # 目标不能高于硬上限，否则单块/章节的 target 判定会与上限冲突。
        self.target_tokens = min(max(1, int(target_tokens)), self.max_tokens)
        self.min_tokens = min_tokens
        self.overlap_tokens = overlap_tokens
        self.chunking_version = chunking_version
        self.chunk_id_namespace = chunk_id_namespace

    def build_chunk(
        self,
        *,
        block: ContentBlock,
        processed: ProcessedBlock,
        kb_id: UUID,
        index: int,
        text: str,
        embedding_text: str | None = None,
        section_type: SectionType | None = None,
        quality_flags: set[QualityFlag] | None = None,
        metadata: dict | None = None,
        extra_block_ids: list[UUID] | None = None,
        semantic_description: str | None = None,
    ) -> ChunkDraft:
        """构造一个 ``ChunkDraft``；正文与 embedding 文本分别保存。

        ``extra_block_ids`` 用于把吸附到特殊块的描述块一并记入来源；
        ``semantic_description`` 只由表格/代码切片器传入，用于落库观察。
        """
        body = text.strip()
        return self.build_chunk_from_text(
            doc_id=block.doc_id,
            kb_id=kb_id,
            block_type=block.block_type,
            section_type=section_type or section_type_for(block.block_type),
            index=index,
            text=body,
            embedding_text=embedding_text or body,
            block_ids=[block.block_id, *(extra_block_ids or [])],
            page_no=block.page_no or None,
            heading=block.heading_text,
            heading_path=block.heading_path,
            language=block.language,
            quality_flags=(
                set(processed.quality_flags)
                if quality_flags is None
                else quality_flags
            ),
            metadata=metadata or {},
            semantic_description=semantic_description,
        )

    def build_chunk_from_text(
        self,
        *,
        doc_id: UUID,
        kb_id: UUID,
        block_type: BlockType,
        section_type: SectionType,
        index: int,
        text: str,
        embedding_text: str,
        block_ids: list[UUID],
        page_no: int | None,
        heading: str | None,
        heading_path: list[str],
        language: str | None,
        quality_flags: set[QualityFlag],
        metadata: dict,
        semantic_description: str | None = None,
    ) -> ChunkDraft:
        """从任意文本片段构造 chunk，供跨块合并/按章节打包使用。"""
        body = text.strip()
        digest = chunk_hash(body)
        return ChunkDraft(
            chunk_id=deterministic_chunk_id(
                kb_id=kb_id,
                doc_id=doc_id,
                chunk_index=index,
                content_hash=digest,
                chunking_version=self.chunking_version,
                namespace=self.chunk_id_namespace,
            ),
            doc_id=doc_id,
            kb_id=kb_id,
            chunk_index=index,
            block_ids=list(block_ids),
            block_type=block_type,
            section_type=section_type,
            text=body,
            embedding_text=embedding_text.strip() or body,
            token_count=estimate_tokens(body),
            page_no=page_no,
            heading=heading,
            heading_path=list(heading_path),
            semantic_description=semantic_description,
            language=language,
            quality_flags=set(quality_flags),
            metadata=metadata,
        )

    def embedding_text(self, block: ContentBlock, text: str) -> str:
        """文本类 chunk 的 embedding：标题 + 正文；只进向量，不落库。"""
        if block.heading_text:
            return f"{block.heading_text}\n{text}"
        return text


class Chunker(Protocol):
    """切片器协议：一个块可以被切成多个 ChunkDraft，但不跨块类型。"""

    async def chunk(
        self,
        block: ContentBlock,
        processed: ProcessedBlock,
        *,
        kb_id: UUID,
        doc_type: DocumentType,
        chunking_version: str,
        start_index: int = 0,
    ) -> list[ChunkDraft]:
        """把一个块切成一个或多个 ChunkDraft；不跨块类型。"""
        ...


def _is_grouped_chunk(chunk: ChunkDraft) -> bool:
    """问答组 chunk 不参与跨块合并，避免问答边界被重新拼掉。

    视觉分组（``visual_group``）只是 PDF 版面上的“组内紧、组间松”，
    不是语义边界；同一标题下多个视觉组只要合并后不超过 max_tokens，
    应允许合并，避免 200+300 token 的两段被拆成两个 chunk。
    """
    return bool(
        chunk.metadata.get("qa")
        or chunk.metadata.get("qa_prefix")
        or chunk.metadata.get("qa_long")
    )


def _merge_two(
    previous: ChunkDraft,
    current: ChunkDraft,
    *,
    chunking_version: str,
    chunk_id_namespace: str,
) -> ChunkDraft:
    """合并两个 chunk，并重新计算正文、embedding、哈希与确定性 id。"""
    text = f"{previous.text.rstrip()}\n{current.text.lstrip()}".strip()
    digest = chunk_hash(text)
    embedding_text = (
        f"{previous.heading}\n{text}" if previous.heading else text
    )
    metadata = dict(previous.metadata)
    metadata["merged_chunks"] = int(metadata.get("merged_chunks", 1)) + 1
    return previous.model_copy(
        update={
            "chunk_id": deterministic_chunk_id(
                kb_id=previous.kb_id,
                doc_id=previous.doc_id,
                chunk_index=previous.chunk_index,
                content_hash=digest,
                chunking_version=chunking_version,
                namespace=chunk_id_namespace,
            ),
            "block_ids": [*previous.block_ids, *current.block_ids],
            "text": text,
            "embedding_text": embedding_text,
            "token_count": estimate_tokens(text),
            "heading_path": previous.heading_path or current.heading_path,
            "quality_flags": previous.quality_flags | current.quality_flags,
            "metadata": metadata,
        }
    )


def merge_adjacent_chunks(
    chunks: list[ChunkDraft],
    *,
    target_tokens: int,
    min_tokens: int,
    max_tokens: int,
    chunking_version: str,
    chunk_id_namespace: str = "verity:chunk",
) -> list[ChunkDraft]:
    """把同一标题下的相邻文本/列表 chunk 合并到目标大小。

    合并条件：
    - 只处理 ``PARAGRAPH`` / ``LIST``，表格、代码、公式保持边界；
    - 文档、KB、section_type、block_type、heading 必须一致；
    - 前一块小于 ``target_tokens`` 且合并后不超过 ``target_tokens``，用于把
      大量短段落打包成正常 chunk；
    - 任一块小于 ``min_tokens`` 且合并后不超过 ``max_tokens``，用于吸收短尾，
      避免被质量闸门当噪音丢弃；
    - 合并后复用前一个 chunk 的 ``chunk_index``，保留 gap（文档 §8.1）。
    """
    if not chunks:
        return []
    merged: list[ChunkDraft] = []
    for chunk in chunks:
        if merged:
            previous = merged[-1]
            combined = previous.token_count + chunk.token_count
            same_context = (
                previous.doc_id == chunk.doc_id
                and previous.kb_id == chunk.kb_id
                and previous.section_type == chunk.section_type
                and previous.block_type == chunk.block_type
                and previous.heading == chunk.heading
                and previous.section_type in _MERGEABLE_SECTION_TYPES
                and not _is_grouped_chunk(previous)
                and not _is_grouped_chunk(chunk)
            )
            should_merge = same_context and (
                (
                    (previous.token_count < min_tokens or chunk.token_count < min_tokens)
                    and combined <= max_tokens
                )
                or (
                    previous.token_count < target_tokens
                    and combined <= target_tokens
                )
            )
            if should_merge:
                merged_text = (
                    f"{previous.text.rstrip()}\n{chunk.text.lstrip()}".strip()
                )
                # 分隔符可能让实际文本略超预算，按真实文本再校验一次。
                if estimate_tokens(merged_text) > max_tokens:
                    should_merge = False
            if should_merge:
                merged[-1] = _merge_two(
                    previous,
                    chunk,
                    chunking_version=chunking_version,
                    chunk_id_namespace=chunk_id_namespace,
                )
                continue
        merged.append(chunk)
    return merged


def coalesce_text_chunks(
    chunks: list[ChunkDraft],
    *,
    max_tokens: int,
    chunking_version: str,
    chunk_id_namespace: str = "verity:chunk",
) -> list[ChunkDraft]:
    """合并同一标题下被表格/代码隔开的正文 chunk。

    表格/代码块会把连续的正文切成多个 buffer（例如“说明：”+代码+后续段落）。
    该函数忽略中间的特殊块，把同一 ``heading_path`` 的正文 chunk 合并到
    ``max_tokens``，减少“一段只有一句话”的碎片；段落与列表视为同一类正文，
    合并后保留前一个 chunk 的 section_type。遇到其他标题的正文 chunk 时
    停止跨段合并，保证章节顺序不被跨越。
    """
    if not chunks:
        return []
    result: list[ChunkDraft] = []
    pending: dict[tuple, int] = {}
    for chunk in chunks:
        if chunk.section_type not in _MERGEABLE_SECTION_TYPES:
            # 表格/代码不参与正文合并，但也不打断同标题正文的累计。
            result.append(chunk)
            continue
        if _is_grouped_chunk(chunk):
            # 问答 chunk 的边界由 QA 分组决定，不能再被同标题正文合并。
            result.append(chunk)
            pending.clear()
            continue
        # 段落/列表属于连续正文，同一标题下允许互相合并。
        key = (chunk.doc_id, chunk.kb_id, tuple(chunk.heading_path))
        index = pending.get(key)
        merged_text = (
            f"{result[index].text.rstrip()}\n{chunk.text.lstrip()}".strip()
            if index is not None
            else ""
        )
        if (
            index is not None
            and result[index].token_count + chunk.token_count <= max_tokens
            # 合并后的分隔符也会消耗 token，按真实文本再校验一次硬上限。
            and estimate_tokens(merged_text) <= max_tokens
        ):
            result[index] = _merge_two(
                result[index],
                chunk,
                chunking_version=chunking_version,
                chunk_id_namespace=chunk_id_namespace,
            )
            continue
        # 出现新的正文段（可能是另一个标题），只允许与紧随其后的同标题正文合并。
        pending.clear()
        pending[key] = len(result)
        result.append(chunk)
    return result


def _prefixed_text(chunk: ChunkDraft) -> str:
    """给合并后的小节正文补上小节标题，避免 2.1/2.2 标签丢失。"""
    leaf = chunk.heading_path[-1] if chunk.heading_path else ""
    if leaf and not chunk.text.lstrip().startswith(leaf):
        return f"{leaf}\n{chunk.text}"
    return chunk.text


def _can_merge_siblings(
    previous: ChunkDraft,
    current: ChunkDraft,
    *,
    min_tokens: int,
    max_tokens: int,
    min_path_depth: int,
    parent_counts: dict[tuple[str, ...], int],
    sibling_min_children: int,
) -> bool:
    """同一父标题下的相邻小 chunk 是否应合并。"""
    if (
        previous.section_type not in _MERGEABLE_SECTION_TYPES
        or current.section_type not in _MERGEABLE_SECTION_TYPES
    ):
        return False
    if _is_grouped_chunk(previous) or _is_grouped_chunk(current):
        return False
    if previous.doc_id != current.doc_id or previous.kb_id != current.kb_id:
        return False
    a_path = previous.heading_path
    b_path = current.heading_path
    if not a_path or not b_path:
        return False
    if a_path[:-1] != b_path[:-1] or a_path[-1] == b_path[-1]:
        return False
    deep = len(a_path) >= min_path_depth and len(b_path) >= min_path_depth
    many_siblings = (
        parent_counts.get(tuple(a_path[:-1]), 0) >= sibling_min_children
    )
    if not (deep or many_siblings):
        return False
    if not (
        previous.token_count < min_tokens or current.token_count < min_tokens
    ):
        return False
    merged_text = f"{_prefixed_text(previous)}\n{_prefixed_text(current)}"
    return estimate_tokens(merged_text) <= max_tokens


def _merge_siblings(
    previous: ChunkDraft,
    current: ChunkDraft,
    *,
    chunking_version: str,
    chunk_id_namespace: str,
) -> ChunkDraft:
    """合并两个同父小节的 chunk，标题路径回退到公共父标题。"""
    merged = _merge_two(
        previous.model_copy(update={"text": _prefixed_text(previous)}),
        current.model_copy(update={"text": _prefixed_text(current)}),
        chunking_version=chunking_version,
        chunk_id_namespace=chunk_id_namespace,
    )
    parent = list(previous.heading_path[:-1])
    leaves = list(dict.fromkeys(
        [previous.heading_path[-1], current.heading_path[-1]]
    ))
    heading = " / ".join(leaves)[:120]
    metadata = dict(merged.metadata)
    metadata["merged_sections"] = leaves
    return merged.model_copy(
        update={
            "heading_path": parent,
            "heading": heading or (parent[-1] if parent else None),
            "metadata": metadata,
            "embedding_text": (
                f"{heading}\n{merged.text}" if heading else merged.text
            ),
            "token_count": estimate_tokens(merged.text),
        }
    )


def _is_eligible_tiny(
    chunk: ChunkDraft,
    *,
    min_tokens: int,
    min_path_depth: int,
    parent_counts: dict[tuple[str, ...], int],
    sibling_min_children: int,
) -> bool:
    """是否可以为了消除过小 chunk 而与相邻 chunk 合并。"""
    if chunk.section_type not in _MERGEABLE_SECTION_TYPES:
        return False
    if _is_grouped_chunk(chunk) or chunk.token_count >= min_tokens:
        return False
    path = chunk.heading_path
    if not path:
        return False
    deep = len(path) >= min_path_depth
    many_siblings = (
        parent_counts.get(tuple(path[:-1]), 0) >= sibling_min_children
    )
    return deep or many_siblings


def _absorb_tiny_neighbors(
    chunks: list[ChunkDraft],
    *,
    min_tokens: int,
    max_tokens: int,
    min_path_depth: int,
    parent_counts: dict[tuple[str, ...], int],
    sibling_min_children: int,
    chunking_version: str,
    chunk_id_namespace: str,
) -> list[ChunkDraft]:
    """最后兜底：把仍过小的 chunk 吸到相邻正文 chunk，避免碎片。"""
    result: list[ChunkDraft] = []
    pending: ChunkDraft | None = None
    for chunk in chunks:
        if pending is not None:
            combined = estimate_tokens(
                f"{_prefixed_text(pending)}\n{chunk.text}"
            )
            if (
                chunk.section_type in _MERGEABLE_SECTION_TYPES
                and not _is_grouped_chunk(chunk)
                and combined <= max_tokens
            ):
                merged = _merge_two(
                    pending.model_copy(update={"text": _prefixed_text(pending)}),
                    chunk,
                    chunking_version=chunking_version,
                    chunk_id_namespace=chunk_id_namespace,
                )
                merged = merged.model_copy(
                    update={
                        "heading_path": chunk.heading_path or merged.heading_path,
                        "heading": chunk.heading or merged.heading,
                        "embedding_text": (
                            f"{chunk.heading}\n{merged.text}"
                            if chunk.heading
                            else merged.text
                        ),
                        "token_count": estimate_tokens(merged.text),
                    }
                )
                result.append(merged)
                pending = None
                continue
            result.append(pending)
            pending = None
        eligible = _is_eligible_tiny(
            chunk,
            min_tokens=min_tokens,
            min_path_depth=min_path_depth,
            parent_counts=parent_counts,
            sibling_min_children=sibling_min_children,
        )
        if eligible and result:
            previous = result[-1]
            if (
                previous.section_type in _MERGEABLE_SECTION_TYPES
                and not _is_grouped_chunk(previous)
            ):
                merged_text = (
                    f"{previous.text.rstrip()}\n{_prefixed_text(chunk)}"
                )
                if estimate_tokens(merged_text) <= max_tokens:
                    merged = _merge_two(
                        previous,
                        chunk.model_copy(
                            update={"text": _prefixed_text(chunk)}
                        ),
                        chunking_version=chunking_version,
                        chunk_id_namespace=chunk_id_namespace,
                    )
                    metadata = dict(merged.metadata)
                    absorbed = list(metadata.get("absorbed_sections") or [])
                    if chunk.heading_path:
                        absorbed.append(chunk.heading_path[-1])
                    metadata["absorbed_sections"] = absorbed
                    result[-1] = merged.model_copy(
                        update={"metadata": metadata}
                    )
                    continue
        if eligible and not result:
            pending = chunk
            continue
        result.append(chunk)
    if pending is not None:
        result.append(pending)
    return result


def merge_small_sibling_chunks(
    chunks: list[ChunkDraft],
    *,
    min_tokens: int,
    max_tokens: int,
    min_path_depth: int,
    sibling_min_children: int = 10,
    chunking_version: str,
    chunk_id_namespace: str = "verity:chunk",
) -> list[ChunkDraft]:
    """把同一父标题下过小的相邻小节合并，避免 2.1/2.2 各成碎片。

    - 深度达到 ``min_path_depth`` 的小节可合并；父标题下小节数量
      达到 ``sibling_min_children`` 的长文档也允许合并浅层小节点，
      但病例报告这类一级章节很少的文档保持原粒度；
    - 只在其中一个 chunk 小于 ``min_tokens`` 时触发；
    - 合并后不超过 ``max_tokens``；
    - 标题路径回退到公共父标题，两个小节名写入 ``metadata.merged_sections``，
      并把小节名补进正文/embedding，保留最小粒度信息。
    """
    if not chunks:
        return []
    # 父标题下的小节数量：数量很多的长文档允许合并浅层小节点；
    # 病例报告这类只有几个一级章节的文档保持原粒度。
    parent_counts: dict[tuple[str, ...], int] = {}
    for item in chunks:
        if (
            item.section_type in _MERGEABLE_SECTION_TYPES
            and not _is_grouped_chunk(item)
            and item.heading_path
        ):
            key = tuple(item.heading_path[:-1])
            parent_counts[key] = parent_counts.get(key, 0) + 1
    merged: list[ChunkDraft] = []
    for chunk in chunks:
        if merged and _can_merge_siblings(
            merged[-1],
            chunk,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            min_path_depth=min_path_depth,
            parent_counts=parent_counts,
            sibling_min_children=sibling_min_children,
        ):
            merged[-1] = _merge_siblings(
                merged[-1],
                chunk,
                chunking_version=chunking_version,
                chunk_id_namespace=chunk_id_namespace,
            )
            continue
        merged.append(chunk)
    return _absorb_tiny_neighbors(
        merged,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        min_path_depth=min_path_depth,
        parent_counts=parent_counts,
        sibling_min_children=sibling_min_children,
        chunking_version=chunking_version,
        chunk_id_namespace=chunk_id_namespace,
    )
