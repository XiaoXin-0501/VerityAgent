"""通用切片打包器：把带来源信息的文本单元打包成 chunk 组。

单个块内的文本切片和跨块的章节切片共用同一套 token 预算/overlap 规则，
避免两套实现出现边界差异。``Unit`` 保留来源 block，便于合并后回填
``block_ids`` / 页码 / 质量标记。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from uuid import UUID

from rag.shared.enums import BlockType, QualityFlag, SectionType
from rag.shared.models import ChunkDraft
from utils.tokens import estimate_tokens, split_by_tokens

from .base import BaseChunker

__all__ = ["Unit", "pack_units", "build_group_chunk"]


@dataclass(slots=True)
class Unit:
    """一个最小切片单元；通常是一句话或一个列表项。"""

    text: str
    block_id: UUID | None = None
    page_no: int | None = None
    language: str | None = None
    quality_flags: set[QualityFlag] = field(default_factory=set)

    @property
    def token_count(self) -> int:
        """估算该单元 token 数。"""
        return estimate_tokens(self.text)


def _tail(units: list[Unit], overlap_tokens: int) -> list[Unit]:
    """取尾部单元作为下一组的 overlap；按整单元保留，避免拆句。"""
    if overlap_tokens <= 0:
        return []
    tail: list[Unit] = []
    tokens = 0
    for unit in reversed(units):
        if tokens + unit.token_count > overlap_tokens:
            break
        tail.insert(0, unit)
        tokens += unit.token_count
    return tail


def _split_unit(unit: Unit, max_tokens: int) -> list[Unit]:
    """把超长单元按 token 上限切成多个同源单元。"""
    pieces = split_by_tokens(unit.text, max_tokens)
    return [
        Unit(
            text=piece.strip(),
            block_id=unit.block_id,
            page_no=unit.page_no,
            language=unit.language,
            quality_flags=set(unit.quality_flags),
        )
        for piece in pieces
        if piece.strip()
    ]


def pack_units(
    units: list[Unit],
    *,
    target_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
) -> list[list[Unit]]:
    """按 target/max 把单元打包成若干组；超长单元会先被硬切。

    - 达到 ``target_tokens`` 时输出；
    - 加入下一单元会超过 ``max_tokens`` 时输出；
    - 输出后保留尾部 ``overlap_tokens`` 作为下一组前缀；
    - 结束时不会重复输出只含 overlap 的尾组。
    """
    if not units:
        return []

    # overlap 不应接近/超过 target，否则每次输出后尾部本身又接近 target，
    # 会连续产生高度重复的 chunk。默认 512/80 不受影响。
    effective_overlap = min(overlap_tokens, max(0, target_tokens // 4))
    groups: list[list[Unit]] = []
    current: list[Unit] = []
    current_tokens = 0
    pending = False

    def emit() -> None:
        """输出当前组，并保留 overlap 尾部。"""
        nonlocal current, current_tokens, pending
        if current:
            groups.append(list(current))
        current = _tail(current, effective_overlap)
        # 单元之间至少有一个分隔符；按每单元 1 token 保守计入硬上限。
        current_tokens = estimate_tokens("".join(unit.text for unit in current)) + max(
            0, len(current) - 1
        )
        pending = False

    for unit in units:
        if unit.token_count > max_tokens:
            if current:
                emit()
            current = []
            current_tokens = 0
            for piece in _split_unit(unit, max_tokens):
                groups.append([piece])
            continue

        if current and current_tokens + unit.token_count > max_tokens:
            emit()
            if current_tokens + unit.token_count > max_tokens:
                # overlap 尾部 + 当前单元仍超限时，放弃重叠以保证硬上限。
                current = []
                current_tokens = 0
        current.append(unit)
        # 第一个单元没有分隔符；后续每个单元保守按 1 token 计分隔符开销。
        current_tokens += unit.token_count + (1 if len(current) > 1 else 0)
        pending = True
        if current_tokens >= target_tokens:
            emit()

    if current and pending:
        emit()
    return groups


def _unique_block_ids(units: list[Unit]) -> list[UUID]:
    """按首次出现顺序返回去重后的来源 block id；无来源的单元跳过。"""
    seen: set[UUID] = set()
    result: list[UUID] = []
    for unit in units:
        if unit.block_id is None or unit.block_id in seen:
            continue
        seen.add(unit.block_id)
        result.append(unit.block_id)
    return result


def build_group_chunk(
    chunker: BaseChunker,
    units: list[Unit],
    *,
    doc_id: UUID,
    kb_id: UUID,
    index: int,
    block_type: BlockType,
    section_type: SectionType,
    heading_text: str | None,
    heading_path: list[str],
    joiner: Callable[[list[str]], str],
    document_description: str | None = None,
) -> ChunkDraft:
    """把一组单元构造为 ``ChunkDraft``（正文与 embedding 分离）。

    ``document_description`` 只写入 metadata 供展示/审计，不拼进
    ``embedding_text``：同一文档所有 chunk 共享描述会让向量趋同，稀释检索
    信号；文档描述按 D-103 在检索结果中按 ``doc_id`` 单独注入。
    """
    text = joiner([unit.text for unit in units])
    if not text:
        text = ""
    flags: set[QualityFlag] = set()
    for unit in units:
        flags.update(unit.quality_flags)
    page_no = next((unit.page_no for unit in units if unit.page_no), None)
    language = next((unit.language for unit in units if unit.language), None)
    embedding_text = f"{heading_text}\n{text}" if heading_text else text
    metadata = (
        {"document_description": document_description}
        if document_description
        else {}
    )
    return chunker.build_chunk_from_text(
        doc_id=doc_id,
        kb_id=kb_id,
        block_type=block_type,
        section_type=section_type,
        index=index,
        text=text,
        embedding_text=embedding_text,
        block_ids=_unique_block_ids(units),
        page_no=page_no,
        heading=heading_text,
        heading_path=heading_path,
        language=language,
        quality_flags=flags,
        metadata=metadata,
    )
