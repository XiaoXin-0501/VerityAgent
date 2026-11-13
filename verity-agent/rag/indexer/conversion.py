"""ChunkDraft / ParsedDocument → 落库模型的转换。

放在 pipeline 之外，保证“切片结果”和“存储模型”解耦：
- ``drafts_to_chunks`` 计算 chunk_hash、质量分与 Weaviate 元数据；
- ``build_parse_manifest`` 汇总解析统计，写 MinIO 的 manifest 入口。
"""

from __future__ import annotations

from rag.shared.enums import QualityFlag
from rag.shared.ids import chunk_hash
from rag.shared.models import (
    Chunk,
    ChunkDraft,
    ParseManifest,
    ParsedDocument,
)
from utils.time import utcnow

from .policy import IndexerPolicy, get_indexer_policy

__all__ = ["drafts_to_chunks", "build_parse_manifest", "quality_score"]


def quality_score(
    flags: set[QualityFlag],
    *,
    policy: IndexerPolicy | None = None,
) -> float:
    """把质量标记折算成 0~1 分（1.0 最可信）。

    公式和扣分表由 :class:`IndexerPolicy` 管理；这里保留函数入口，
    兼容旧调用方与测试。策略文件见 ``config/rag_policy.json``。
    """
    return (policy or get_indexer_policy()).quality.score(flags)


def drafts_to_chunks(
    drafts: list[ChunkDraft],
    *,
    parsed: ParsedDocument,
    index_version: int,
    chunking_version: str,
    policy: IndexerPolicy | None = None,
) -> list[Chunk]:
    """把切片草稿转成可写 Weaviate 的 ``Chunk``。

    - ``chunk_hash`` 基于正文而不是 embedding_text，保证同内容可复现；
    - ``quality_score`` 由 ``draft.quality_flags`` 与索引策略叠加扣分得到，
      只做检索降权，不在这里丢弃内容；
    - ``semantic_description`` 只由表格/代码切片器写入，纯文本 chunk 为空；
    - ``doc_type/parser_id/chunking_version/index_version`` 来自本次入库；
    - 标题路径只保留在模型里（Weaviate v1 不写 heading，见 D-76）。
    """
    effective = policy or get_indexer_policy()
    created_at = utcnow()
    return [
        Chunk(
            chunk_id=draft.chunk_id,
            kb_id=draft.kb_id,
            doc_id=draft.doc_id,
            chunk_index=draft.chunk_index,
            section_type=draft.section_type,
            block_type=draft.block_type,
            page_no=draft.page_no,
            content=draft.text,
            token_count=draft.token_count,
            heading=draft.heading,
            heading_path=list(draft.heading_path),
            semantic_description=draft.semantic_description,
            language=draft.language,
            quality_score=effective.quality.score(draft.quality_flags),
            chunk_hash=chunk_hash(draft.text),
            doc_type=parsed.profile.doc_type.value,
            parser_id=parsed.parser_id,
            chunking_version=chunking_version,
            index_version=index_version,
            created_at=created_at,
        )
        for draft in drafts
    ]


def build_parse_manifest(
    parsed: ParsedDocument,
    *,
    source_sha256: str,
    token_count: int,
) -> ParseManifest:
    """汇总一次解析结果，生成 MinIO manifest 入口对象。

    blocks.jsonl 的 artifact 由 ``ManifestStore.write`` 在真正写对象后补登记，
    这样 size/key 与实际写入保持一致。
    """
    languages = list(
        dict.fromkeys(
            list(parsed.profile.languages or [])
            + [
                block.language
                for block in parsed.blocks
                if block.language
            ]
        )
    )
    return ParseManifest(
        doc_id=parsed.doc_id,
        kb_id=parsed.kb_id,
        parser_id=parsed.parser_id,
        parser_version=parsed.parser_version,
        source_sha256=source_sha256,
        profile=parsed.profile,
        page_count=len(parsed.pages) or (parsed.profile.page_count or 0),
        block_count=len(parsed.blocks),
        token_count=token_count,
        languages=languages,
        artifacts=list(parsed.artifacts),
        warnings=list(parsed.warnings),
    )
