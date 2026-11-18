"""RAG 领域模型（公共 DTO + 内部流水线模型）。

公共门面只导出 `rag.public` 中的类型；内部模型仍放在本模块，避免业务层
直接依赖 parser/chunker/store 实现。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from provider.embedding.types import EmbeddingUsage
from utils.time import utcnow

from .enums import (
    BlockType,
    CodeParseStatus,
    DocumentType,
    IndexingErrorCode,
    IndexingStage,
    IndexingStatus,
    IndexingWarningCode,
    IndexMode,
    QualityFlag,
    RAGMode,
    RetrievalMode,
    RetrievalStage,
    RetrievalWarningCode,
    SectionType,
    SourceType,
    SymbolType,
)

__all__ = [
    "DocumentSource",
    "IndexDocumentRequest",
    "DocumentProfile",
    "ParserSpec",
    "ArtifactRef",
    "TableBlock",
    "CodeBlock",
    "FigureBlock",
    "FormulaBlock",
    "ContentBlock",
    "Page",
    "ParsedDocument",
    "ParseManifest",
    "ProcessedBlock",
    "ChunkingSpec",
    "ChunkDraft",
    "Chunk",
    "NoiseReport",
    "DocumentDescriptionCandidate",
    "ChunkEnrichment",
    "GraphKeyword",
    "GraphRelation",
    "ChunkGraphExtraction",
    "EmbeddingInput",
    "EmbeddingBatch",
    "IndexCommit",
    "IndexingError",
    "IndexingWarning",
    "IndexingCounts",
    "IndexingJob",
    "IndexingJobView",
    "RetrievalFilter",
    "HybridPolicy",
    "RerankPolicy",
    "GraphExpansionPolicy",
    "NeighborPolicy",
    "RetrievalRequest",
    "RetrievedChunk",
    "DocumentMeta",
    "Citation",
    "RagContext",
    "RetrievalStageTrace",
    "RetrievalTrace",
    "RetrievalWarning",
    "RetrievalResult",
    "QueryAnalysis",
]


class _Model(BaseModel):
    """RAG 模型基类：拒绝未知字段，默认不可变，字段名即序列化名。"""

    model_config = ConfigDict(
        extra="forbid", frozen=True, populate_by_name=True
    )


# ——————————————————— 文档接入 ———————————————————

class DocumentSource(_Model):
    object_key: str
    filename: str
    extension: str
    mime_type: str = "application/octet-stream"
    size_bytes: int = Field(ge=0)
    sha256: str
    language: str | None = None
    source_type: SourceType = SourceType.UPLOAD
    uploaded_by: UUID | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class IndexDocumentRequest(_Model):
    kb_id: UUID
    doc_id: UUID
    source: DocumentSource
    mode: IndexMode = IndexMode.CREATE
    parser_hint: str | None = None
    chunking_profile: str | None = None
    graph_build_enabled: bool | None = None
    description_override: str | None = None
    priority: int = 100
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentProfile(_Model):
    doc_type: DocumentType = DocumentType.UNKNOWN
    mime_type: str | None = None
    extension: str | None = None
    page_count: int | None = None
    has_text_layer: bool = True
    is_scanned: bool = False
    has_tables: bool = False
    has_code: bool = False
    has_images: bool = False
    has_formulas: bool = False
    languages: list[str] = Field(default_factory=list)
    text_coverage: float = Field(default=1.0, ge=0.0, le=1.0)
    ocr_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    quality_score: float = Field(default=1.0, ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)


class ParserSpec(_Model):
    parser_id: str
    version: str = "1"
    doc_types: set[DocumentType] = Field(default_factory=set)
    extensions: set[str] = Field(default_factory=set)
    mime_types: set[str] = Field(default_factory=set)
    priority: int = 100
    supports_scanned: bool = False
    supports_tables: bool = False
    supports_code: bool = False
    options: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


class ArtifactRef(_Model):
    kind: str
    object_key: str
    size_bytes: int = 0
    content_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ——————————————————— 解析块 ———————————————————

class TableBlock(_Model):
    caption: str | None = None
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    merged_cells: list[dict[str, Any]] = Field(default_factory=list)
    markdown: str | None = None
    html: str | None = None
    confidence: float | None = None
    summary: str | None = None


class CodeBlock(_Model):
    language: str | None = None
    file_path: str | None = None
    symbol_path: str | None = None
    symbol_type: SymbolType = SymbolType.BLOCK
    signature: str | None = None
    docstring: str | None = None
    imports: list[str] = Field(default_factory=list)
    body: str = ""
    start_line: int | None = None
    end_line: int | None = None
    parse_status: CodeParseStatus = CodeParseStatus.OK


class FigureBlock(_Model):
    image_ref: str
    caption: str | None = None
    alt_text: str | None = None
    ocr_text: str | None = None
    ocr_confidence: float | None = None
    bbox: list[float] | None = None


class FormulaBlock(_Model):
    latex: str
    text: str | None = None
    confidence: float | None = None


class ContentBlock(_Model):
    block_id: UUID
    doc_id: UUID
    page_no: int = 0
    order: int
    block_type: BlockType
    text: str = ""
    html: str | None = None
    table: TableBlock | None = None
    code: CodeBlock | None = None
    figure: FigureBlock | None = None
    formula: FormulaBlock | None = None
    caption: str | None = None
    heading_level: int | None = None
    heading_text: str | None = None
    # 从一级标题到当前块所在标题的完整路径；``heading_text`` 是其拼接展示形式。
    heading_path: list[str] = Field(default_factory=list)
    bbox: list[float] | None = None
    confidence: float | None = None
    language: str | None = None
    source_parser: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class Page(_Model):
    page_no: int = 0
    width: float | None = None
    height: float | None = None
    rotation: int | None = None
    text: str = ""
    blocks: list[ContentBlock] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class DocumentDescriptionCandidate(_Model):
    """文档描述候选：解析/规则/LLM 产出的文档级描述。"""

    doc_id: UUID
    description: str | None = None
    source: str = "auto"
    confidence: float = 0.0
    # 未直接采用的候选文本（供 LLM/人工复核）。
    candidate: str | None = None
    warnings: list[str] = Field(default_factory=list)


class ParsedDocument(_Model):
    doc_id: UUID
    kb_id: UUID
    parser_id: str
    parser_version: str
    profile: DocumentProfile = Field(default_factory=DocumentProfile)
    pages: list[Page] = Field(default_factory=list)
    blocks: list[ContentBlock] = Field(default_factory=list)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    # 文档级描述；由 description 模块在解析后填充，供入库/PG documents.description 使用。
    description: DocumentDescriptionCandidate | None = None
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)


class ParseManifest(_Model):
    version: int = 1
    doc_id: UUID
    kb_id: UUID
    parser_id: str
    parser_version: str
    source_sha256: str
    profile: DocumentProfile
    page_count: int = 0
    block_count: int = 0
    token_count: int = 0
    languages: list[str] = Field(default_factory=list)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)


class ProcessedBlock(_Model):
    block: ContentBlock
    processor_id: str | None = None
    text: str = ""
    embedding_text: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    quality_flags: set[QualityFlag] = Field(default_factory=set)


# ——————————————————— 切片 ———————————————————

class ChunkingSpec(_Model):
    strategy_id: str
    version: str = "1"
    doc_types: set[DocumentType] = Field(default_factory=set)
    block_types: set[BlockType] = Field(default_factory=set)
    target_tokens: int = 300
    max_tokens: int = 600
    min_tokens: int = 40
    overlap_tokens: int = 80
    respect_block_boundaries: bool = True
    repeat_table_header: bool = True
    include_heading_in_embedding: bool = True
    max_chunks_per_block: int | None = None
    # 同 doc_type/block_type 多策略时的优先级，越小越优先
    priority: int = 100
    enabled: bool = True


class ChunkDraft(_Model):
    chunk_id: str
    doc_id: UUID
    kb_id: UUID
    chunk_index: int
    block_ids: list[UUID] = Field(default_factory=list)
    block_type: BlockType
    section_type: SectionType
    text: str
    embedding_text: str
    token_count: int = 0
    page_no: int | None = None
    heading: str | None = None
    heading_path: list[str] = Field(default_factory=list)
    # 特殊块（表格/代码/Excel）的语义描述；用于 heading+描述向量化，
    # 纯文本 chunk 为空。由切片阶段写入，落 Weaviate 便于观察/管理。
    semantic_description: str | None = None
    language: str | None = None
    quality_flags: set[QualityFlag] = Field(default_factory=set)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(_Model):
    chunk_id: str
    kb_id: UUID
    doc_id: UUID
    chunk_index: int
    section_type: SectionType
    block_type: BlockType
    page_no: int | None = None
    content: str
    token_count: int = 0
    # chunker 提供的标题展示串（可能是完整路径拼接）；落 Weaviate 时
    # DocChunk.heading 取 heading_path 末级，完整路径写 heading_path
    heading: str | None = None
    # 一级标题到当前标题的完整路径；写入 Weaviate DocChunk.heading_path
    heading_path: list[str] = Field(default_factory=list)
    # 特殊块语义描述；写入 Weaviate DocChunk.semantic_description
    semantic_description: str | None = None
    language: str | None = None
    quality_score: float = Field(default=1.0, ge=0.0, le=1.0)
    chunk_hash: str
    # 落 Weaviate DocChunk 的元数据
    doc_type: str | None = None
    parser_id: str | None = None
    chunking_version: str | None = None
    index_version: int | None = None
    created_at: datetime = Field(default_factory=utcnow)


# ——————————————————— 质量 / 富化 ———————————————————

class NoiseReport(_Model):
    # 质量闸门会逐块累计统计，允许在本地报告对象上赋值。
    model_config = ConfigDict(
        extra="forbid", frozen=False, populate_by_name=True
    )

    dropped_blocks: int = 0
    dropped_chunks: int = 0
    duplicate_chunks: int = 0
    low_quality_chunks: int = 0
    boilerplate_blocks: int = 0
    warnings: list[IndexingWarningCode] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class ChunkEnrichment(_Model):
    chunk_id: str
    keywords: list[str] = Field(default_factory=list)
    table_summary: str | None = None
    code_summary: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphKeyword(_Model):
    """图关键词节点；``type`` 来自领域画像的实体类型清单。"""

    name: str
    type: str = ""


class GraphRelation(_Model):
    """图关系三元组；端点类型仅用于写 Neo4j Keyword.type。"""

    subject: str
    relation: str
    object: str
    confidence: float | None = None
    subject_type: str | None = None
    object_type: str | None = None


class ChunkGraphExtraction(_Model):
    """单个 chunk 的图抽取产物；关键词必须是带类型的领域实体。"""

    chunk_id: str
    keywords: list[GraphKeyword] = Field(default_factory=list)
    relations: list[GraphRelation] = Field(default_factory=list)
    # 抽取来源模型；便于审计与缓存失效排查
    model: str | None = None
    # 整体置信度；空结果时为 None
    confidence: float | None = None


class EmbeddingInput(_Model):
    chunk_id: str
    text: str


class EmbeddingBatch(_Model):
    model_id: str
    dimensions: int
    inputs: list[EmbeddingInput] = Field(default_factory=list)
    vectors: list[list[float]] = Field(default_factory=list)
    usage: EmbeddingUsage | None = None
    batch_no: int = 0
    batch_total: int = 1


class IndexCommit(_Model):
    job_id: str
    doc_id: UUID
    kb_id: UUID
    new_chunk_ids: list[str] = Field(default_factory=list)
    stale_chunk_ids: list[str] = Field(default_factory=list)
    deleted_chunk_ids: list[str] = Field(default_factory=list)
    index_version: int = 0
    chunk_count: int = 0
    token_count: int = 0
    committed_at: datetime = Field(default_factory=utcnow)


# ——————————————————— 入库任务 ———————————————————

class IndexingError(_Model):
    code: IndexingErrorCode
    stage: IndexingStage
    message: str
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class IndexingWarning(_Model):
    code: IndexingWarningCode
    stage: IndexingStage
    message: str
    user_visible: bool = False


class IndexingCounts(_Model):
    pages: int = 0
    blocks: int = 0
    chunks: int = 0
    dropped_blocks: int = 0
    dropped_chunks: int = 0
    tokens: int = 0
    low_quality_blocks: int = 0


class IndexingJob(_Model):
    job_id: str
    kb_id: UUID
    doc_id: UUID
    mode: IndexMode
    status: IndexingStatus = IndexingStatus.QUEUED
    stage: IndexingStage = IndexingStage.ACCEPTED
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    attempts: int = 0
    max_attempts: int = 5
    error: IndexingError | None = None
    warnings: list[IndexingWarning] = Field(default_factory=list)
    locked_until: datetime | None = None
    locked_by: str | None = None
    next_retry_at: datetime | None = None
    index_version_before: int = 0
    index_version_after: int = 0
    counts: IndexingCounts = Field(default_factory=IndexingCounts)
    created_at: datetime = Field(default_factory=utcnow)
    started_at: datetime | None = None
    ended_at: datetime | None = None


class IndexingJobView(_Model):
    """公共投影，去掉锁/内部字段。"""

    job_id: str
    kb_id: UUID
    doc_id: UUID
    mode: IndexMode
    status: IndexingStatus
    stage: IndexingStage
    progress: float = 0.0
    attempts: int = 0
    error: IndexingError | None = None
    warnings: list[IndexingWarning] = Field(default_factory=list)
    index_version_before: int = 0
    index_version_after: int = 0
    counts: IndexingCounts = Field(default_factory=IndexingCounts)
    created_at: datetime = Field(default_factory=utcnow)
    started_at: datetime | None = None
    ended_at: datetime | None = None


# ——————————————————— 检索 ———————————————————

class RetrievalFilter(_Model):
    doc_ids: list[UUID] = Field(default_factory=list)
    doc_types: list[DocumentType] = Field(default_factory=list)
    section_types: list[SectionType] = Field(default_factory=list)
    block_types: list[BlockType] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    page_min: int | None = None
    page_max: int | None = None
    created_after: datetime | None = None
    tags: list[str] = Field(default_factory=list)


class HybridPolicy(_Model):
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    top_k_bm25: int = 20
    top_k_vector: int = 20
    fusion: str = "rrf"
    rrf_k: int = 60


class RerankPolicy(_Model):
    enabled: bool = True
    model_id: str | None = None
    top_n: int = 20
    min_score: float | None = None
    batch_size: int = 16


class GraphExpansionPolicy(_Model):
    enabled: bool = False
    max_hops: int = 1
    allowed_relations: list[str] = Field(default_factory=list)
    min_evidence: int = 2
    max_next_hops: int = 5
    hub_as_intermediate: bool = False
    min_rerank_score: float | None = None


class NeighborPolicy(_Model):
    enabled: bool = True
    before: int = 1
    after: int = 1
    max_tokens: int = 1000
    same_doc: bool = True
    same_block_type: bool = True


class RetrievalRequest(_Model):
    # 主召回 query；INJECTED 模式是用户原话，工具模式是 Agent 选择的检索 query
    query: str = Field(min_length=1)
    # 用户原话；Agent 查询规划产出的展示/审计字段，不参与召回
    original_query: str | None = None
    # Agent 查询规划的改写 query；非空时与 query 一起召回
    rewritten_query: str | None = None
    # Agent 查询规划的 RAG 召回意图；rerank 使用，不参与 BM25/向量召回
    rag_intent: str | None = None
    kb_ids: list[UUID]
    user_id: UUID
    permissions: frozenset[str] = frozenset()
    # 为空时使用 RetrieverConfig.top_k；显式传值覆盖配置默认
    top_k: int | None = Field(default=None, ge=1)
    mode: RetrievalMode = RetrievalMode.HYBRID
    filters: RetrievalFilter | None = None
    hybrid: HybridPolicy | None = None
    rerank: RerankPolicy | None = None
    graph: GraphExpansionPolicy | None = None
    neighbors: NeighborPolicy | None = None
    min_score: float | None = None
    min_quality_score: float | None = None
    token_budget: int | None = None
    trace_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(_Model):
    chunk_id: str
    doc_id: UUID
    kb_id: UUID
    doc_name: str | None = None
    chunk_index: int | None = None
    section_type: SectionType | None = None
    block_type: BlockType | None = None
    page_no: int | None = None
    content: str
    score: float | None = None
    bm25_score: float | None = None
    vector_score: float | None = None
    rerank_score: float | None = None
    heading: str | None = None
    heading_path: list[str] = Field(default_factory=list)
    # 特殊块（表格/代码）的语义描述；纯文本 chunk 为空
    semantic_description: str | None = None
    language: str | None = None
    quality_score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    citation: "Citation | None" = None


class DocumentMeta(_Model):
    doc_id: UUID
    kb_id: UUID
    doc_name: str
    description: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    description_missing: bool = False


class Citation(_Model):
    chunk_id: str
    doc_id: UUID
    kb_id: UUID
    doc_name: str | None = None
    page_no: int | None = None
    section_type: str | None = None
    block_type: str | None = None
    chunk_index: int | None = None
    snippet: str
    source_url: str | None = None


class RagContext(_Model):
    query: str
    kb_ids: list[UUID]
    chunks: list[RetrievedChunk]
    documents: dict[str, DocumentMeta] = Field(default_factory=dict)
    top_k: int
    duration_ms: int | None = None
    mode: RAGMode = RAGMode.INJECTED


class RetrievalStageTrace(_Model):
    stage: RetrievalStage
    input_count: int = 0
    output_count: int = 0
    duration_ms: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalTrace(_Model):
    trace_id: str
    stages: list[RetrievalStageTrace] = Field(default_factory=list)
    candidate_count: int = 0
    final_count: int = 0
    timings_ms: dict[str, int] = Field(default_factory=dict)
    alpha: float | None = None
    thresholds: dict[str, float] = Field(default_factory=dict)
    index_version: int | None = None
    cache_hit: bool = False
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)


class RetrievalWarning(_Model):
    code: RetrievalWarningCode
    message: str
    stage: RetrievalStage | None = None
    user_visible: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(_Model):
    query: str
    # 用户原话 / Agent 改写 / 召回意图；未提供时为空
    original_query: str | None = None
    rewritten_query: str | None = None
    rag_intent: str | None = None
    normalized_query: str
    kb_ids: list[UUID]
    mode: RetrievalMode
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    documents: dict[str, DocumentMeta] = Field(default_factory=dict)
    context: RagContext
    citations: list[Citation] = Field(default_factory=list)
    trace: RetrievalTrace
    warnings: list[RetrievalWarning] = Field(default_factory=list)
    duration_ms: int = 0


class QueryAnalysis(_Model):
    normalized_query: str
    language: str | None = None
    keywords: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    # 召回意图标签（如 LOOKUP/COMPARE）；由 Agent 查询规划注入
    intent: str | None = None
    suggested_filters: RetrievalFilter | None = None
    # Agent 查询规划产出的改写 query；未启用/不需要 RAG 时为空
    variants: list[str] = Field(default_factory=list)
