"""RAG 领域枚举。

与文档 §3–§9 对齐；存储层只存字符串值，枚举用于代码分支与校验。
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "DocumentType",
    "SUPPORTED_DOC_TYPES",
    "BlockType",
    "SectionType",
    "QualityFlag",
    "SymbolType",
    "CodeParseStatus",
    "SourceType",
    "IndexMode",
    "IndexingStatus",
    "IndexingStage",
    "IndexingErrorCode",
    "IndexingWarningCode",
    "RetrievalMode",
    "RAGMode",
    "RetrievalStage",
    "RetrievalWarningCode",
    "RetrievalErrorCode",
    "QueryIntent",
]


class DocumentType(StrEnum):
    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    MARKDOWN = "markdown"
    HTML = "html"
    TEXT = "text"
    SPREADSHEET = "spreadsheet"
    CSV = "csv"
    CODE = "code"
    IMAGE = "image"
    UNKNOWN = "unknown"


SUPPORTED_DOC_TYPES: frozenset[DocumentType] = frozenset(
    {
        DocumentType.PDF,
        DocumentType.DOCX,
        DocumentType.MARKDOWN,
        DocumentType.TEXT,
        DocumentType.SPREADSHEET,
    }
)


class BlockType(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    CODE = "code"
    FIGURE = "figure"
    CAPTION = "caption"
    FORMULA = "formula"
    REFERENCE = "reference"
    FOOTNOTE = "footnote"
    HEADER = "header"
    FOOTER = "footer"
    BOILERPLATE = "boilerplate"
    TOC = "toc"
    METADATA = "metadata"


class SectionType(StrEnum):
    PARAGRAPH = "paragraph"
    TABLE = "table"
    CODE = "code"
    FIGURE = "figure"
    LIST = "list"
    FORMULA = "formula"
    REFERENCE = "reference"


class QualityFlag(StrEnum):
    LOW_OCR = "LOW_OCR"
    LOW_TABLE = "LOW_TABLE"
    LOW_TEXT_COVERAGE = "LOW_TEXT_COVERAGE"
    DUPLICATE = "DUPLICATE"
    TRUNCATED = "TRUNCATED"
    BOILERPLATE = "BOILERPLATE"
    CODE_PARSE_FALLBACK = "CODE_PARSE_FALLBACK"
    EMBEDDING_TRUNCATED = "EMBEDDING_TRUNCATED"


class SymbolType(StrEnum):
    FILE = "file"
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    BLOCK = "block"


class CodeParseStatus(StrEnum):
    OK = "ok"
    PARTIAL = "partial"
    FALLBACK = "fallback"


class SourceType(StrEnum):
    UPLOAD = "upload"
    API = "api"
    MANUAL = "manual"
    URL = "url"


class IndexMode(StrEnum):
    CREATE = "create"
    UPDATE = "update"
    REINDEX = "reindex"
    DELETE = "delete"


class IndexingStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DEAD = "dead"


class IndexingStage(StrEnum):
    ACCEPTED = "ACCEPTED"
    FETCHING = "FETCHING"
    PARSING = "PARSING"
    NORMALIZING = "NORMALIZING"
    CLASSIFYING = "CLASSIFYING"
    PROCESSING = "PROCESSING"
    CHUNKING = "CHUNKING"
    DEDUPING = "DEDUPING"
    ENRICHING = "ENRICHING"
    DESCRIBING = "DESCRIBING"
    EMBEDDING = "EMBEDDING"
    VECTOR_UPSERT = "VECTOR_UPSERT"
    GRAPH_UPSERT = "GRAPH_UPSERT"
    COMMITTING = "COMMITTING"
    DONE = "DONE"
    DELETING_PG = "DELETING_PG"
    DELETING_VECTOR = "DELETING_VECTOR"
    DELETING_GRAPH = "DELETING_GRAPH"
    DELETING_OBJECT = "DELETING_OBJECT"


class IndexingErrorCode(StrEnum):
    UNSUPPORTED_TYPE = "UNSUPPORTED_TYPE"
    CORRUPT_FILE = "CORRUPT_FILE"
    ENCRYPTED = "ENCRYPTED"
    EMPTY_CONTENT = "EMPTY_CONTENT"
    PARSE_FAILED = "PARSE_FAILED"
    LOW_QUALITY = "LOW_QUALITY"
    EMBEDDING_FAILED = "EMBEDDING_FAILED"
    VECTOR_WRITE_FAILED = "VECTOR_WRITE_FAILED"
    GRAPH_WRITE_FAILED = "GRAPH_WRITE_FAILED"
    COMMIT_FAILED = "COMMIT_FAILED"
    CANCELLED = "CANCELLED"
    DEAD = "DEAD"
    TIMEOUT = "TIMEOUT"
    INTERNAL = "INTERNAL"


class IndexingWarningCode(StrEnum):
    LOW_TEXT_COVERAGE = "LOW_TEXT_COVERAGE"
    LOW_OCR = "LOW_OCR"
    LOW_TABLE = "LOW_TABLE"
    LOW_CLASSIFICATION = "LOW_CLASSIFICATION"
    CODE_PARSE_FALLBACK = "CODE_PARSE_FALLBACK"
    DUPLICATE_DROPPED = "DUPLICATE_DROPPED"
    BOILERPLATE_DROPPED = "BOILERPLATE_DROPPED"
    CHUNK_TOO_SHORT = "CHUNK_TOO_SHORT"
    CHUNK_TRUNCATED = "CHUNK_TRUNCATED"
    DESCRIPTION_FALLBACK = "DESCRIPTION_FALLBACK"
    GRAPH_SKIPPED = "GRAPH_SKIPPED"
    PARSE_FALLBACK = "PARSE_FALLBACK"


class RetrievalMode(StrEnum):
    HYBRID = "hybrid"
    BM25 = "bm25"
    VECTOR = "vector"


class RAGMode(StrEnum):
    TOOL = "tool"
    INJECTED = "injected"
    OFF = "off"


class RetrievalStage(StrEnum):
    PERMISSION = "PERMISSION"
    ANALYZE = "ANALYZE"
    QUERY_EMBED = "QUERY_EMBED"
    BM25 = "BM25"
    VECTOR = "VECTOR"
    GRAPH = "GRAPH"
    FUSION = "FUSION"
    RERANK = "RERANK"
    DOC_STATUS = "DOC_STATUS"
    DOC_META = "DOC_META"
    FILTER = "FILTER"
    NEIGHBOR = "NEIGHBOR"
    BUDGET = "BUDGET"


class RetrievalWarningCode(StrEnum):
    KB_PERMISSION_DENIED = "KB_PERMISSION_DENIED"
    RERANK_FAILED = "RERANK_FAILED"
    GRAPH_FAILED = "GRAPH_FAILED"
    GRAPH_SKIPPED = "GRAPH_SKIPPED"
    NEIGHBOR_BUDGET_EXCEEDED = "NEIGHBOR_BUDGET_EXCEEDED"
    BUDGET_TRUNCATED = "BUDGET_TRUNCATED"
    CACHE_ERROR = "CACHE_ERROR"
    DOC_META_MISSING = "DOC_META_MISSING"


class RetrievalErrorCode(StrEnum):
    INVALID_REQUEST = "INVALID_REQUEST"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    DEPENDENCY_UNAVAILABLE = "DEPENDENCY_UNAVAILABLE"
    INTERNAL = "INTERNAL"


class QueryIntent(StrEnum):
    LOOKUP = "lookup"
    COMPARE = "compare"
    SUMMARIZE = "summarize"
    HOWTO = "howto"
