"""RAG 领域枚举。

与文档 §3–§9 对齐；存储层只存字符串值，枚举用于代码分支与校验。
"""

from __future__ import annotations

from enum import StrEnum

from common.enums import RAGMode

__all__ = [
    "DocumentType",
    "SUPPORTED_DOC_TYPES",
    "BlockType",
    "SectionType",
    "QualityFlag",
    "SymbolType",
    "CodeParseStatus",
    "CodeLanguage",
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
    """内容质量缺陷标记。

    标记只描述“哪里可能不可靠”，不决定是否丢弃；丢弃/合并由 QualityGate/
    DedupPolicy 决定，保留下来时由 `IndexerPolicy.quality.penalties` 折算成
    chunk.quality_score 供检索降权。
    """

    # OCR 置信度低于 quality.min_ocr_confidence：文字可能有错字/漏字
    LOW_OCR = "LOW_OCR"
    # 表格识别置信度低于 quality.min_table_confidence：单元格/行列可能错位
    LOW_TABLE = "LOW_TABLE"
    # 文档/页面文本层覆盖率低：扫描件、图片为主，chunk 可能只是原文一部分
    LOW_TEXT_COVERAGE = "LOW_TEXT_COVERAGE"
    # 与其它 chunk 重复/近重复：默认丢弃，保留时表示几乎没有新增信息
    DUPLICATE = "DUPLICATE"
    # 内容被真正截断（不是正常切片）：尾部可能缺失
    TRUNCATED = "TRUNCATED"
    # 页眉/页脚/导航/广告等模板化文本：默认丢弃
    BOILERPLATE = "BOILERPLATE"
    # 代码 AST 解析失败，退化为纯文本切分：正文完整但缺结构元数据
    CODE_PARSE_FALLBACK = "CODE_PARSE_FALLBACK"
    # embedding 输入超过模型上限被截断：只影响向量，正文仍完整
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


class CodeLanguage(StrEnum):
    """代码语言；扩展名 → 语言的映射集中在枚举旁边，不在 parser 内散落。"""

    PYTHON = "python"
    JAVA = "java"
    TYPESCRIPT = "typescript"
    TSX = "tsx"
    JAVASCRIPT = "javascript"
    JSX = "jsx"
    GO = "go"
    RUST = "rust"
    C = "c"
    CPP = "cpp"

    @classmethod
    def from_extension(cls, extension: str) -> "CodeLanguage | None":
        """按扩展名识别语言；未知返回 None（由调用方回退文本处理）。"""
        value = (extension or "").lower().strip()
        if value and not value.startswith("."):
            value = f".{value}"
        return _CODE_LANGUAGE_BY_EXT.get(value)


# 扩展名别名（.h 归 C、.hpp 归 C++、.tsx/.jsx 保留独立语言标记）。
_CODE_LANGUAGE_BY_EXT: dict[str, CodeLanguage] = {
    ".py": CodeLanguage.PYTHON,
    ".java": CodeLanguage.JAVA,
    ".ts": CodeLanguage.TYPESCRIPT,
    ".tsx": CodeLanguage.TSX,
    ".js": CodeLanguage.JAVASCRIPT,
    ".jsx": CodeLanguage.JSX,
    ".go": CodeLanguage.GO,
    ".rs": CodeLanguage.RUST,
    ".c": CodeLanguage.C,
    ".cpp": CodeLanguage.CPP,
    ".h": CodeLanguage.C,
    ".hpp": CodeLanguage.CPP,
}


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
    """检索阶段可降级告警；单个 KB/阶段失败不应中断整次检索。"""

    KB_PERMISSION_FILTERED = "KB_PERMISSION_FILTERED"
    NO_PERMISSION = "NO_PERMISSION"
    LOW_RECALL = "LOW_RECALL"
    QUERY_TOO_LONG = "QUERY_TOO_LONG"
    RERANK_DISABLED = "RERANK_DISABLED"
    RERANK_FAILED = "RERANK_FAILED"
    GRAPH_SKIPPED = "GRAPH_SKIPPED"
    GRAPH_FAILED = "GRAPH_FAILED"
    GRAPH_LOW_EVIDENCE = "GRAPH_LOW_EVIDENCE"
    DOC_STATUS_FILTERED = "DOC_STATUS_FILTERED"
    QUALITY_FILTERED = "QUALITY_FILTERED"
    NEIGHBOR_TRUNCATED = "NEIGHBOR_TRUNCATED"
    NEIGHBOR_BUDGET_EXCEEDED = "NEIGHBOR_BUDGET_EXCEEDED"
    NEIGHBOR_FAILED = "NEIGHBOR_FAILED"
    BUDGET_TRUNCATED = "BUDGET_TRUNCATED"
    INDEX_VERSION_MISMATCH = "INDEX_VERSION_MISMATCH"
    CACHE_MISS = "CACHE_MISS"
    CACHE_ERROR = "CACHE_ERROR"
    EMBEDDING_FAILED = "EMBEDDING_FAILED"
    DOC_META_MISSING = "DOC_META_MISSING"


class RetrievalErrorCode(StrEnum):
    """检索内部阶段错误码；对外统一映射为 ``RagError``。"""

    INVALID_REQUEST = "INVALID_REQUEST"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    EMBEDDING_FAILED = "EMBEDDING_FAILED"
    VECTOR_STORE_ERROR = "VECTOR_STORE_ERROR"
    GRAPH_STORE_ERROR = "GRAPH_STORE_ERROR"
    RERANK_ERROR = "RERANK_ERROR"
    DEPENDENCY_UNAVAILABLE = "DEPENDENCY_UNAVAILABLE"
    TIMEOUT = "TIMEOUT"
    INTERNAL = "INTERNAL"


class QueryIntent(StrEnum):
    LOOKUP = "lookup"
    COMPARE = "compare"
    SUMMARIZE = "summarize"
    HOWTO = "howto"
