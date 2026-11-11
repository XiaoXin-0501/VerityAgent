"""RAG 公共类型导出。

外部只应 import 这里的类型；Parser/Processor/Chunker/Store 等内部实现不导出。
"""

from __future__ import annotations

from config.rag import IndexerConfig, RetrieverConfig
from exception import RagError, RagErrorCode

from .shared.enums import (
    DocumentType,
    IndexMode,
    IndexingErrorCode,
    IndexingStage,
    IndexingStatus,
    IndexingWarningCode,
    RAGMode,
    RetrievalMode,
    RetrievalStage,
    RetrievalWarningCode,
)
from .shared.models import (
    Citation,
    DocumentMeta,
    DocumentSource,
    GraphExpansionPolicy,
    HybridPolicy,
    IndexDocumentRequest,
    IndexingError,
    IndexingJobView,
    IndexingWarning,
    NeighborPolicy,
    RagContext,
    RerankPolicy,
    RetrievalFilter,
    RetrievalRequest,
    RetrievalResult,
    RetrievalTrace,
    RetrievalWarning,
    RetrievedChunk,
)

__all__ = [
    # config
    "IndexerConfig",
    "RetrieverConfig",
    # errors
    "RagError",
    "RagErrorCode",
    # enums
    "DocumentType",
    "IndexMode",
    "IndexingStatus",
    "IndexingStage",
    "IndexingErrorCode",
    "IndexingWarningCode",
    "RetrievalMode",
    "RAGMode",
    "RetrievalStage",
    "RetrievalWarningCode",
    # models
    "DocumentSource",
    "IndexDocumentRequest",
    "IndexingJobView",
    "IndexingError",
    "IndexingWarning",
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
    "RetrievalTrace",
    "RetrievalWarning",
    "RetrievalResult",
]
