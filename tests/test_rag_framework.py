"""RAG 框架单元测试：DTO、确定性 id、Registry、RRF、Facade。"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from uuid import uuid4

_ROOT = Path(__file__).resolve().parents[1]
_AGENT_ROOT = _ROOT / "verity-agent"
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

from config.rag import IndexerConfig, RetrieverConfig  # noqa: E402
from exception import RagError, RagErrorCode  # noqa: E402
from rag.facade import (  # noqa: E402
    RagDependencies,
    create_indexer,
    create_retriever,
)
from rag.retriever.fusion import rrf_fuse  # noqa: E402
from rag.shared.enums import (  # noqa: E402
    DocumentType,
    IndexMode,
    IndexingStatus,
    RetrievalMode,
)
from rag.shared.ids import chunk_hash, deterministic_chunk_id  # noqa: E402
from rag.shared.models import (  # noqa: E402
    DocumentMeta,
    DocumentProfile,
    DocumentSource,
    IndexDocumentRequest,
    IndexingJobView,
    RagContext,
    RetrievalRequest,
    RetrievalResult,
    RetrievalTrace,
    RetrievedChunk,
)
from rag.shared.registry import ParserRegistry  # noqa: E402


def test_deterministic_chunk_id() -> None:
    kb_id, doc_id = uuid4(), uuid4()
    digest = chunk_hash("hello")
    first = deterministic_chunk_id(
        kb_id=kb_id,
        doc_id=doc_id,
        chunk_index=1,
        content_hash=digest,
        chunking_version="v1",
    )
    second = deterministic_chunk_id(
        kb_id=kb_id,
        doc_id=doc_id,
        chunk_index=1,
        content_hash=digest,
        chunking_version="v1",
    )
    different = deterministic_chunk_id(
        kb_id=kb_id,
        doc_id=doc_id,
        chunk_index=2,
        content_hash=digest,
        chunking_version="v1",
    )
    assert first == second
    assert first != different


def test_rrf_fusion() -> None:
    fused = rrf_fuse([["a", "b"], ["b", "c"]], key=lambda item: item)
    assert fused[0][0] == "b"
    assert {item for item, _ in fused} == {"a", "b", "c"}


def test_parser_registry_resolves_by_extension() -> None:
    registry = ParserRegistry()
    from rag.shared.models import ParsedDocument, ParserSpec

    class Parser:
        async def parse(self, source, profile) -> ParsedDocument:
            return ParsedDocument(
                doc_id=uuid4(),
                kb_id=uuid4(),
                parser_id="md",
                parser_version="1",
                profile=profile,
            )

    parser = Parser()
    registry.register(
        ParserSpec(
            parser_id="md",
            extensions={".md"},
            doc_types={DocumentType.MARKDOWN},
        ),
        parser,
    )
    source = DocumentSource(
        object_key="source/kb/doc.md",
        filename="doc.md",
        extension=".md",
        size_bytes=10,
        sha256="a" * 64,
    )
    spec, impl = registry.resolve(
        source, DocumentProfile(doc_type=DocumentType.MARKDOWN)
    )
    assert spec.parser_id == "md"
    assert impl is parser


def test_facade_requires_pipeline() -> None:
    async def run() -> None:
        try:
            await create_indexer(IndexerConfig())
        except RagError as exc:
            assert exc.error_code is RagErrorCode.DEPENDENCY_UNAVAILABLE
        else:
            raise AssertionError("missing pipeline should fail")

        try:
            await create_retriever(RetrieverConfig())
        except RagError as exc:
            assert exc.error_code is RagErrorCode.DEPENDENCY_UNAVAILABLE
        else:
            raise AssertionError("missing pipeline should fail")

    asyncio.run(run())


def test_facade_delegates_to_pipeline() -> None:
    async def run() -> None:
        kb_id, doc_id = uuid4(), uuid4()

        class IndexingPipeline:
            async def submit(self, request):
                return IndexingJobView(
                    job_id="job-1",
                    kb_id=request.kb_id,
                    doc_id=request.doc_id,
                    mode=request.mode,
                    status=IndexingStatus.QUEUED,
                    stage="ACCEPTED",
                )

            async def get_job(self, job_id):
                return None

            async def cancel(self, job_id):
                return None

            async def delete(self, doc_id):
                return IndexingJobView(
                    job_id="job-2",
                    kb_id=kb_id,
                    doc_id=doc_id,
                    mode=IndexMode.DELETE,
                    status=IndexingStatus.QUEUED,
                    stage="DELETING_PG",
                )

            async def close(self):
                return None

        class RetrievalPipeline:
            async def retrieve(self, request):
                chunk = RetrievedChunk(
                    chunk_id="c1",
                    doc_id=doc_id,
                    kb_id=kb_id,
                    content="result",
                )
                context = RagContext(
                    query=request.query,
                    kb_ids=request.kb_ids,
                    chunks=[chunk],
                    documents={},
                    top_k=request.top_k,
                )
                return RetrievalResult(
                    query=request.query,
                    normalized_query=request.query,
                    kb_ids=request.kb_ids,
                    mode=request.mode,
                    chunks=[chunk],
                    documents={},
                    context=context,
                    trace=RetrievalTrace(trace_id="trace"),
                )

            async def close(self):
                return None

        deps = RagDependencies(
            indexing_pipeline=IndexingPipeline(),
            retrieval_pipeline=RetrievalPipeline(),
        )
        indexer = await create_indexer(IndexerConfig(), dependencies=deps)
        job = await indexer.index(
            IndexDocumentRequest(
                kb_id=kb_id,
                doc_id=doc_id,
                mode=IndexMode.CREATE,
                source=DocumentSource(
                    object_key="source/kb/doc.md",
                    filename="doc.md",
                    extension=".md",
                    size_bytes=10,
                    sha256="a" * 64,
                ),
            )
        )
        assert job.job_id == "job-1"

        retriever = await create_retriever(RetrieverConfig(), dependencies=deps)
        result = await retriever.retrieve(
            RetrievalRequest(
                query="hello",
                kb_ids=[kb_id],
                user_id=uuid4(),
                permissions=frozenset(),
                mode=RetrievalMode.HYBRID,
            )
        )
        assert result.chunks[0].content == "result"

    asyncio.run(run())
