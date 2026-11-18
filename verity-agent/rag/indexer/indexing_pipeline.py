"""默认入库流水线：解析 → 切片 → 向量化 → Weaviate → Neo4j → PG 提交。

职责边界：

- 复用 ``TextIngestionPipeline`` 做解析/切片/描述提取，不重复实现 parser；
- 通过 ``VectorChunkStore / GraphIndexStore / ManifestStore / SourceStore``
  访问外部存储，只依赖协议；
- 通过 ``IndexingRepository`` 写 PG 状态，不直接拼 SQL；
- 任务状态机与阶段见《RAG 模块数据结构设计》§8.7。

v1 的“进程内后台任务”只是默认接线：生产可以关闭 ``run_in_background``，
由 PG ``claim_index_jobs`` 的 worker 调用 ``submit``/``execute``。所有写入
使用确定性 chunk_id，重试幂等。
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from config.rag import IndexerConfig
from exception import (
    MinioObjectNotFoundError,
    RagError,
    RagErrorCode,
)
from rag.shared.enums import (
    IndexingErrorCode,
    IndexingStage,
    IndexingStatus,
    IndexingWarningCode,
    IndexMode,
)
from rag.shared.events import (
    RagEvent,
    RagEventName,
    RagEventSink,
    emit_rag_event,
)
from rag.shared.models import (
    DocumentSource,
    IndexDocumentRequest,
    IndexingError,
    IndexingJob,
    IndexingJobView,
    IndexingWarning,
)
from store.pg.models.knowledge import Document
from utils.hashing import sha256_bytes
from utils.log_context import bind_index_context, reset_index_context
from utils.serialization import json_dumps
from utils.time import utcnow

from .conversion import build_parse_manifest, drafts_to_chunks
from .embedding import EmbeddingResult, EmbeddingService
from .enrichment import (
    GraphExtractor,
    build_graph_extractor,
    load_domain_profile,
)
from .job import InMemoryJobStore, JobStore, TERMINAL_STATUSES, to_job_view
from .repository import IndexingRepository
from .stores import (
    GraphIndexStore,
    ManifestStore,
    SourceStore,
    VectorChunkStore,
)
from .text_pipeline import TextIngestionPipeline

__all__ = ["DefaultIndexingPipeline", "STAGE_PROGRESS"]

logger = logging.getLogger(__name__)

# 入库/删除阶段进度（内置表）；只用于 job view / PG 投影。
STAGE_PROGRESS: dict[IndexingStage, float] = {
    IndexingStage.FETCHING: 0.05,
    IndexingStage.PARSING: 0.15,
    IndexingStage.CHUNKING: 0.3,
    IndexingStage.DEDUPING: 0.35,
    IndexingStage.DESCRIBING: 0.4,
    IndexingStage.ENRICHING: 0.45,
    IndexingStage.EMBEDDING: 0.55,
    IndexingStage.VECTOR_UPSERT: 0.7,
    IndexingStage.GRAPH_UPSERT: 0.82,
    IndexingStage.COMMITTING: 0.92,
    IndexingStage.DONE: 1.0,
    IndexingStage.DELETING_PG: 0.1,
    IndexingStage.DELETING_VECTOR: 0.35,
    IndexingStage.DELETING_GRAPH: 0.55,
    IndexingStage.DELETING_OBJECT: 0.75,
}


class _Cancelled(Exception):
    """内部取消信号；不暴露给调用方。"""


class DefaultIndexingPipeline:
    """v1 默认入库流水线。"""

    def __init__(
        self,
        config: IndexerConfig | None = None,
        *,
        ingestion: TextIngestionPipeline,
        embedding: EmbeddingService,
        vector_store: VectorChunkStore,
        graph_store: GraphIndexStore,
        manifest_store: ManifestStore,
        source_store: SourceStore,
        repository: IndexingRepository | None = None,
        job_store: JobStore | None = None,
        graph_extractor: GraphExtractor | None = None,
        graph_chat_model: Any | None = None,
        prompt_registry: Any | None = None,
        graph_cache: Any | None = None,
        event_sink: RagEventSink | None = None,
        worker_id: str = "rag-indexer",
        run_in_background: bool = True,
    ) -> None:
        """注入所有外部依赖；流水线本身不创建连接。"""
        self.config = config or IndexerConfig()
        self.ingestion = ingestion
        self.embedding = embedding
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.manifest_store = manifest_store
        self.source_store = source_store
        self.repository = repository
        self.job_store = job_store or InMemoryJobStore()
        # 显式注入时所有 KB 共用；否则按 KB 的 domain_profile 懒加载并缓存。
        self._graph_extractor_override = graph_extractor
        self._graph_chat_model = graph_chat_model
        self._prompt_registry = prompt_registry
        self._graph_cache = graph_cache
        self.event_sink = event_sink
        self._graph_extractors: dict[str, GraphExtractor] = {}
        self.worker_id = worker_id
        self.run_in_background = run_in_background
        self._tasks: dict[str, asyncio.Task] = {}
        self._cancelled: set[str] = set()
        # 文档 -> 运行中的入库 job；删除文档时先取消并等待它退出，
        # 避免入库在删除之后继续写向量/图造成孤儿数据。
        self._index_jobs_by_doc: dict[UUID, str] = {}
        # 正在删除的文档；禁止同一进程内再次提交入库。
        self._deleting_docs: set[UUID] = set()

    # ——————————————————— 公共接口 ———————————————————

    async def submit(self, request: IndexDocumentRequest) -> IndexingJobView:
        """创建入库 job；后台模式下立即返回 QUEUED/RUNNING 视图。"""
        return await self._create_and_run(
            request,
            background=self.run_in_background,
        )

    async def execute(self, request: IndexDocumentRequest) -> IndexingJobView:
        """同步执行入库，供 PG worker 使用；不创建进程内后台任务。"""
        return await self._create_and_run(request, background=False)

    async def _create_and_run(
        self,
        request: IndexDocumentRequest,
        *,
        background: bool,
    ) -> IndexingJobView:
        """校验请求、创建 job，并按 ``background`` 决定执行方式。"""
        if request.mode is IndexMode.DELETE:
            raise RagError(
                RagErrorCode.INVALID_REQUEST,
                "删除请使用 Indexer.delete(doc_id)",
            )
        if request.doc_id in self._deleting_docs:
            raise RagError(
                RagErrorCode.CONFLICT,
                "文档正在删除，不能提交入库",
                details={"doc_id": str(request.doc_id)},
            )
        document = await self._get_document(request.doc_id)
        if document is not None:
            if document.kb_id != request.kb_id:
                mismatch: dict[str, Any] = {
                    "doc_id": str(request.doc_id),
                    "kb_id": str(request.kb_id),
                    "doc_kb_id": str(document.kb_id),
                }
                raise RagError(
                    RagErrorCode.INVALID_REQUEST,
                    "doc_id 与 kb_id 不匹配",
                    details=mismatch,
                )
        await self._ensure_mode_allowed(document, request.mode)
        # REINDEX 先 bump KB 版本，让重切片期间的旧检索缓存立即失效（D-78）。
        if request.mode is IndexMode.REINDEX and self.repository is not None:
            await self.repository.bump_index_version(request.kb_id)
        await self._ensure_processing(document)
        index_version_before = int(
            getattr(document, "index_version", 0) or 0
        )
        job = IndexingJob(
            job_id=str(uuid4()),
            kb_id=request.kb_id,
            doc_id=request.doc_id,
            mode=request.mode,
            status=IndexingStatus.QUEUED,
            stage=IndexingStage.ACCEPTED,
            max_attempts=self.config.index_max_attempts,
            index_version_before=index_version_before,
        )
        await self.job_store.put(job)
        source = self._with_description_override(request.source, request)
        self._index_jobs_by_doc[request.doc_id] = job.job_id
        await self._emit(
            RagEventName.INDEX_START,
            {
                "job_id": job.job_id,
                "mode": request.mode.value,
                "index_version_before": index_version_before,
            },
            doc_id=request.doc_id,
            kb_id=request.kb_id,
        )
        await self._launch(
            job,
            self._run_index(job, request, source),
            background=background,
        )
        current = await self.job_store.get(job.job_id)
        return to_job_view(current or job)

    async def get_job(self, job_id: str) -> IndexingJobView | None:
        """读取 job；内存 JobStore 或持久化投影由注入实现决定。"""
        job = await self.job_store.get(job_id)
        return to_job_view(job) if job is not None else None

    async def cancel(self, job_id: str) -> None:
        """取消排队/运行中的 job。

        幂等语义（§1.6）：job 不存在或已是终态时直接 no-op，调用方不需要先
        查询状态；worker 只在阶段边界检查取消信号，已写入的 chunk 不回滚。
        """
        job = await self.job_store.get(job_id)
        if job is None or job.status in TERMINAL_STATUSES:
            return
        self._cancelled.add(job_id)
        if job.status is IndexingStatus.QUEUED:
            await self._finish_cancelled(job)

    async def delete(self, doc_id: UUID) -> IndexingJobView:
        """删除文档：先停止在途入库，再 PG 软删 → Weaviate → Neo4j → MinIO。"""
        document = await self._get_document(doc_id)
        if document is None:
            raise RagError(
                RagErrorCode.NOT_FOUND,
                "文档不存在",
                details={"doc_id": str(doc_id)},
            )
        # 先取消/等待同一文档的在途入库，防止删除后入库继续写入。
        self._deleting_docs.add(doc_id)
        try:
            await self._stop_index_task(doc_id)
        except BaseException:
            # 删除请求本身被取消时不留下永久“正在删除”标记。
            self._deleting_docs.discard(doc_id)
            raise
        job = IndexingJob(
            job_id=str(uuid4()),
            kb_id=document.kb_id,
            doc_id=doc_id,
            mode=IndexMode.DELETE,
            status=IndexingStatus.QUEUED,
            stage=IndexingStage.ACCEPTED,
            max_attempts=self.config.index_max_attempts,
        )
        await self.job_store.put(job)
        await self._launch(
            job,
            self._run_delete(job),
            background=self.run_in_background,
        )
        current = await self.job_store.get(job.job_id)
        return to_job_view(current or job)

    async def close(self) -> None:
        """取消进程内后台任务；已持久化的任务由 worker 租约超时回收。"""
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    # ——————————————————— 任务调度 ———————————————————

    async def _launch(
        self,
        job: IndexingJob,
        coro,
        *,
        background: bool,
    ) -> None:
        """按调用方要求决定同步执行还是创建后台任务。"""
        if background:
            task = asyncio.create_task(coro)
            self._tasks[job.job_id] = task

            def _cleanup(_: asyncio.Task) -> None:
                """任务结束时清理索引表；协程未启动就被取消也会走到这里。"""
                self._tasks.pop(job.job_id, None)
                if self._index_jobs_by_doc.get(job.doc_id) == job.job_id:
                    self._index_jobs_by_doc.pop(job.doc_id, None)

            task.add_done_callback(_cleanup)
            return
        await coro

    async def _stop_index_task(self, doc_id: UUID) -> None:
        """取消并等待同一文档的在途入库任务。

        删除操作必须先让入库停在阶段边界/取消点，否则删除完成后入库仍可能
        继续写 Weaviate/Neo4j。等待超过 ``cancel_wait_s`` 后记录告警，
        删除继续执行；随后入库若尝试提交会被 PG 的 deleted_at 条件挡住。
        """
        job_id = self._index_jobs_by_doc.get(doc_id)
        if job_id is None:
            return
        await self.cancel(job_id)
        task = self._tasks.get(job_id)
        if task is None or task.done():
            return
        # 直接取消 asyncio 任务，让长时间 embedding/LLM 阶段也能尽快停下。
        task.cancel()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.1, float(self.config.cancel_wait_s))
        # 轮询 done 而不是 await 任务：子任务取消时的 CancelledError 不应被
        # 误判为删除请求本身被取消；父任务真正取消时 sleep 会正常抛出。
        while not task.done():
            if loop.time() >= deadline:
                logger.warning(
                    "等待入库任务退出超时，继续删除: doc=%s job=%s",
                    doc_id,
                    job_id,
                )
                break
            await asyncio.sleep(0.05)

    async def _run_index(
        self,
        job: IndexingJob,
        request: IndexDocumentRequest,
        source: DocumentSource,
    ) -> None:
        """入库任务主流程；所有异常收敛为 FAILED/DEAD。"""
        log_tokens = bind_index_context(
            doc_id=str(job.doc_id),
            kb_id=str(job.kb_id),
        )
        try:
            job = await self._begin(job)
            job = await self._index(job, request, source)
        except _Cancelled:
            await self._finish_cancelled(job)
        except asyncio.CancelledError:
            await self._finish_cancelled(job)
            raise
        except Exception as exc:  # noqa: BLE001 - 任务边界统一收敛
            await self._finish_failed(job, exc)
        finally:
            reset_index_context(log_tokens)
            self._tasks.pop(job.job_id, None)
            if self._index_jobs_by_doc.get(job.doc_id) == job.job_id:
                self._index_jobs_by_doc.pop(job.doc_id, None)

    async def _run_delete(self, job: IndexingJob) -> None:
        """删除任务主流程；失败时保留 deleting 状态供补偿重试。"""
        log_tokens = bind_index_context(
            doc_id=str(job.doc_id),
            kb_id=str(job.kb_id),
        )
        try:
            job = await self._begin(job)
            # 删除阶段同样会写 index_progress；先拿租约，满足 worker 校验。
            acquire = (
                getattr(self.repository, "acquire_index_lease", None)
                if self.repository is not None
                else None
            )
            if acquire is not None:
                await acquire(
                    job.doc_id,
                    worker_id=self.worker_id,
                    lease_seconds=self.config.index_lease_seconds,
                )
            job = await self._delete(job)
        except _Cancelled:
            await self._finish_cancelled(job)
        except asyncio.CancelledError:
            # 保留取消语义：收尾后继续向上传播，避免 asyncio 任务吞掉取消。
            await self._finish_cancelled(job)
            raise
        except Exception as exc:  # noqa: BLE001 - 任务边界统一收敛
            job = await self._latest(job)
            error = _build_error(exc, job.stage)
            logger.exception("RAG 删除任务失败: %s", job.job_id)
            await self._update(
                job,
                status=IndexingStatus.FAILED,
                error=error,
                ended_at=utcnow(),
            )
        finally:
            reset_index_context(log_tokens)
            self._tasks.pop(job.job_id, None)
            self._deleting_docs.discard(job.doc_id)

    async def _begin(self, job: IndexingJob) -> IndexingJob:
        """置 RUNNING 并累计尝试次数。"""
        return await self._update(
            job,
            status=IndexingStatus.RUNNING,
            attempts=job.attempts + 1,
            started_at=job.started_at or utcnow(),
        )

    # ——————————————————— 入库 ———————————————————

    async def _index(
        self,
        job: IndexingJob,
        request: IndexDocumentRequest,
        source: DocumentSource,
    ) -> IndexingJob:
        """执行入库的所有阶段。"""
        started = time.perf_counter()
        last = started
        timings: dict[str, int] = {}

        def mark(name: str) -> None:
            """记录当前阶段耗时，供入库摘要日志分析。"""
            nonlocal last
            now = time.perf_counter()
            timings[name] = int((now - last) * 1000)
            last = now

        # 1) FETCHING：读取原件并校验内容哈希。
        job = await self._set_stage(job, IndexingStage.FETCHING)
        data = await self.source_store.read_bytes(source)
        if not data:
            raise RagError(
                RagErrorCode.INVALID_REQUEST,
                "源文件为空",
                details={"doc_id": str(request.doc_id)},
            )
        source_sha256 = sha256_bytes(data)
        if source.sha256 and source.sha256.lower() != source_sha256:
            job = await self._warn(
                job,
                IndexingWarningCode.PARSE_FALLBACK,
                "源文件 sha256 与声明不一致，以实际内容为准",
            )
        mark("fetch")

        # 2) PARSING：解析 + 切片 + 描述提取由 TextIngestionPipeline 完成。
        job = await self._set_stage(job, IndexingStage.PARSING)
        result = await self.ingestion.process(
            source,
            data,
            doc_id=request.doc_id,
            kb_id=request.kb_id,
            parser_hint=request.parser_hint,
        )
        mark("parse_chunk")
        if not result.chunks:
            raise RagError(
                RagErrorCode.INVALID_REQUEST,
                "所有 chunk 被质量闸门过滤",
                details={"doc_id": str(request.doc_id)},
            )
        counts = job.counts.model_copy(
            update={
                "pages": len(result.parsed.pages)
                or (result.parsed.profile.page_count or 0),
                "blocks": len(result.parsed.blocks),
                "chunks": len(result.chunks),
                "dropped_blocks": result.noise.dropped_blocks,
                "dropped_chunks": result.noise.dropped_chunks,
                "low_quality_blocks": result.noise.low_quality_chunks,
                "tokens": sum(item.token_count for item in result.chunks),
            }
        )
        job = await self._update(job, counts=counts)
        # v1 的 parse/chunk/dedup/description 在一个调用内完成；这里补记阶段
        # 便于 job view 与 PG index_stage 反映真实边界。
        job = await self._set_stage(job, IndexingStage.CHUNKING)
        job = await self._set_stage(job, IndexingStage.DEDUPING)
        job = await self._set_stage(job, IndexingStage.DESCRIBING)
        job = await self._warn_many(job, result.warnings, IndexingStage.PARSING)

        # 3) 写解析产物 manifest + blocks。
        manifest = build_parse_manifest(
            result.parsed,
            source_sha256=source_sha256,
            token_count=counts.tokens,
        )
        manifest_path = await self.manifest_store.write(
            manifest,
            result.parsed.blocks,
        )
        mark("manifest")

        # 本次写入的版本号与落库模型；后续图/向量阶段复用同一份 Chunk。
        index_version = job.index_version_before + 1
        chunks = drafts_to_chunks(
            result.chunks,
            parsed=result.parsed,
            index_version=index_version,
            chunking_version=self.config.chunking_version,
        )

        # 4) ENRICHING：LLM 图实体/关系抽取。
        build_graph = (
            self.config.graph_build_enabled
            if request.graph_build_enabled is None
            else request.graph_build_enabled
        )
        job = await self._set_stage(job, IndexingStage.ENRICHING)
        extractions = {}
        if build_graph:
            domain_profile = await self._get_kb_domain_profile(request.kb_id)
            graph_extractor = self._graph_extractor_for(domain_profile)
            # 抽取器内部按 chunk 数/字符数切批并限制并发。
            extractions = await graph_extractor.extract_batch(
                chunks,
                max_concurrency=self.config.graph_llm_concurrency,
            )
        else:
            job = await self._warn(
                job,
                IndexingWarningCode.GRAPH_SKIPPED,
                "建图开关关闭，跳过 Neo4j 写入（仅测试/开发使用）",
            )
        mark("graph_extract")

        # 5) EMBEDDING：使用 embedding_text，不落库正文。
        job = await self._set_stage(job, IndexingStage.EMBEDDING)
        vectors = await self.embedding.embed_chunks(result.chunks)
        self._check_dimensions(vectors)
        mark("embedding")

        # 6) VECTOR_UPSERT：
        # Weaviate 批量 insert 对已存在 uuid 会报错，因此先删除“同 id 的旧版本”，
        # 再写新 chunk；文档此时是 processing，读路径已过滤，短暂空窗安全。
        job = await self._set_stage(job, IndexingStage.VECTOR_UPSERT)
        new_ids = {chunk.chunk_id for chunk in chunks}
        existing_ids = set(await self.vector_store.list_chunk_ids(request.doc_id))
        replaced_ids = sorted(existing_ids & new_ids)
        if replaced_ids:
            await self.vector_store.delete_chunks(replaced_ids)
        await self.vector_store.upsert_chunks(chunks, vectors.vectors)
        stale_ids = sorted(existing_ids - new_ids)
        if stale_ids:
            await self.vector_store.delete_chunks(stale_ids)
        mark("vector_upsert")

        # 7) GRAPH_UPSERT：图写入失败不回滚 Weaviate（D-56/§8.4）。
        job = await self._set_stage(job, IndexingStage.GRAPH_UPSERT)
        if build_graph:
            for chunk in chunks:
                extraction = extractions.get(chunk.chunk_id)
                if extraction is None:
                    continue
                # 表格/代码等低语义块在抽取器里返回空结果；不写空 ChunkRef，
                # 保证图里只保留来自标题 + 正文的概念与关系。
                if not extraction.keywords and not extraction.relations:
                    continue
                await self.graph_store.upsert_chunk(
                    chunk.chunk_id,
                    str(chunk.doc_id),
                    str(chunk.kb_id),
                    extraction.keywords,
                )
                await self.graph_store.upsert_relations(
                    chunk.chunk_id,
                    str(chunk.doc_id),
                    str(chunk.kb_id),
                    extraction.relations,
                )
        mark("graph_upsert")

        # 8) COMMITTING：描述 + PG 状态 + 版本。
        job = await self._set_stage(job, IndexingStage.COMMITTING)
        description = (
            result.parsed.description.description
            if result.parsed.description is not None
            else None
        )
        if self.repository is not None:
            if description:
                await self.repository.set_description(request.doc_id, description)
            await self.repository.mark_completed(
                request.doc_id,
                token_count=counts.tokens,
                chunk_count=len(chunks),
                parse_manifest_key=manifest_path,
                index_version=index_version,
                worker_id=self.worker_id,
            )
            await self.repository.bump_index_version(request.kb_id)

        job = await self._update(
            job,
            stage=IndexingStage.DONE,
            progress=1.0,
            status=IndexingStatus.SUCCEEDED,
            index_version_after=index_version,
            ended_at=utcnow(),
        )
        mark("commit")
        graph_entities = sum(len(item.keywords) for item in extractions.values())
        graph_relations = sum(
            len(item.relations) for item in extractions.values()
        )
        await self._emit(
            RagEventName.INDEX_END,
            {
                "job_id": job.job_id,
                "mode": request.mode.value,
                "status": "succeeded",
                "size_bytes": len(data),
                "pages": counts.pages,
                "blocks": counts.blocks,
                "chunks": len(chunks),
                "tokens": counts.tokens,
                "index_version": index_version,
                "graph_entities": graph_entities,
                "graph_relations": graph_relations,
                "stage_timings": timings,
                "duration_ms": int((time.perf_counter() - started) * 1000),
            },
            doc_id=request.doc_id,
            kb_id=request.kb_id,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
        return job

    # ——————————————————— 删除 ———————————————————

    async def _delete(self, job: IndexingJob) -> IndexingJob:
        """执行跨存储删除；每步幂等，失败保留 deleting 供补偿。"""
        job = await self._set_stage(job, IndexingStage.DELETING_PG)
        if self.repository is not None:
            await self.repository.mark_deleting(job.doc_id)

        job = await self._set_stage(job, IndexingStage.DELETING_VECTOR)
        await self.vector_store.delete_document(job.doc_id)

        job = await self._set_stage(job, IndexingStage.DELETING_GRAPH)
        await self.graph_store.delete_document(str(job.doc_id))

        job = await self._set_stage(job, IndexingStage.DELETING_OBJECT)
        await self.manifest_store.delete(job.kb_id, job.doc_id)
        await self.source_store.delete_document(
            job.kb_id,
            job.doc_id,
        )

        if self.repository is not None:
            await self.repository.bump_index_version(job.kb_id)
            await self.repository.mark_deleted(job.doc_id)

        await self._emit(
            RagEventName.INDEX_END,
            {"job_id": job.job_id, "mode": "delete", "status": "succeeded"},
            doc_id=job.doc_id,
            kb_id=job.kb_id,
        )
        return await self._update(
            job,
            stage=IndexingStage.DONE,
            progress=1.0,
            status=IndexingStatus.SUCCEEDED,
            ended_at=utcnow(),
        )

    # ——————————————————— 状态/错误 ———————————————————

    async def _set_stage(
        self,
        job: IndexingJob,
        stage: IndexingStage,
    ) -> IndexingJob:
        """切换阶段并写 PG index_progress；同时检查取消信号。"""
        job = await self._update(
            job,
            stage=stage,
            progress=STAGE_PROGRESS.get(stage, job.progress),
        )
        if self.repository is not None:
            await self.repository.update_progress(
                job.doc_id,
                stage=stage,
                progress=job.progress,
                worker_id=self.worker_id,
                # 删除流程 mark_deleting 已写 deleted_at，仍需记录后续阶段。
                include_deleted=job.mode is IndexMode.DELETE,
            )
        self._check_cancelled(job.job_id)
        await self._emit(
            RagEventName.INDEX_STAGE,
            {
                "job_id": job.job_id,
                "stage": stage.value,
                "progress": job.progress,
                "mode": job.mode.value,
            },
            doc_id=job.doc_id,
            kb_id=job.kb_id,
        )
        return job

    async def _emit(
        self,
        name: RagEventName,
        payload: dict[str, Any],
        *,
        doc_id: UUID | None = None,
        kb_id: UUID | None = None,
        duration_ms: int | None = None,
    ) -> None:
        """发送入库事件；未注入 sink 时 no-op。"""
        await emit_rag_event(
            self.event_sink,
            RagEvent(
                name=name,
                payload=payload,
                doc_id=str(doc_id) if doc_id else None,
                kb_id=str(kb_id) if kb_id else None,
                duration_ms=duration_ms,
            ),
        )

    async def _update(self, job: IndexingJob, **updates) -> IndexingJob:
        """不可变 job 的统一替换 + 持久化。"""
        updated = job.model_copy(update=updates)
        await self.job_store.put(updated)
        return updated

    async def _warn(
        self,
        job: IndexingJob,
        code: IndexingWarningCode,
        message: str,
        stage: IndexingStage | None = None,
    ) -> IndexingJob:
        """追加一条去重告警。"""
        wave = IndexingWarning(
            code=code,
            stage=stage or job.stage,
            message=message,
        )
        if any(
            item.code == code and item.message == message
            for item in job.warnings
        ):
            return job
        await self._emit(
            RagEventName.INDEX_WARNING,
            {
                "job_id": job.job_id,
                "code": code.value,
                "message": message,
                "stage": (stage or job.stage).value,
            },
            doc_id=job.doc_id,
            kb_id=job.kb_id,
        )
        return await self._update(
            job,
            warnings=[*job.warnings, wave],
        )

    async def _warn_many(
        self,
        job: IndexingJob,
        messages: list[str],
        stage: IndexingStage,
    ) -> IndexingJob:
        """把解析器/切片的字符串告警映射成结构化告警。"""
        for message in messages:
            job = await self._warn(
                job,
                _warning_code(message),
                message,
                stage=stage,
            )
        return job

    async def _finish_cancelled(self, job: IndexingJob) -> None:
        """统一取消收尾：终态 CANCELLED + PG failed/CANCELLED 投影。"""
        job = await self._latest(job)
        self._cancelled.discard(job.job_id)
        error = IndexingError(
            code=IndexingErrorCode.CANCELLED,
            stage=job.stage,
            message="入库任务已取消",
        )
        await self._update(
            job,
            status=IndexingStatus.CANCELLED,
            error=error,
            ended_at=utcnow(),
        )
        await self._record_failure(job, error)
        await self._emit(
            RagEventName.INDEX_END,
            {
                "job_id": job.job_id,
                "mode": job.mode.value,
                "status": "cancelled",
                "error_code": error.code.value,
                "stage": job.stage.value,
            },
            doc_id=job.doc_id,
            kb_id=job.kb_id,
        )

    async def _finish_failed(self, job: IndexingJob, exc: Exception) -> None:
        """统一失败收尾：错误映射、DEAD 判定、PG 退避。"""
        # _set_stage 可能已把新阶段写进 JobStore，失败收尾必须读最新状态。
        job = await self._latest(job)
        error = _build_error(exc, job.stage)
        status = (
            IndexingStatus.DEAD
            if job.attempts >= job.max_attempts
            else IndexingStatus.FAILED
        )
        if status is IndexingStatus.DEAD:
            error = error.model_copy(update={"code": IndexingErrorCode.DEAD})
        logger.exception(
            "RAG 入库任务失败: job=%s stage=%s code=%s",
            job.job_id,
            job.stage,
            error.code,
        )
        await self._update(
            job,
            status=status,
            error=error,
            ended_at=utcnow(),
        )
        await self._record_failure(job, error)
        await self._emit(
            RagEventName.INDEX_END,
            {
                "job_id": job.job_id,
                "mode": job.mode.value,
                "status": status.value,
                "error_code": error.code.value,
                "stage": job.stage.value,
            },
            doc_id=job.doc_id,
            kb_id=job.kb_id,
        )

    async def _latest(self, job: IndexingJob) -> IndexingJob:
        """读取 JobStore 中的最新 job；不存在时回退到入参。"""
        return await self.job_store.get(job.job_id) or job

    async def _record_failure(
        self,
        job: IndexingJob,
        error: IndexingError,
    ) -> None:
        """写 PG 失败/取消状态；PG 写失败只记录日志，不覆盖任务结果。"""
        if self.repository is None:
            return
        try:
            await self.repository.mark_failed(
                job.doc_id,
                error=json_dumps(error.model_dump(mode="json")),
                worker_id=self.worker_id,
                retry_backoff_s=self.config.index_retry_backoff_s,
                max_attempts=self.config.index_max_attempts,
            )
        except Exception:  # noqa: BLE001 - 失败收尾不能再次抛出
            logger.exception("写 PG 入库失败状态失败: %s", job.doc_id)

    def _check_cancelled(self, job_id: str) -> None:
        """阶段边界检查取消信号。"""
        if job_id in self._cancelled:
            raise _Cancelled()

    def _check_dimensions(self, result: EmbeddingResult) -> None:
        """校验 embedding 维度与配置一致，避免写坏 collection。"""
        expected = int(self.config.embedding_dimensions)
        if result.dimensions and result.dimensions != expected:
            raise RagError(
                RagErrorCode.INVALID_CONFIG,
                "embedding 维度与 IndexerConfig.embedding_dimensions 不一致",
                details={
                    "expected": expected,
                    "actual": result.dimensions,
                    "model_id": self.embedding.model_id,
                },
            )
        declared = self.embedding.dimensions
        if declared is not None and declared != expected:
            raise RagError(
                RagErrorCode.INVALID_CONFIG,
                "embedding 模型声明维度与配置不一致",
                details={"expected": expected, "declared": declared},
            )

    async def _get_kb_domain_profile(self, kb_id: UUID) -> str | None:
        """读取 KB 绑定的领域画像名；未配置/仓库不支持时回退全局配置。"""
        if self.repository is None:
            return None
        getter = getattr(self.repository, "get_knowledge_base", None)
        if not callable(getter):
            return None
        # 仓库协议可选扩展；运行时按存在性探测。
        kb = await getter(kb_id)  # pyright: ignore[reportGeneralTypeIssues]
        return getattr(kb, "domain_profile", None)

    def _graph_extractor_for(self, domain_profile: str | None) -> GraphExtractor:
        """按 KB 的领域画像选择抽取器；同画像实例复用。"""
        if self._graph_extractor_override is not None:
            return self._graph_extractor_override
        name = (domain_profile or self.config.domain_profile or "pet_medical").strip()
        cached = self._graph_extractors.get(name)
        if cached is not None:
            return cached
        configured_path = self.config.domain_profile_path
        profile_path = (
            configured_path
            if configured_path and Path(configured_path).stem == name
            else None
        )
        profile = load_domain_profile(
            name,
            path=profile_path,
            directory=self.config.domain_profile_dir,
        )
        extractor = build_graph_extractor(
            self.config,
            profile=profile,
            chat_model=self._graph_chat_model,
            prompts=self._prompt_registry,
            cache=self._graph_cache,
            event_sink=self.event_sink,
        )
        self._graph_extractors[name] = extractor
        return extractor

    async def _get_document(self, doc_id: UUID) -> Document | None:
        """读取 PG 文档；未注入 repository 时返回 None（测试场景）。"""
        if self.repository is None:
            return None
        document = await self.repository.get_document(doc_id)
        if document is None:
            raise RagError(
                RagErrorCode.NOT_FOUND,
                "文档不存在",
                details={"doc_id": str(doc_id)},
            )
        return document

    @staticmethod
    def _document_status(document: Document | None) -> str | None:
        """取文档状态的字符串值；测试投影可能没有 status。"""
        if document is None:
            return None
        status = getattr(document, "status", None)
        return getattr(status, "value", status)

    async def _ensure_mode_allowed(
        self,
        document: Document | None,
        mode: IndexMode,
    ) -> None:
        """CREATE 不允许覆盖已有索引；UPDATE/REINDEX 可以。"""
        if document is None or mode is not IndexMode.CREATE:
            return
        if self._document_status(document) == "completed":
            raise RagError(
                RagErrorCode.CONFLICT,
                "文档已有索引；请使用 UPDATE 或 REINDEX",
                details={"doc_id": str(document.id)},
            )

    async def _ensure_processing(self, document: Document | None) -> None:
        """入库前把文档置 processing，并确保当前 worker 持有租约。

        worker 通过 ``claim_index_jobs`` 领取时锁字段已经写好；直接调用
        ``submit``/``execute`` 时没有 claim，需要补租约，否则后续
        ``update_index_progress/mark_index_completed`` 的 worker 校验会失败。
        """
        if self.repository is None or document is None:
            return
        value = self._document_status(document)
        if value in {"deleting", "deleted"}:
            raise RagError(
                RagErrorCode.CONFLICT,
                "文档正在删除或已删除，不能入库",
                details={"doc_id": str(document.id), "status": str(value)},
            )
        acquire = getattr(self.repository, "acquire_index_lease", None)
        if acquire is not None:
            await acquire(
                document.id,
                worker_id=self.worker_id,
                lease_seconds=self.config.index_lease_seconds,
            )
            return
        if value != "processing":
            await self.repository.mark_processing(document.id)

    @staticmethod
    def _with_description_override(
        source: DocumentSource,
        request: IndexDocumentRequest,
    ) -> DocumentSource:
        """把人工描述合并进 source.metadata，复用描述提取优先级。"""
        override = (request.description_override or "").strip()
        if not override:
            return source
        metadata = dict(source.metadata)
        metadata["description"] = override
        return source.model_copy(update={"metadata": metadata})


def _warning_code(message: str) -> IndexingWarningCode:
    """把自由文本告警映射到枚举；未知告警按 PARSE_FALLBACK 记录。"""
    text = message or ""
    if "描述" in text:
        return IndexingWarningCode.DESCRIPTION_FALLBACK
    if "OCR" in text or "覆盖率" in text:
        return IndexingWarningCode.LOW_TEXT_COVERAGE
    if "表格" in text:
        return IndexingWarningCode.LOW_TABLE
    if "重复" in text:
        return IndexingWarningCode.DUPLICATE_DROPPED
    if "截断" in text:
        return IndexingWarningCode.CHUNK_TRUNCATED
    if "过短" in text:
        return IndexingWarningCode.CHUNK_TOO_SHORT
    return IndexingWarningCode.PARSE_FALLBACK


def _build_error(exc: Exception, stage: IndexingStage) -> IndexingError:
    """把阶段内异常转换成 ``IndexingError``。"""
    code = _error_code(exc, stage)
    details: dict[str, Any] = {"cause": str(exc)}
    raw_details = getattr(exc, "details", None)
    if isinstance(raw_details, dict):
        details.update(
            {
                str(key): value
                for key, value in raw_details.items()
                if isinstance(value, (str, int, float, bool)) or value is None
            }
        )
    return IndexingError(
        code=code,
        stage=stage,
        message=str(exc) or code.value,
        retryable=bool(getattr(exc, "retryable_value", False))
        or stage
        in {
            IndexingStage.FETCHING,
            IndexingStage.EMBEDDING,
            IndexingStage.VECTOR_UPSERT,
            IndexingStage.GRAPH_UPSERT,
            IndexingStage.COMMITTING,
        },
        details=details,
    )


def _error_code(exc: Exception, stage: IndexingStage) -> IndexingErrorCode:
    """阶段 + 异常类型 → 稳定错误码。"""
    if stage is IndexingStage.EMBEDDING:
        return IndexingErrorCode.EMBEDDING_FAILED
    if stage is IndexingStage.VECTOR_UPSERT:
        return IndexingErrorCode.VECTOR_WRITE_FAILED
    if stage is IndexingStage.GRAPH_UPSERT:
        return IndexingErrorCode.GRAPH_WRITE_FAILED
    if stage is IndexingStage.COMMITTING:
        return IndexingErrorCode.COMMIT_FAILED
    if stage is IndexingStage.FETCHING:
        if isinstance(exc, MinioObjectNotFoundError):
            return IndexingErrorCode.CORRUPT_FILE
        return IndexingErrorCode.PARSE_FAILED
    if isinstance(exc, RagError):
        message = exc.message
        if "不支持" in message or "未启用" in message:
            return IndexingErrorCode.UNSUPPORTED_TYPE
        if (
            "没有可索引文本" in message
            or "覆盖率过低" in message
            or "源文件为空" in message
        ):
            return IndexingErrorCode.EMPTY_CONTENT
        if "chunk" in message or "切片" in message:
            return IndexingErrorCode.LOW_QUALITY
        if "解析" in message:
            return IndexingErrorCode.PARSE_FAILED
    return IndexingErrorCode.INTERNAL
