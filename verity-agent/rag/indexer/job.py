"""入库任务状态机与内存 JobStore（框架层）。

持久化版本复用 PG ``documents.index_*``（D-77）；本模块只定义运行态接口，
方便 Indexer 门面和测试使用。
"""

from __future__ import annotations

from typing import Protocol

from rag.shared.enums import IndexingStatus
from rag.shared.models import IndexingJob, IndexingJobView

__all__ = ["JobStore", "InMemoryJobStore", "to_job_view", "TERMINAL_STATUSES"]

TERMINAL_STATUSES = frozenset(
    {
        IndexingStatus.SUCCEEDED,
        IndexingStatus.FAILED,
        IndexingStatus.CANCELLED,
        IndexingStatus.DEAD,
    }
)


def to_job_view(job: IndexingJob) -> IndexingJobView:
    return IndexingJobView(
        job_id=job.job_id,
        kb_id=job.kb_id,
        doc_id=job.doc_id,
        mode=job.mode,
        status=job.status,
        stage=job.stage,
        progress=job.progress,
        attempts=job.attempts,
        error=job.error,
        warnings=job.warnings,
        index_version_before=job.index_version_before,
        index_version_after=job.index_version_after,
        counts=job.counts,
        created_at=job.created_at,
        started_at=job.started_at,
        ended_at=job.ended_at,
    )


class JobStore(Protocol):
    async def put(self, job: IndexingJob) -> IndexingJob: ...
    async def get(self, job_id: str) -> IndexingJob | None: ...
    async def delete(self, job_id: str) -> bool: ...


class InMemoryJobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, IndexingJob] = {}

    async def put(self, job: IndexingJob) -> IndexingJob:
        self._jobs[job.job_id] = job
        return job

    async def get(self, job_id: str) -> IndexingJob | None:
        return self._jobs.get(job_id)

    async def delete(self, job_id: str) -> bool:
        return self._jobs.pop(job_id, None) is not None
