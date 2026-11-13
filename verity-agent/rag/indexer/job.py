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
    """把内部运行态 job 投影成公共视图，去掉锁与租约字段。"""
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
    """运行态 job 存储协议；生产实现可把状态投影到 PG ``documents.index_*``。"""

    async def put(self, job: IndexingJob) -> IndexingJob:
        """保存/覆盖 job，返回保存后的对象。"""
        ...

    async def get(self, job_id: str) -> IndexingJob | None:
        """按 id 读取 job；不存在返回 None。"""
        ...

    async def delete(self, job_id: str) -> bool:
        """删除 job；返回是否真的删除。"""
        ...


class InMemoryJobStore:
    """进程内 JobStore；只适合单实例/开发与测试。"""

    def __init__(self) -> None:
        """初始化空 job 表。"""
        self._jobs: dict[str, IndexingJob] = {}

    async def put(self, job: IndexingJob) -> IndexingJob:
        """覆盖保存 job。"""
        self._jobs[job.job_id] = job
        return job

    async def get(self, job_id: str) -> IndexingJob | None:
        """按 id 读取 job。"""
        return self._jobs.get(job_id)

    async def delete(self, job_id: str) -> bool:
        """删除 job；返回是否存在。"""
        return self._jobs.pop(job_id, None) is not None
