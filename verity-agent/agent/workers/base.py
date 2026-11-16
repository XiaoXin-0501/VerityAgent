"""统一 WorkerTask 协议与运行器。

现有记忆/删除/对账 worker 仍可直接 run_forever；本框架为后续压缩、
跨库删除、对账等任务提供统一的 claim/execute/heartbeat/complete/fail 形状。
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "TaskStatus",
    "TaskLease",
    "TaskError",
    "TaskHandle",
    "TaskResult",
    "WorkerPolicy",
    "WorkerTask",
    "WorkerRunner",
]

logger = logging.getLogger(__name__)


class TaskStatus(StrEnum):
    """任务状态；终态不可回退。"""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DEAD = "dead"
    CANCELLED = "cancelled"


class _Model(BaseModel):
    """Worker DTO 基类。"""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class TaskLease(_Model):
    """worker 对单个任务的租约。"""

    worker_id: str
    locked_until: datetime


class TaskError(_Model):
    """任务错误；由 worker 决定是否重试。"""

    code: str
    message: str
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class TaskHandle(_Model):
    """claim 成功后的任务句柄。"""

    task_id: str
    task_type: str
    lease: TaskLease
    attempt: int = 1
    payload: dict[str, Any] = Field(default_factory=dict)


class TaskResult(_Model):
    """任务执行结果。"""

    status: TaskStatus = TaskStatus.SUCCEEDED
    counts: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    duration_ms: int = 0


class WorkerPolicy(_Model):
    """worker 轮询/租约/重试策略。"""

    scan_interval_s: float = Field(default=30.0, gt=0)
    batch_size: int = Field(default=20, ge=1)
    max_concurrency: int = Field(default=1, ge=1)
    lease_s: int = Field(default=120, ge=10)
    heartbeat_s: float = Field(default=30.0, gt=0)
    max_attempts: int = Field(default=3, ge=1)
    backoff_base_s: float = Field(default=30.0, gt=0)
    backoff_max_s: float = Field(default=600.0, gt=0)


class WorkerTask(Protocol):
    """一个可被 WorkerRunner 驱动的任务实现。"""

    task_type: str

    async def claim(self, worker_id: str, lease_s: int) -> TaskHandle | None:
        """原子领取一个任务；无任务返回 None。"""
        ...

    async def execute(self, handle: TaskHandle) -> TaskResult:
        """执行任务；业务异常由 runner 转 fail。"""
        ...

    async def heartbeat(self, handle: TaskHandle, lease_s: int) -> bool:
        """续租；租约丢失返回 False。"""
        ...

    async def complete(self, handle: TaskHandle, result: TaskResult) -> None:
        """写成功终态。"""
        ...

    async def fail(self, handle: TaskHandle, error: TaskError) -> None:
        """写失败/重试/dead。"""
        ...


class WorkerRunner:
    """通用单任务 worker 循环。"""

    def __init__(
        self,
        task: WorkerTask,
        *,
        worker_id: str,
        policy: WorkerPolicy | None = None,
    ) -> None:
        """保存任务实现与策略。"""
        self.task = task
        self.worker_id = worker_id
        self.policy = policy or WorkerPolicy()
        self._stop = asyncio.Event()

    def stop(self) -> None:
        """请求 runner 退出。"""
        self._stop.set()

    async def run_forever(self) -> None:
        """持续 claim/execute；空闲或失败后按策略等待。"""
        while not self._stop.is_set():
            worked = await self.run_once()
            if worked:
                continue
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=self.policy.scan_interval_s,
                )
            except asyncio.TimeoutError:
                continue

    async def run_once(self) -> bool:
        """执行一个任务；返回是否处理了任务。"""
        handle = await self.task.claim(self.worker_id, self.policy.lease_s)
        if handle is None:
            return False
        heartbeat = asyncio.create_task(
            self._heartbeat_loop(handle),
            name=f"worker-heartbeat-{handle.task_id}",
        )
        try:
            result = await self.task.execute(handle)
            await self.task.complete(handle, result)
            return True
        except asyncio.CancelledError:
            await self.task.fail(
                handle,
                TaskError(code="CANCELLED", message="任务已取消"),
            )
            raise
        except Exception as exc:  # noqa: BLE001 - 任务失败转 TaskError
            logger.exception("Worker 任务失败: %s", handle.task_id)
            await self.task.fail(
                handle,
                TaskError(
                    code=type(exc).__name__,
                    message=str(exc)[:500],
                    retryable=True,
                ),
            )
            return True
        finally:
            heartbeat.cancel()
            try:
                await heartbeat
            except asyncio.CancelledError:
                pass

    async def _heartbeat_loop(self, handle: TaskHandle) -> None:
        """按策略续租；丢失租约后退出循环。"""
        while True:
            await asyncio.sleep(self.policy.heartbeat_s)
            try:
                ok = await self.task.heartbeat(handle, self.policy.lease_s)
            except Exception:  # noqa: BLE001 - 续租失败按丢失处理
                logger.warning("Worker 续租失败: %s", handle.task_id, exc_info=True)
                return
            if not ok:
                return
