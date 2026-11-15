"""Agent 后台任务运行器。

后台任务只做**不阻塞主回复**的收尾工作：记忆入队、投影更新、指标刷新等。
主循环通过 ``submit`` 投递，Agent 在取消/关闭时按 run_id 统一取消并等待。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Coroutine

from utils.time import utcnow

__all__ = ["BackgroundTaskRunner", "BackgroundTaskInfo"]

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BackgroundTaskInfo:
    """后台任务元数据；仅用于排障/取消，不落库。"""

    name: str
    run_id: str | None
    task: asyncio.Task
    created_at: datetime = field(default_factory=utcnow)


class BackgroundTaskRunner:
    """进程内后台任务集合；支持按 run 取消与优雅关闭。"""

    def __init__(self, *, logger_: logging.Logger | None = None) -> None:
        """创建空任务集合；不启动后台线程/事件循环。"""
        self._logger = logger_ or logger
        self._tasks: dict[asyncio.Task, BackgroundTaskInfo] = {}
        self._closed = False

    def submit(
        self,
        name: str,
        coro: Coroutine[Any, Any, Any],
        *,
        run_id: str | None = None,
        timeout_s: float | None = None,
    ) -> asyncio.Task:
        """投递后台任务；任务失败只记录，不影响主流程。"""
        if self._closed:
            raise RuntimeError("BackgroundTaskRunner 已关闭")
        task = asyncio.create_task(
            self._run(name, coro, timeout_s=timeout_s),
            name=f"verity-bg-{name}",
        )
        info = BackgroundTaskInfo(name=name, run_id=run_id, task=task)
        self._tasks[task] = info
        task.add_done_callback(self._on_done)
        return task

    async def cancel_run(self, run_id: str) -> None:
        """取消某个 run 的所有后台任务；不存在时 no-op。"""
        tasks = [
            info.task
            for info in list(self._tasks.values())
            if info.run_id == run_id and not info.task.done()
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def close(self, grace_s: float = 5.0) -> None:
        """等待任务完成；超过宽限期强制取消，幂等。"""
        if self._closed:
            return
        self._closed = True
        tasks = [info.task for info in list(self._tasks.values()) if not info.task.done()]
        if not tasks:
            return
        _, pending = await asyncio.wait(tasks, timeout=max(0.0, grace_s))
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    def stats(self) -> dict[str, int]:
        """返回当前/累计任务计数，供健康检查/排障。"""
        active = sum(1 for info in self._tasks.values() if not info.task.done())
        return {"active": active, "tracked": len(self._tasks)}

    async def _run(
        self,
        name: str,
        coro: Coroutine[Any, Any, Any],
        *,
        timeout_s: float | None,
    ) -> Any:
        """执行单个任务并统一超时/异常处理。"""
        try:
            if timeout_s is not None and timeout_s > 0:
                return await asyncio.wait_for(coro, timeout=timeout_s)
            return await coro
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - 后台任务失败只记录
            self._logger.exception("后台任务失败: %s", name)
            return None

    def _on_done(self, task: asyncio.Task) -> None:
        """任务结束后移除引用；异常已由 _run 记录。"""
        info = self._tasks.pop(task, None)
        if info is not None and not task.cancelled():
            try:
                task.exception()
            except asyncio.CancelledError:
                pass
            except Exception:  # pragma: no cover - 已记录
                pass
