"""Hook 分发器：选择、排序、采样、执行、超时、错误隔离、后台队列。"""

from __future__ import annotations

import asyncio
import logging
from time import perf_counter

from config.hooks import HooksConfig
from exception import HookClosedError, HookExecutionError
from utils.sampling import stable_sample

from .enum import (
    HookExecutionMode,
    HookFailurePolicy,
    HookStatus,
)
from .models import (
    HookContext,
    HookDispatchResult,
    HookOutcome,
    HookStats,
    RegisteredHook,
)
from .registry import HookRegistry

__all__ = ["HookDispatcher"]


class HookDispatcher:
    def __init__(
        self,
        registry: HookRegistry,
        config: HooksConfig,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        self.registry = registry
        self.config = config
        self._logger = logger or logging.getLogger(__name__)
        self._queue: asyncio.Queue[tuple[HookContext, RegisteredHook]] | None = None
        self._worker: asyncio.Task | None = None
        self._closed = False
        self._stats: dict[tuple[str, str], HookStats] = {}

    # ——————————————————— 公共入口 ———————————————————

    async def emit(self, context: HookContext) -> HookDispatchResult:
        started = perf_counter()
        result = HookDispatchResult(event=context.event)
        if not self.config.enabled:
            return result
        if self._closed:
            raise HookClosedError("HookDispatcher 已关闭")

        hooks = sorted(
            self.registry.list(context.event),
            key=lambda hook: (hook.spec.priority, hook.sequence),
        )
        result.selected = len(hooks)
        inline: list[RegisteredHook] = []
        for hook in hooks:
            if not hook.spec.enabled:
                result.skipped += 1
                continue
            if not stable_sample(
                f"{context.trace_id}:{hook.spec.name}", hook.spec.sampling_rate
            ):
                result.skipped += 1
                continue
            if hook.spec.mode is HookExecutionMode.BACKGROUND:
                if not self._enqueue(hook, context):
                    result.skipped += 1
                continue
            inline.append(hook)

        if inline:
            if any(
                hook.spec.mode is HookExecutionMode.INLINE_PARALLEL for hook in inline
            ):
                outcomes = await asyncio.gather(
                    *(self._run(hook, context) for hook in inline)
                )
            else:
                outcomes = [await self._run(hook, context) for hook in inline]
            result.outcomes.extend(outcomes)
            result.executed = len(outcomes)
            result.failed = sum(
                1 for item in outcomes if item.status is HookStatus.FAILED
            )
        result.duration_ms = int((perf_counter() - started) * 1000)
        return result

    async def flush(self) -> None:
        if self._queue is not None:
            await self._queue.join()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self.flush()
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None

    def get_stats(self) -> list[HookStats]:
        return list(self._stats.values())

    # ——————————————————— 内部实现 ———————————————————

    def _enqueue(self, hook: RegisteredHook, context: HookContext) -> bool:
        queue = self._ensure_worker()
        try:
            queue.put_nowait((context, hook))
            return True
        except asyncio.QueueFull:
            if self.config.drop_on_overflow:
                self._logger.warning(
                    "Hooks 后台队列已满，丢弃事件: event=%s hook=%s",
                    context.event,
                    hook.spec.name,
                )
                return False
            raise

    def _ensure_worker(self) -> asyncio.Queue[tuple[HookContext, RegisteredHook]]:
        if self._queue is None:
            self._queue = asyncio.Queue(maxsize=self.config.max_queue_size)
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(
                self._background_worker(), name="verity-hooks-worker"
            )
        return self._queue

    async def _background_worker(self) -> None:
        assert self._queue is not None
        while True:
            context, hook = await self._queue.get()
            try:
                await self._run(hook, context)
            except HookExecutionError:
                # FAIL_CLOSED 在后台没有调用方可以阻断，只能记录。
                self._logger.exception("后台 Hook 失败: %s", hook.spec.name)
            except Exception:  # pragma: no cover - 防御
                self._logger.exception("后台 Hook 未处理异常: %s", hook.spec.name)
            finally:
                self._queue.task_done()

    async def _run(self, hook: RegisteredHook, context: HookContext) -> HookOutcome:
        started = perf_counter()
        timeout = hook.spec.timeout_s or self.config.dispatch_timeout_s
        try:
            await asyncio.wait_for(hook.handler(context), timeout=timeout)
        except asyncio.TimeoutError as exc:
            duration = int((perf_counter() - started) * 1000)
            outcome = HookOutcome(
                hook_name=hook.spec.name,
                status=HookStatus.TIMEOUT,
                duration_ms=duration,
                error="timeout",
            )
            self._record(hook, context, outcome, timed_out=True)
            if hook.spec.failure_policy is HookFailurePolicy.FAIL_CLOSED:
                raise HookExecutionError(
                    f"Hook 超时: {hook.spec.name}",
                    details={"hook": hook.spec.name},
                ) from exc
            return outcome
        except Exception as exc:
            duration = int((perf_counter() - started) * 1000)
            outcome = HookOutcome(
                hook_name=hook.spec.name,
                status=HookStatus.FAILED,
                duration_ms=duration,
                error=str(exc),
            )
            self._record(hook, context, outcome)
            if hook.spec.failure_policy is HookFailurePolicy.FAIL_CLOSED:
                raise HookExecutionError(
                    f"Hook 执行失败: {hook.spec.name}",
                    details={"hook": hook.spec.name, "cause": str(exc)},
                ) from exc
            if hook.spec.failure_policy is not HookFailurePolicy.IGNORE:
                self._logger.warning(
                    "Hook 执行失败: %s (%s)", hook.spec.name, exc
                )
            return outcome
        duration = int((perf_counter() - started) * 1000)
        outcome = HookOutcome(
            hook_name=hook.spec.name,
            status=HookStatus.SUCCESS,
            duration_ms=duration,
        )
        self._record(hook, context, outcome)
        return outcome

    def _record(
        self,
        hook: RegisteredHook,
        context: HookContext,
        outcome: HookOutcome,
        *,
        timed_out: bool = False,
    ) -> None:
        key = (hook.spec.name, context.event.value)
        stats = self._stats.get(key)
        if stats is None:
            stats = HookStats(hook_name=hook.spec.name, event=context.event)
            self._stats[key] = stats
        stats.hook_name = hook.spec.name
        stats.event = context.event
        stats.executed += 1
        stats.total_duration_ms += outcome.duration_ms
        if outcome.status is HookStatus.FAILED:
            stats.failed += 1
            stats.last_error = outcome.error
        if timed_out:
            stats.timed_out += 1
