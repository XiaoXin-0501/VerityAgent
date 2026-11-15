"""Agent 公共门面与流式句柄。

对外只有 ``create_agent`` / ``Agent`` / ``AgentStream``。内部 runtime、ports、
loop 不从这里导出。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

from agent.public import (
    AgentEvent,
    AgentRequest,
    AgentResult,
    AgentRunView,
)
from agent.runtime.enums import (
    AgentErrorCode,
    AgentEventType,
    CancelReason,
    RunStatus,
    TurnStatus,
)
from agent.runtime.errors import AgentRuntimeError, agent_error
from agent.runtime.events import AgentEventBuffer, AgentEventPublisher
from agent.runtime.loop import AgentLoop
from agent.runtime.ports import AgentDependencies
from agent.runtime.background import BackgroundTaskRunner
from config.agent import AgentConfig
from utils.ids import new_run_id, new_turn_id
from utils.time import utcnow

logger = logging.getLogger(__name__)

__all__ = ["Agent", "AgentStream", "create_agent", "AgentDependencies"]


@dataclass
class _ActiveRun:
    """本进程活跃/近期 run 的运行态与事件缓冲。"""

    run_id: str
    turn_id: str
    conversation_id: UUID
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    cancel_reason: CancelReason = CancelReason.USER
    buffer: AgentEventBuffer = field(default_factory=AgentEventBuffer)
    subscribers: set[asyncio.Queue[AgentEvent | None]] = field(default_factory=set)
    result: AgentResult | None = None
    result_future: asyncio.Future[AgentResult] | None = None
    task: asyncio.Task | None = None
    last_event: AgentEvent | None = None
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)


class Agent:
    """Agent 运行门面；依赖通过构造函数注入。"""

    def __init__(self, config: AgentConfig, dependencies: AgentDependencies) -> None:
        """保存配置、依赖与进程内 run 注册表。"""
        self.config = config
        self.deps = dependencies
        self._loop = AgentLoop(config, dependencies)
        self._runs: dict[str, _ActiveRun] = {}
        self._cancel_tasks: set[asyncio.Task] = set()
        self.background_tasks = (
            dependencies.background_tasks
            if dependencies.background_tasks is not None
            else BackgroundTaskRunner()
        )
        # runtime 端口与 facade 必须使用同一个 runner，避免 close 漏掉任务。
        dependencies.background_tasks = self.background_tasks
        self._closed = False

    async def run(self, request: AgentRequest) -> AgentResult:
        """执行并等待终态；正常失败也返回 AgentResult，不抛业务异常。"""
        self._ensure_open()
        self._prune_runs()
        active = self._new_active(request)
        self._runs[active.run_id] = active
        task = asyncio.create_task(
            self._execute(request, active),
            name=f"verity-agent-{active.run_id}",
        )
        active.task = task
        return await task

    def stream(self, request: AgentRequest) -> "AgentStream":
        """开始流式执行；返回事件异步迭代句柄。"""
        self._ensure_open()
        self._prune_runs()
        active = self._new_active(request)
        self._runs[active.run_id] = active
        queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue(
            maxsize=self.config.stream.max_buffer_events
        )
        active.subscribers.add(queue)
        task = asyncio.create_task(
            self._execute(request, active),
            name=f"verity-agent-{active.run_id}",
        )
        active.task = task
        return AgentStream(
            run_id=active.run_id,
            turn_id=active.turn_id,
            queue=queue,
            task=task,
            result_future=self._result_future(active),
        )

    def attach(
        self,
        run_id: str,
        last_event_id: str | None = None,
    ) -> "AgentStream":
        """连接已有 run：先回放缓冲，再订阅后续事件。"""
        self._ensure_open()
        self._prune_runs()
        active = self._runs.get(run_id)
        if active is None:
            raise AgentRuntimeError(
                agent_error(
                    AgentErrorCode.RUN_NOT_FOUND,
                    "run 不存在或已过期",
                    details={"run_id": run_id},
                )
            )
        queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue(
            maxsize=self.config.stream.max_buffer_events
        )
        for event in active.buffer.replay(last_event_id):
            queue.put_nowait(event)
        if active.result is not None:
            queue.put_nowait(None)
        else:
            active.subscribers.add(queue)
        return AgentStream(
            run_id=active.run_id,
            turn_id=active.turn_id,
            queue=queue,
            task=active.task,
            result_future=self._result_future(active),
        )

    async def cancel(
        self,
        run_id: str,
        reason: CancelReason = CancelReason.USER,
    ) -> None:
        """幂等取消；不存在/已终态 no-op。"""
        self._prune_runs()
        active = self._runs.get(run_id)
        if active is None or active.result is not None:
            return
        if active.cancel_event.is_set():
            return
        active.cancel_reason = reason
        active.cancel_event.set()
        await self.background_tasks.cancel_run(run_id)
        # 先给主循环一个收尾窗口；超过 cancel_grace_s 仍不结束则强制取消 task。
        if active.task is not None and not active.task.done():
            task = asyncio.create_task(
                self._cancel_after_grace(active),
                name=f"verity-agent-cancel-{run_id}",
            )
            self._cancel_tasks.add(task)
            task.add_done_callback(self._cancel_tasks.discard)

    async def get_run(self, run_id: str) -> AgentRunView | None:
        """读取运行视图；不存在或超过 run_state_ttl_s 返回 None。"""
        self._prune_runs()
        active = self._runs.get(run_id)
        if active is None:
            return None
        if active.result is not None:
            view = _view_from_result(active.result)
            return view.model_copy(
                update={
                    "created_at": active.created_at,
                    "updated_at": active.updated_at,
                    "last_event_id": (
                        active.last_event.id
                        if active.last_event is not None
                        else active.buffer.last_event_id
                    ),
                }
            )
        status = RunStatus.RUNNING
        turn_status = TurnStatus.CALLING_LLM
        if active.last_event is not None:
            if active.last_event.event_type in (
                AgentEventType.TOOL_STARTED,
                AgentEventType.TOOL_RESULT,
                AgentEventType.ASSISTANT_TOOL_CALLS,
            ):
                status, turn_status = RunStatus.WAITING_TOOL, TurnStatus.EXECUTING_TOOLS
            elif active.last_event.event_type is AgentEventType.TURN_STARTED:
                turn_status = TurnStatus.PREPARING_CONTEXT
        return AgentRunView(
            run_id=active.run_id,
            turn_id=active.turn_id,
            conversation_id=active.conversation_id,
            status=status,
            turn_status=turn_status,
            last_event_id=active.last_event.id if active.last_event else None,
            created_at=active.created_at,
            updated_at=active.updated_at,
        )

    async def close(self) -> None:
        """取消本实例活跃 run 并等待收尾；超过宽限期强制取消 task。"""
        if self._closed:
            return
        self._closed = True
        for active in self._runs.values():
            if active.result is None:
                active.cancel_event.set()
        tasks = [
            item.task
            for item in self._runs.values()
            if item.task is not None and not item.task.done()
        ]
        if tasks:
            _, pending = await asyncio.wait(
                tasks,
                timeout=self.config.cancel_grace_s,
            )
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        if self._cancel_tasks:
            await asyncio.gather(*list(self._cancel_tasks), return_exceptions=True)
        await self.background_tasks.close(self.config.cancel_grace_s)

    # ——————————————————— 内部 ———————————————————

    def _prune_runs(self) -> None:
        """淘汰超过 run_state_ttl_s 的终态 run，避免进程内注册表无界增长。"""
        ttl = max(1, int(self.config.run_state_ttl_s))
        now = utcnow()
        expired = [
            run_id
            for run_id, active in self._runs.items()
            if active.result is not None
            and (now - active.updated_at).total_seconds() > ttl
        ]
        for run_id in expired:
            self._runs.pop(run_id, None)

    async def _cancel_after_grace(self, active: _ActiveRun) -> None:
        """等待收尾宽限期；仍不结束则强制取消执行 task。"""
        task = active.task
        if task is None:
            return
        try:
            await asyncio.wait_for(
                asyncio.shield(task),
                timeout=self.config.cancel_grace_s,
            )
        except asyncio.TimeoutError:
            if not task.done():
                task.cancel()
        except asyncio.CancelledError:
            if not task.done():
                task.cancel()
            raise
        except Exception:  # noqa: BLE001 - 终态由 _execute 负责
            return

    def _ensure_open(self) -> None:
        """拒绝关闭后的新 run/stream/attach。"""
        if self._closed:
            raise AgentRuntimeError(
                agent_error(AgentErrorCode.INVALID_STATE, "Agent 已关闭")
            )

    def _new_active(self, request: AgentRequest) -> _ActiveRun:
        """创建一次 run 的本地注册项、事件缓冲和结果 future。"""
        return _ActiveRun(
            run_id=new_run_id(),
            turn_id=new_turn_id(),
            conversation_id=request.conversation_id,
            buffer=AgentEventBuffer(max_events=self.config.stream.replay_buffer_size),
            result_future=asyncio.get_running_loop().create_future(),
        )

    def _result_future(self, active: _ActiveRun) -> asyncio.Future[AgentResult]:
        """返回/复用 run 的 result future，避免 attach 创建第二个等待者。"""
        assert active.result_future is not None
        if active.result is not None and not active.result_future.done():
            active.result_future.set_result(active.result)
        return active.result_future

    async def _execute(self, request: AgentRequest, active: _ActiveRun) -> AgentResult:
        """运行 loop，并在终态广播 sentinel/保存 result。"""
        publisher = AgentEventPublisher(
            run_id=active.run_id,
            turn_id=active.turn_id,
            buffer=active.buffer,
            sink=self._publish(active),
        )
        try:
            result = await self._loop.run(
                request,
                run_id=active.run_id,
                turn_id=active.turn_id,
                publisher=publisher,
                cancel_event=active.cancel_event,
            )
        except asyncio.CancelledError:
            result = _cancelled_result(active)
            raise
        except Exception as exc:  # noqa: BLE001 - facade 兜底
            logger.exception("Agent 执行异常")
            result = _failed_result(active, exc)
        finally:
            # 如果 loop/任务在异常路径没有 result，这里也必须给 attach 一个终态。
            if "result" not in locals() or result is None:
                result = _failed_result(active, RuntimeError("agent run ended without result"))
            active.result = result
            active.updated_at = utcnow()
            if active.result_future is not None and not active.result_future.done():
                active.result_future.set_result(result)
            for queue in list(active.subscribers):
                _put_nowait(queue, None)
            active.subscribers.clear()
        return result

    def _publish(self, active: _ActiveRun):
        async def publish(event: AgentEvent) -> None:
            """更新最近事件并广播给所有 attach 的消费者。"""
            active.last_event = event
            active.updated_at = utcnow()
            for queue in list(active.subscribers):
                _put_nowait(queue, event)

        return publish


class AgentStream:
    """单消费者流式句柄；``result()`` 可直接等待终态。"""

    def __init__(
        self,
        *,
        run_id: str,
        turn_id: str,
        queue: asyncio.Queue[AgentEvent | None],
        task: asyncio.Task | None,
        result_future: asyncio.Future[AgentResult],
    ) -> None:
        """保存队列、任务与结果 future；不重复启动 run。"""
        self.run_id = run_id
        self.turn_id = turn_id
        self._queue = queue
        self._task = task
        self._result_future = result_future
        self._last_event_id: str | None = None
        self._finished = False

    @property
    def last_event_id(self) -> str | None:
        """最后一条已消费事件的 id。"""
        return self._last_event_id

    def __aiter__(self) -> "AgentStream":
        return self

    async def __anext__(self) -> AgentEvent:
        """逐条产出事件；队列收到 None 后结束。"""
        if self._finished:
            raise StopAsyncIteration
        item = await self._queue.get()
        if item is None:
            self._finished = True
            raise StopAsyncIteration
        self._last_event_id = item.id
        return item

    async def result(self) -> AgentResult:
        """等待终态结果；可在迭代前/后调用。"""
        if self._result_future.done():
            return self._result_future.result()
        # future 由 _execute 结束时显式 set_result；这里只等待，不重复消费队列。
        return await self._result_future


def _put_nowait(queue: asyncio.Queue, item) -> None:
    """非阻塞投递；慢消费者丢最旧事件，避免阻塞主循环。"""
    try:
        queue.put_nowait(item)
    except asyncio.QueueFull:
        try:
            queue.get_nowait()
            queue.put_nowait(item)
        except (asyncio.QueueEmpty, asyncio.QueueFull):
            pass


def _view_from_result(result: AgentResult) -> AgentRunView:
    """终态 AgentResult -> AgentRunView。"""
    assistant = result.assistant_message
    return AgentRunView(
        run_id=result.run_id,
        turn_id=result.turn_id,
        conversation_id=result.conversation_id,
        status=result.status,
        turn_status=(
            TurnStatus.COMPLETED
            if result.status is RunStatus.COMPLETED
            else TurnStatus.CANCELLED
            if result.status is RunStatus.CANCELLED
            else TurnStatus.FAILED
        ),
        user_message_id=result.user_message_id,
        assistant_message_id=assistant.id if assistant else None,
        assistant_status=assistant.status if assistant else None,
        content=assistant.content if assistant else "",
        thinking=list(assistant.thinking) if assistant else [],
        tool_calls=list(assistant.tool_calls) if assistant else [],
        usage=result.usage,
        result=result,
        error=result.error,
        created_at=utcnow(),
        updated_at=utcnow(),
    )


def _cancelled_result(active: _ActiveRun) -> AgentResult:
    """任务被外部取消时的兜底结果。"""
    return AgentResult(
        run_id=active.run_id,
        conversation_id=active.conversation_id,
        turn_id=active.turn_id,
        status=RunStatus.CANCELLED,
        user_message_id=UUID(int=0),
        error=agent_error(AgentErrorCode.CANCELLED, "运行已取消"),
    )


def _failed_result(active: _ActiveRun, exc: Exception) -> AgentResult:
    """未捕获异常兜底，保证每个 run 有终态。"""
    return AgentResult(
        run_id=active.run_id,
        conversation_id=active.conversation_id,
        turn_id=active.turn_id,
        status=RunStatus.FAILED,
        user_message_id=UUID(int=0),
        error=agent_error(
            AgentErrorCode.INTERNAL,
            "Agent 运行失败",
            cause=str(exc),
        ),
    )


async def create_agent(
    config: AgentConfig,
    *,
    dependencies: AgentDependencies | None = None,
) -> Agent:
    """创建 Agent；framework 阶段要求显式注入依赖。"""
    if dependencies is None:
        raise AgentRuntimeError(
            agent_error(
                AgentErrorCode.DEPENDENCY_UNAVAILABLE,
                "create_agent 需要注入 AgentDependencies（LLM/消息/租约端口）",
            )
        )
    return Agent(config, dependencies)
