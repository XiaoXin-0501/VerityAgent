"""Agent 公共门面与流式句柄。

对外只有 ``create_agent`` / ``Agent`` / ``AgentStream``。内部 runtime、ports、
loop 不从这里导出。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

from agent.public import (
    AgentEvent,
    AgentRequest,
    AgentResult,
    AgentRunView,
)
from agent.message.models import UserMessage, user_payload_matches
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
from utils.hashing import sha256_json
from utils.ids import new_turn_id, run_id_for
from utils.time import utcnow

logger = logging.getLogger(__name__)

__all__ = ["Agent", "AgentStream", "create_agent", "AgentDependencies"]


@dataclass
class _ActiveRun:
    """本进程活跃/近期 run 的运行态与事件缓冲。"""

    run_id: str
    turn_id: str
    conversation_id: UUID
    # 请求指纹；同一 client_message_id 重试时用于识别“同 id 不同参数”
    fingerprint: str = ""
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
        self.event_store = dependencies.event_store
        self.background_tasks = (
            dependencies.background_tasks
            if dependencies.background_tasks is not None
            else BackgroundTaskRunner()
        )
        # runtime 端口与 facade 必须使用同一个 runner，避免 close 漏掉任务。
        dependencies.background_tasks = self.background_tasks
        self._closed = False

    async def run(self, request: AgentRequest) -> AgentResult:
        """执行并等待终态；正常失败也返回 AgentResult，不抛业务异常。

        同一 ``(conversation_id, client_message_id)`` 重试时复用稳定 run id：
        本进程内直接等待已有结果，跨实例则轮询 Redis 快照等待终态。
        """
        self._ensure_open()
        self._prune_runs()
        run_id = run_id_for(request.conversation_id, request.client_message_id)
        existing = self._runs.get(run_id)
        if existing is not None:
            _assert_same_request(existing, request)
            if (
                existing.result is not None
                and existing.result.status
                in {RunStatus.CONFLICT, RunStatus.EXPIRED}
                and self.event_store is not None
            ):
                # 本实例只是租约竞争者；若持有者已在 Redis 写出结果，以其为准。
                snapshot = await self.event_store.snapshot(run_id)
                if snapshot is not None:
                    if snapshot.result is not None:
                        return snapshot.result
                    try:
                        return await self._await_remote_result(run_id)
                    except AgentRuntimeError:
                        # 远端快照过期；本实例至少还有冲突终态可返回。
                        assert existing.result is not None
                        return existing.result
            return await self._result_future(existing)
        if self.event_store is not None:
            snapshot = await self.event_store.snapshot(run_id)
            if snapshot is not None:
                # 跨实例快照没有请求指纹；以 PG 已落库的用户消息为准校验参数。
                await self._verify_persisted_request(request)
                if snapshot.result is not None:
                    return snapshot.result
                try:
                    return await self._await_remote_result(run_id)
                except AgentRuntimeError as exc:
                    return _remote_error_result(request, run_id, exc.error)
        active = self._new_active(request)
        self._runs[active.run_id] = active
        task = asyncio.create_task(
            self._execute(request, active),
            name=f"verity-agent-{active.run_id}",
        )
        active.task = task
        return await task

    def stream(self, request: AgentRequest) -> "AgentStream":
        """开始流式执行；返回事件异步迭代句柄。

        进程内同一幂等键重复调用会直接 attach 到已有 run，回放缓冲事件。
        跨实例重试由 HTTP 层先 ``get_run`` 判断；直接调用 ``stream`` 时不做
        Redis 同步查询，避免把公开的同步入口改成 async。
        """
        self._ensure_open()
        self._prune_runs()
        run_id = run_id_for(request.conversation_id, request.client_message_id)
        existing = self._runs.get(run_id)
        if existing is not None:
            _assert_same_request(existing, request)
            return self.attach(run_id)
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
            if self.event_store is None:
                raise AgentRuntimeError(
                    agent_error(
                        AgentErrorCode.RUN_NOT_FOUND,
                        "run 不存在或已过期",
                        details={"run_id": run_id},
                    )
                )
            return self._attach_remote(run_id, last_event_id)
        queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue(
            maxsize=self.config.stream.max_buffer_events
        )
        for event in active.buffer.replay(last_event_id):
            # replay_buffer_size 可能大于单连接队列；逐条回放时按“丢最旧”
            # 保持最新事件可消费，不能因为一次长时间 run 让 attach 抛
            # QueueFull。
            _put_nowait(queue, event)
        if active.result is not None:
            _put_nowait(queue, None)
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
        if active is None:
            if self.event_store is not None:
                await self.event_store.request_cancel(run_id, reason.value)
            return
        if active.result is not None:
            return
        if active.cancel_event.is_set():
            return
        active.cancel_reason = reason
        active.cancel_event.set()
        if self.event_store is not None:
            await self.event_store.request_cancel(run_id, reason.value)
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
            if self.event_store is not None:
                return await self.event_store.snapshot(run_id)
            return None
        if active.result is not None:
            if (
                active.result.status in {RunStatus.CONFLICT, RunStatus.EXPIRED}
                and self.event_store is not None
            ):
                # 租约竞争者优先返回持有者写出的真实终态。
                remote = await self.event_store.snapshot(run_id)
                if remote is not None and remote.result is not None:
                    return remote
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
        # 流程日志由进程级 handler 持有；关闭 Agent 时一并释放文件。
        flow_log = getattr(self.deps, "flow_log", None)
        close_flow = getattr(flow_log, "close", None)
        if callable(close_flow):
            try:
                close_flow()
            except Exception:  # noqa: BLE001 - 日志关闭失败不影响退出
                pass

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

    def _attach_remote(
        self,
        run_id: str,
        last_event_id: str | None,
    ) -> "AgentStream":
        """本地无 run 时，从 Redis Stream 回放并轮询终态。"""
        queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue(
            maxsize=self.config.stream.max_buffer_events
        )
        future = asyncio.get_running_loop().create_future()
        task = asyncio.create_task(
            self._remote_attach_loop(
                run_id,
                last_event_id,
                queue,
                future,
            ),
            name=f"verity-agent-remote-{run_id}",
        )
        return AgentStream(
            run_id=run_id,
            turn_id="",
            queue=queue,
            task=task,
            result_future=future,
        )

    async def _verify_persisted_request(self, request: AgentRequest) -> None:
        """校验同一幂等键的请求参数与已落库用户消息一致。"""
        existing = await self.deps.messages.find_by_client_message_id(
            request.conversation_id,
            request.client_message_id,
        )
        if existing is None:
            return
        message = existing.message
        if isinstance(message, UserMessage) and user_payload_matches(
            message,
            content=request.content,
            images=request.images,
            files=request.files,
        ):
            return
        raise AgentRuntimeError(
            agent_error(
                AgentErrorCode.IDEMPOTENCY_CONFLICT,
                "同一 client_message_id 已用于不同内容或参数",
                details={"client_message_id": str(request.client_message_id)},
            )
        )

    async def _await_remote_result(self, run_id: str) -> AgentResult:
        """轮询 Redis 快照，等待其他实例上的同一 run 进入终态。"""
        if self.event_store is None:
            raise AgentRuntimeError(
                agent_error(AgentErrorCode.RUN_NOT_FOUND, "run 不存在或已过期")
            )
        while True:
            snapshot = await self.event_store.snapshot(run_id)
            if snapshot is None:
                raise AgentRuntimeError(
                    agent_error(
                        AgentErrorCode.RUN_NOT_FOUND,
                        "run 不存在或已过期",
                        details={"run_id": run_id},
                    )
                )
            if snapshot.result is not None:
                return snapshot.result
            if snapshot.status.value in _TERMINAL_RUN_STATUSES:
                raise AgentRuntimeError(
                    agent_error(
                        AgentErrorCode.RUN_NOT_FOUND,
                        "run 已进入终态但结果不可用",
                        details={"run_id": run_id, "status": snapshot.status.value},
                    )
                )
            await asyncio.sleep(self.config.remote_event_poll_interval_s)

    async def _remote_attach_loop(
        self,
        run_id: str,
        last_event_id: str | None,
        queue: asyncio.Queue[AgentEvent | None],
        future: asyncio.Future[AgentResult],
    ) -> None:
        """轮询 Redis 快照/事件，直到收到终态。"""
        if self.event_store is None:
            if not future.done():
                future.set_exception(
                    AgentRuntimeError(
                        agent_error(AgentErrorCode.RUN_NOT_FOUND, "run 不存在")
                    )
                )
            _put_nowait(queue, None)
            return
        last_seq = _event_seq(last_event_id)
        try:
            while True:
                events = await self.event_store.replay(
                    run_id,
                    f"{run_id}:{last_seq}" if last_seq else None,
                )
                for event in events:
                    if event.seq <= last_seq:
                        continue
                    last_seq = event.seq
                    _put_nowait(queue, event)
                snapshot = await self.event_store.snapshot(run_id)
                if snapshot is None:
                    raise AgentRuntimeError(
                        agent_error(
                            AgentErrorCode.RUN_NOT_FOUND,
                            "run 不存在或已过期",
                            details={"run_id": run_id},
                        )
                    )
                if snapshot.status.value in {
                    "completed",
                    "failed",
                    "cancelled",
                    "expired",
                    "conflict",
                }:
                    if snapshot.result is not None:
                        future.set_result(snapshot.result)
                    else:
                        future.set_exception(
                            AgentRuntimeError(
                                agent_error(
                                    AgentErrorCode.RUN_NOT_FOUND,
                                    "run 终态结果不可用",
                                    details={"run_id": run_id},
                                )
                            )
                        )
                    _put_nowait(queue, None)
                    return
                await asyncio.sleep(self.config.remote_event_poll_interval_s)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - 远程流失败要收敛
            if not future.done():
                future.set_exception(exc)
            _put_nowait(queue, None)

    async def _save_event_store_run(
        self,
        active: _ActiveRun,
        status: str,
        result: AgentResult | None = None,
    ) -> None:
        """写 Redis run 快照；失败由 event_store 自行降级。"""
        if self.event_store is None:
            return
        await self.event_store.save_run(
            run_id=active.run_id,
            conversation_id=active.conversation_id,
            turn_id=active.turn_id,
            status=status,
            result=result,
        )

    async def _watch_remote_cancel(self, active: _ActiveRun) -> None:
        """优先订阅 Redis Pub/Sub，失败后退回轮询取消标记。"""
        if self.event_store is None:
            return
        watch: Any = getattr(self.event_store, "watch_cancel", None)
        if callable(watch):
            try:
                async for _ in watch(active.run_id):  # pyright: ignore[reportGeneralTypeIssues]
                    active.cancel_event.set()
                    return
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - 订阅失败退回轮询
                logger.debug("取消 Pub/Sub 监听失败，退回轮询", exc_info=True)
        while active.result is None and not active.cancel_event.is_set():
            try:
                if await self.event_store.cancel_requested(active.run_id):
                    active.cancel_event.set()
                    return
            except Exception:  # noqa: BLE001 - 观测失败不影响 run
                return
            await asyncio.sleep(self.config.remote_cancel_poll_interval_s)

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
            # 稳定 run id：同一幂等键的重试命中同一 run，而不是重跑。
            run_id=run_id_for(request.conversation_id, request.client_message_id),
            turn_id=new_turn_id(),
            conversation_id=request.conversation_id,
            fingerprint=_request_fingerprint(request),
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
        """抢占租约、运行 loop，并在终态广播 sentinel/保存 result。

        租约在 facade 预抢而不是等 loop 内部抢：同一稳定 run id 的并发请求
        只有一个实例能写事件/快照，避免重复 seq 污染 Redis Stream。
        """
        publisher = AgentEventPublisher(
            run_id=active.run_id,
            turn_id=active.turn_id,
            buffer=active.buffer,
            sink=self._publish(active),
        )
        cancel_watch: asyncio.Task | None = None
        result: AgentResult | None = None
        acquired = False
        loop_started = False
        try:
            acquired = await self.deps.leases.acquire(
                active.conversation_id,
                active.run_id,
                self.config.policy.run_lease_seconds,
            )
            if not acquired:
                result = _conflict_result(active)
                # 让流式消费者收到本地终态事件；不写事件存储，避免污染持有者。
                await self._emit_local_terminal(active, result)
            else:
                cancel_watch = (
                    asyncio.create_task(
                        self._watch_remote_cancel(active),
                        name=f"verity-agent-cancel-watch-{active.run_id}",
                    )
                    if self.event_store is not None
                    else None
                )
                await self._save_event_store_run(active, "running")
                loop_started = True
                result = await self._loop.run(
                    request,
                    run_id=active.run_id,
                    turn_id=active.turn_id,
                    publisher=publisher,
                    cancel_event=active.cancel_event,
                    lease_acquired=True,
                )
        except asyncio.CancelledError:
            result = _cancelled_result(active)
            raise
        except Exception as exc:  # noqa: BLE001 - facade 兜底
            logger.exception("Agent 执行异常")
            result = _failed_result(active, exc)
        finally:
            # 如果 loop/任务在异常路径没有 result，这里也必须给 attach 一个终态。
            if result is None:
                result = _failed_result(
                    active,
                    RuntimeError("agent run ended without result"),
                )
            assert result is not None
            if cancel_watch is not None:
                cancel_watch.cancel()
                try:
                    await cancel_watch
                except asyncio.CancelledError:
                    pass
            active.result = result
            active.updated_at = utcnow()
            if acquired and not loop_started:
                # 预抢租约后、进入 loop 前失败的路径需要在这里释放；loop 内部
                # 的 release 幂等，进入 loop 后就不再重复释放。
                try:
                    await self.deps.leases.release(
                        active.conversation_id,
                        active.run_id,
                        result.status.value,
                    )
                except Exception:  # noqa: BLE001 - 释放失败不能覆盖 result
                    logger.warning("释放 run 租约失败: %s", active.run_id, exc_info=True)
            if acquired:
                # 未抢到租约的实例不能覆盖租约持有者的事件快照。
                try:
                    await self._save_event_store_run(
                        active,
                        result.status.value,
                        result=result,
                    )
                except asyncio.CancelledError:
                    # 强制取消时仍要先保证 result future 被 set，避免 attach 永久等待。
                    pass
                except Exception:  # noqa: BLE001 - 快照写入失败不能覆盖 result
                    logger.warning(
                        "写入 run 终态快照失败: %s", active.run_id, exc_info=True
                    )
            if active.result_future is not None and not active.result_future.done():
                active.result_future.set_result(result)
            for queue in list(active.subscribers):
                _put_nowait(queue, None)
            active.subscribers.clear()
        assert result is not None
        return result

    async def _emit_local_terminal(
        self,
        active: _ActiveRun,
        result: AgentResult,
    ) -> None:
        """只写本地缓冲/订阅者的终态事件；用于租约冲突等非拥有者路径。"""
        publisher = AgentEventPublisher(
            run_id=active.run_id,
            turn_id=active.turn_id,
            buffer=active.buffer,
            sink=None,
        )
        error = (
            result.error.model_dump(mode="json")
            if result.error is not None
            else {}
        )
        failed = await publisher.emit(AgentEventType.RUN_FAILED, error)
        done = await publisher.emit(
            AgentEventType.DONE,
            {"run_id": active.run_id, "status": result.status.value},
        )
        for event in (failed, done):
            active.last_event = event
            for queue in list(active.subscribers):
                _put_nowait(queue, event)

    def _publish(self, active: _ActiveRun):
        """构造发布协程：更新运行快照、写事件存储并广播给订阅者。"""
        async def publish(event: AgentEvent) -> None:
            """更新最近事件、写 Redis Stream 并广播给 attach 的消费者。"""
            active.last_event = event
            active.updated_at = utcnow()
            if self.event_store is not None:
                await self.event_store.append_event(event)
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
        """返回自身作为异步迭代器。"""
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


_TERMINAL_RUN_STATUSES = {
    RunStatus.COMPLETED.value,
    RunStatus.FAILED.value,
    RunStatus.CANCELLED.value,
    RunStatus.EXPIRED.value,
    RunStatus.CONFLICT.value,
}


def _request_fingerprint(request: AgentRequest) -> str:
    """计算影响执行结果的请求指纹；用于识别同幂等键的不同参数。"""
    rag_mode = getattr(request.rag_mode, "value", request.rag_mode)
    return sha256_json(
        {
            "content": request.content,
            "images": [item.model_dump(mode="json") for item in request.images],
            "files": [item.model_dump(mode="json") for item in request.files],
            "model_id": request.model_id,
            "rag_mode": rag_mode,
            "tool_choice": (
                request.tool_choice.model_dump(mode="json")
                if request.tool_choice is not None
                else None
            ),
            "max_tool_rounds": request.max_tool_rounds,
            # INJECTED RAG 从 metadata 读取的检索参数也会影响结果。
            "rag_kb_ids": request.metadata.get("kb_ids"),
            "rag_top_k": request.metadata.get("top_k"),
        }
    )


def _assert_same_request(active: _ActiveRun, request: AgentRequest) -> None:
    """同一幂等键重试时校验参数一致；不一致返回 IDEMPOTENCY_CONFLICT。"""
    if active.fingerprint and active.fingerprint != _request_fingerprint(request):
        raise AgentRuntimeError(
            agent_error(
                AgentErrorCode.IDEMPOTENCY_CONFLICT,
                "同一 client_message_id 已用于不同内容或参数",
                details={"client_message_id": str(request.client_message_id)},
            )
        )


def _remote_error_result(
    request: AgentRequest,
    run_id: str,
    error,
) -> AgentResult:
    """远端快照过期时返回失败结果，遵守 run 不抛业务异常的契约。"""
    return AgentResult(
        run_id=run_id,
        conversation_id=request.conversation_id,
        turn_id="",
        status=RunStatus.FAILED,
        user_message_id=UUID(int=0),
        error=error,
    )


def _event_seq(event_id: str | None) -> int:
    """从 ``run_id:seq`` 解析序号；非法值按 0 处理。"""
    if not event_id:
        return 0
    try:
        return int(str(event_id).rsplit(":", 1)[-1])
    except (TypeError, ValueError):
        return 0


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


def _conflict_result(active: _ActiveRun) -> AgentResult:
    """租约被其他实例/run 占用时的终态；不写事件存储，避免污染持有者。"""
    return AgentResult(
        run_id=active.run_id,
        conversation_id=active.conversation_id,
        turn_id=active.turn_id,
        status=RunStatus.CONFLICT,
        user_message_id=UUID(int=0),
        error=agent_error(
            AgentErrorCode.LEASE_CONFLICT,
            "会话已有活跃 run",
            details={"conversation_id": str(active.conversation_id)},
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
