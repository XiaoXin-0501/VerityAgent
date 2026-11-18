"""Agent 主循环（v1 framework）。

当前落地范围：

- 会话租约、用户/assistant/tool 消息落库、幂等回合重放；
- 非流式/流式 LLM 调用、工具轮、usage/结果组装；
- ContextAssembler + PromptBuilder、记忆 ready、回合后压缩后台任务；
- 事件发布与 Run/Turn 终态；Redis 事件缓冲、跨实例取消由 facade 负责；
- 端口全部可注入，测试不依赖 PG/Redis/Provider。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from datetime import timedelta
from time import perf_counter
from typing import TYPE_CHECKING, Any, Coroutine
from uuid import UUID

from agent.context.models import ContextSection
from agent.llm import (
    FilePart,
    FinishReason,
    GenerationInfo,
    ImagePart,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMRole,
    LLMStreamPhase,
    LLMToolCall,
    LLMToolDescriptor,
    LLMUsage,
    ModelPurpose,
    TextPart,
    ThinkingPart,
)
from agent.message.enum import MessageStatus
from agent.message.models import (
    AssistantMessage,
    RawMessage,
    StoredMessage,
    Thinking,
    ToolCall,
    ToolMessage,
    UserMessage,
    user_payload_matches,
)
from agent.prompting import PromptBuilder
from agent.runtime.enums import (
    AgentErrorCode,
    AgentEventType,
    RunStatus,
    TurnStatus,
)
from agent.runtime.errors import AgentRuntimeError, agent_error
from agent.runtime.events import AgentEventPublisher
from agent.runtime.models import AgentLimits, AgentWarningData
from agent.runtime.ports import AgentDependencies
from agent.tools.enums import ToolResultStatus
from agent.tools.models import ToolContext, ToolResult
from config.agent import AgentConfig
from exception import ConfigurationError
from utils.ids import uuid7
from utils.log_context import bind_run_context, reset_run_context
from utils.log_summary import summary_block
from utils.time import utcnow

if TYPE_CHECKING:
    from agent.public import AgentRequest, AgentResult

logger = logging.getLogger(__name__)

__all__ = ["AgentLoop"]


class AgentLoop:
    """把模型、工具、消息端口串成一个可测试的主循环。"""

    def __init__(self, config: AgentConfig, dependencies: AgentDependencies) -> None:
        """保存配置与端口；不建立外部连接。"""
        self.config = config
        self.deps = dependencies
        self.prompt_builder = PromptBuilder(default_model_id=dependencies.model_id)

    async def run(
        self,
        request: AgentRequest,
        *,
        run_id: str,
        turn_id: str,
        publisher: AgentEventPublisher,
        cancel_event: asyncio.Event,
        lease_acquired: bool = False,
    ) -> AgentResult:
        """执行一次 run，返回公共 AgentResult。"""
        from agent.public import AgentResult

        started = utcnow()
        usage = LLMUsage()
        warnings: list[AgentWarningData] = []
        message_ids: list[UUID] = []
        final_assistant: AssistantMessage | None = None
        llm_calls = 0
        tool_rounds = 0
        conversation_id = request.conversation_id
        # 日志上下文：后续所有 LLM/工具/RAG 日志都带同一个 run/conv 短 id。
        log_tokens = bind_run_context(
            run_id=run_id,
            conversation_id=str(conversation_id),
        )
        logger.info(
            "Agent run 开始: content_chars=%d images=%d files=%d "
            "client_message_id=%s",
            len(request.content or ""),
            len(request.images),
            len(request.files),
            request.client_message_id,
        )
        # 心跳续租失败与用户取消分开标记：前者终态是 EXPIRED，后者是 CANCELLED。
        lease_lost = asyncio.Event()
        # 提前初始化，避免异常路径依赖 locals() 判断；这些值在 finally 中一定会用到。
        user_stored: Any = None
        heartbeat_task: asyncio.Task | None = None
        result: AgentResult | None = None

        async def emit(event_type: AgentEventType, data=None, **kwargs):
            """构造绑定当前 publisher 的事件发送协程。"""
            return await publisher.emit(event_type, data or {}, **kwargs)

        turn_terminal_emitted = False

        async def emit_turn_terminal(status: TurnStatus, **extra) -> None:
            """发送一次 turn 终态；异常路径与成功路径共用，避免漏发/重发。"""
            nonlocal turn_terminal_emitted
            if turn_terminal_emitted:
                return
            turn_terminal_emitted = True
            event_type = {
                TurnStatus.COMPLETED: AgentEventType.TURN_COMPLETED,
                TurnStatus.CANCELLED: AgentEventType.TURN_CANCELLED,
            }.get(status, AgentEventType.TURN_FAILED)
            payload = {"turn_id": turn_id, "status": status.value, **extra}
            await emit(event_type, payload)

        try:
            existing = await self.deps.messages.find_by_client_message_id(
                conversation_id,
                request.client_message_id,
            )
            if existing is not None:
                replayed = await self._replay_existing_turn(
                    existing,
                    request,
                    run_id=run_id,
                    turn_id=turn_id,
                )
                if replayed is None:
                    raise AgentRuntimeError(
                        agent_error(
                            AgentErrorCode.IDEMPOTENCY_CONFLICT,
                            "client_message_id 已用于不同内容，或原回合无法安全重放",
                            details={
                                "client_message_id": str(request.client_message_id)
                            },
                        )
                    )
                # 幂等命中：不重复执行，只把已有结果补成完整事件对。
                result = replayed
                await emit(
                    AgentEventType.RUN_COMPLETED,
                    replayed.model_dump(mode="json"),
                )
                await emit(
                    AgentEventType.DONE,
                    {"run_id": run_id, "status": RunStatus.COMPLETED.value},
                )
                return replayed

            # facade 已预抢租约时直接进入主循环；直接调用 AgentLoop 的调用方
            # 仍在这里抢占，保证单测/自定义装配可用。
            if not lease_acquired:
                acquired = await self.deps.leases.acquire(
                    conversation_id,
                    run_id,
                    self.config.policy.run_lease_seconds,
                )
                if not acquired:
                    raise AgentRuntimeError(
                        agent_error(
                            AgentErrorCode.LEASE_CONFLICT,
                            "会话已有活跃 run",
                            details={"conversation_id": str(conversation_id)},
                        )
                    )

            heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(
                    conversation_id,
                    run_id,
                    cancel_event,
                    lease_lost,
                ),
                name=f"verity-agent-heartbeat-{run_id}",
            )
            await emit(
                AgentEventType.RUN_STARTED,
                {
                    "run_id": run_id,
                    "turn_id": turn_id,
                    "conversation_id": str(conversation_id),
                },
            )
            # 轮次 id：从本次用户提问到最终回答的所有消息共用；摘要 topic
            # 的 source_round_ids 与轮次召回工具都以它为锚点。
            round_id = uuid7()
            await emit(
                AgentEventType.TURN_STARTED,
                {
                    "turn_id": turn_id,
                    "round_id": str(round_id),
                    "conversation_id": str(conversation_id),
                },
            )
            await self._emit_hook(
                "run.start",
                {"run_id": run_id, "status": "running"},
                request=request,
                run_id=run_id,
                turn_id=turn_id,
            )
            await self._emit_hook(
                "turn.start",
                {"turn_id": turn_id, "status": "calling_llm"},
                request=request,
                run_id=run_id,
                turn_id=turn_id,
            )

            user_message = UserMessage(
                conversation_id=conversation_id,
                content=request.content,
                images=list(request.images),
                files=list(request.files),
            )
            user_stored = await self.deps.messages.append_user(
                user_message,
                client_message_id=request.client_message_id,
                # 队列已接入时，用户消息与 user_messages 暂存行同事务写入；
                # 回合终态只需 mark_ready，崩溃也不会漏抽取。
                user_id=(
                    request.user_id if self.deps.memory_queue is not None else None
                ),
                round_id=round_id,
            )
            message_ids.append(user_stored.message.id)
            await self._emit_hook(
                "message.appended",
                {
                    "message_id": str(user_stored.message.id),
                    "role": "user",
                    "status": "completed",
                    "content_bytes": _content_bytes(user_stored.message),
                    "image_count": len(request.images),
                    "file_count": len(request.files),
                },
                request=request,
                run_id=run_id,
                turn_id=turn_id,
            )

            history: list[StoredMessage] = await self.deps.messages.load_window(
                conversation_id,
                limit=self.deps.history_limit,
            )
            # 某些适配器按最近 N 条读取；确保本轮用户消息在窗口里。
            if not any(item.message.id == user_message.id for item in history):
                history.append(user_stored)

            limits = self._limits(request)
            # 轮次召回工具只在摘要压缩到阈值后暴露：工具已在注册表里，
            # 这里按会话压缩次数动态追加权限码，不修改全局注册表。
            effective_permissions = await self._tool_permissions(
                request.permissions,
                conversation_id,
            )
            tool_selection = (
                self.deps.tools.select(effective_permissions)
                if self.deps.tools is not None
                else None
            )
            tools = self._tool_descriptors(tool_selection)
            llm_messages = await self._assemble_context(
                history,
                request,
                tools,
                run_id=run_id,
                turn_id=turn_id,
            )

            while True:
                self._check_runtime(limits, cancel_event, lease_lost)
                if llm_calls >= limits.max_llm_calls:
                    raise AgentRuntimeError(
                        agent_error(
                            AgentErrorCode.TOOL_ROUND_LIMIT,
                            "LLM 调用次数达到上限",
                            details={"max_llm_calls": limits.max_llm_calls},
                        )
                    )
                llm_calls += 1
                assistant_id = uuid7()
                await emit(
                    AgentEventType.ASSISTANT_STARTED,
                    {
                        "message_id": str(assistant_id),
                        "provider": self.deps.model_id,
                    },
                )
                llm_request = self._llm_request(
                    request,
                    run_id=run_id,
                    turn_id=turn_id,
                    messages=llm_messages,
                    tools=tools,
                    deadline=limits.deadline,
                )
                use_stream = bool(getattr(request, "stream", False)) and hasattr(
                    self.deps.llm, "stream"
                )
                if use_stream:
                    llm_request = llm_request.model_copy(update={"stream": True})
                await self._emit_hook(
                    "llm.request.start",
                    {
                        "provider": self.deps.model_id,
                        "model": llm_request.model_id,
                        "purpose": llm_request.context.purpose.value,
                        "message_count": len(llm_request.messages),
                        "tool_count": len(llm_request.tools),
                        "stream": use_stream,
                    },
                    request=request,
                    run_id=run_id,
                    turn_id=turn_id,
                    call_id=llm_request.request_id,
                )
                remaining = max(
                    0.01,
                    (limits.deadline - utcnow()).total_seconds(),
                )
                try:
                    if use_stream:
                        response = await self._stream_response(
                            llm_request,
                            request=request,
                            run_id=run_id,
                            turn_id=turn_id,
                            assistant_id=assistant_id,
                            publisher=publisher,
                            deadline=limits.deadline,
                            cancel_event=cancel_event,
                        )
                    else:
                        response = await asyncio.wait_for(
                            self.deps.llm.chat(llm_request),
                            timeout=remaining,
                        )
                except asyncio.TimeoutError as exc:
                    await self._emit_hook(
                        "llm.error",
                        {
                            "error_code": AgentErrorCode.TIMEOUT.value,
                            "retryable": True,
                            "model": llm_request.model_id,
                        },
                        request=request,
                        run_id=run_id,
                        turn_id=turn_id,
                        call_id=llm_request.request_id,
                    )
                    raise AgentRuntimeError(
                        agent_error(
                            AgentErrorCode.TIMEOUT,
                            "LLM 调用超过回合 deadline",
                            retryable=True,
                            details={"model_id": self.deps.model_id},
                        )
                    ) from exc
                except ConfigurationError as exc:
                    await self._emit_hook(
                        "llm.error",
                        {
                            "error_code": AgentErrorCode.INVALID_CONFIG.value,
                            "retryable": False,
                            "model": llm_request.model_id,
                        },
                        request=request,
                        run_id=run_id,
                        turn_id=turn_id,
                        call_id=llm_request.request_id,
                    )
                    raise AgentRuntimeError(
                        agent_error(
                            AgentErrorCode.INVALID_CONFIG,
                            "模型配置无效或请求模型不可用",
                            details={"model_id": llm_request.model_id},
                            cause=str(exc),
                        )
                    ) from exc
                except AgentRuntimeError:
                    raise
                except Exception as exc:  # noqa: BLE001 - 端口异常收敛
                    await self._emit_hook(
                        "llm.error",
                        {
                            "error_code": AgentErrorCode.MODEL_UNAVAILABLE.value,
                            "retryable": True,
                            "model": llm_request.model_id,
                            "cause": str(exc),
                        },
                        request=request,
                        run_id=run_id,
                        turn_id=turn_id,
                        call_id=llm_request.request_id,
                    )
                    raise AgentRuntimeError(
                        agent_error(
                            AgentErrorCode.MODEL_UNAVAILABLE,
                            "模型调用失败",
                            retryable=True,
                            details={"model_id": self.deps.model_id},
                            cause=str(exc),
                        )
                    ) from exc

                assistant = self._assistant_message(
                    response,
                    conversation_id=conversation_id,
                    assistant_id=assistant_id,
                )
                assistant_stored = await self.deps.messages.append_assistant(
                    assistant,
                    round_id=round_id,
                )
                message_ids.append(assistant_stored.message.id)
                await self._emit_hook(
                    "message.appended",
                    {
                        "message_id": str(assistant.id),
                        "role": "assistant",
                        "status": assistant.status.value,
                        "content_bytes": _content_bytes(assistant),
                        "thinking_count": len(assistant.thinking),
                        "tool_call_count": len(assistant.tool_calls),
                    },
                    request=request,
                    run_id=run_id,
                    turn_id=turn_id,
                    call_id=assistant.id.hex,
                )
                # 工具轮的下一次 LLM 请求必须包含 assistant(tool_calls)；否则
                # tool 结果前面缺少调用来源，OpenAI/Anthropic 协议不成立。
                assistant_llm = _to_llm_message(assistant)
                if assistant_llm is not None:
                    llm_messages.append(assistant_llm)
                usage = _add_usage(usage, response.usage)
                final_assistant = assistant
                await emit(
                    AgentEventType.ASSISTANT_COMPLETED,
                    {
                        "message_id": str(assistant.id),
                        "content": assistant.content,
                        "thinking": [item.text for item in assistant.thinking],
                        "tool_calls": [
                            call.model_dump(mode="json")
                            for call in assistant.tool_calls
                        ],
                        "finish_reason": response.finish_reason.value,
                        "usage": response.usage.model_dump(mode="json"),
                    },
                    message_id=assistant.id,
                )
                await emit(
                    AgentEventType.USAGE,
                    {
                        "call_id": response.request_id,
                        "usage": response.usage.model_dump(mode="json"),
                    },
                )
                await self._emit_hook(
                    "llm.usage",
                    {
                        "provider": response.provider,
                        "model": response.model,
                        "model_id": llm_request.model_id,
                        "usage": response.usage.model_dump(mode="json"),
                        "purpose": llm_request.context.purpose.value,
                        "finish_reason": response.finish_reason.value,
                        "call_id": response.request_id,
                        "latency_ms": response.latency_ms,
                    },
                    request=request,
                    run_id=run_id,
                    turn_id=turn_id,
                    call_id=response.request_id,
                    duration_ms=response.latency_ms,
                )
                await self._emit_hook(
                    "llm.request.end",
                    {
                        "provider": response.provider,
                        "model": response.model,
                        "purpose": llm_request.context.purpose.value,
                        "finish_reason": response.finish_reason.value,
                        "usage": response.usage.model_dump(mode="json"),
                    },
                    request=request,
                    run_id=run_id,
                    turn_id=turn_id,
                    call_id=response.request_id,
                    duration_ms=response.latency_ms,
                )

                if response.finish_reason is FinishReason.CONTENT_FILTER:
                    raise AgentRuntimeError(
                        agent_error(
                            AgentErrorCode.MODEL_CONTENT_FILTERED,
                            "模型输出被内容策略拦截",
                            details={"model_id": self.deps.model_id},
                        )
                    )
                if response.finish_reason is FinishReason.ERROR:
                    raise AgentRuntimeError(
                        agent_error(
                            AgentErrorCode.MODEL_UNAVAILABLE,
                            "模型返回错误终态",
                            retryable=True,
                            details={"model_id": self.deps.model_id},
                        )
                    )

                if (
                    response.finish_reason is FinishReason.TOOL_CALLS
                    and assistant.tool_calls
                ):
                    if tool_rounds >= limits.max_tool_rounds:
                        raise AgentRuntimeError(
                            agent_error(
                                AgentErrorCode.TOOL_ROUND_LIMIT,
                                "工具轮数达到上限",
                                details={"max_tool_rounds": limits.max_tool_rounds},
                            )
                        )
                    tool_rounds += 1
                    await emit(
                        AgentEventType.ASSISTANT_TOOL_CALLS,
                        {
                            "message_id": str(assistant.id),
                            "tool_calls": [
                                call.model_dump(mode="json")
                                for call in assistant.tool_calls
                            ],
                        },
                        message_id=assistant.id,
                    )
                    tool_messages = await self._execute_tools(
                        assistant,
                        request=request,
                        permissions=effective_permissions,
                        run_id=run_id,
                        turn_id=turn_id,
                        conversation_id=conversation_id,
                        deadline=limits.deadline,
                        publisher=publisher,
                        cancel_event=cancel_event,
                    )
                    for tool_message in tool_messages:
                        stored = await self.deps.messages.append_tool(
                            tool_message,
                            round_id=round_id,
                        )
                        message_ids.append(stored.message.id)
                        converted = _to_llm_message(stored.message)
                        if converted is not None:
                            llm_messages.append(converted)
                        tool_call_id = getattr(stored.message, "tool_call_id", None)
                        await self._emit_hook(
                            "message.appended",
                            {
                                "message_id": str(stored.message.id),
                                "role": "tool",
                                "status": stored.message.status.value,
                                "content_bytes": _content_bytes(stored.message),
                                "tool_call_id": tool_call_id,
                            },
                            request=request,
                            run_id=run_id,
                            turn_id=turn_id,
                            call_id=tool_call_id,
                        )
                    if tool_selection is not None and self.deps.tools is not None:
                        # router 工具调用后展开组内工具；非 router 调用是 no-op。
                        activate = getattr(self.deps.tools, "activate_group", None)
                        if callable(activate):
                            for call in assistant.tool_calls:
                                try:
                                    tool_selection = activate(
                                        tool_selection,
                                        call.name,
                                        effective_permissions,
                                    )
                                except Exception:  # noqa: BLE001 - 路由失败不阻断回合
                                    logger.warning(
                                        "工具路由激活失败: %s",
                                        call.name,
                                        exc_info=True,
                                    )
                        tools = self._tool_descriptors(tool_selection)
                    continue

                break

            if lease_lost.is_set():
                raise AgentRuntimeError(
                    agent_error(
                        AgentErrorCode.LEASE_LOST,
                        "会话 run 租约已丢失",
                        retryable=False,
                        details={"conversation_id": str(conversation_id)},
                    )
                )
            if cancel_event.is_set():
                raise AgentRuntimeError(
                    agent_error(AgentErrorCode.CANCELLED, "运行已取消")
                )
            duration = int((utcnow() - started).total_seconds() * 1000)
            result = AgentResult(
                run_id=run_id,
                conversation_id=conversation_id,
                turn_id=turn_id,
                status=RunStatus.COMPLETED,
                user_message_id=user_stored.message.id,
                assistant_message=final_assistant,
                message_ids=message_ids,
                usage=usage,
                warnings=[_public_warning(item) for item in warnings],
                duration_ms=duration,
            )
            if self.deps.memory_queue is not None:
                # 入队已在 append_user 的事务中完成；这里只把本会话暂存行置 ready。
                self._schedule_background(
                    "memory.mark_ready",
                    self._mark_memory_ready(conversation_id),
                    run_id=run_id,
                )
            if self.deps.compressor is not None:
                self._schedule_background(
                    "memory.compress",
                    self.deps.compressor.maybe_compress(
                        conversation_id,
                        run_id=run_id,
                    ),
                    run_id=run_id,
                )
            await emit_turn_terminal(
                TurnStatus.COMPLETED,
                assistant_message_id=(
                    str(final_assistant.id) if final_assistant is not None else None
                ),
                usage=usage.model_dump(mode="json"),
            )
            await self._emit_hook(
                "turn.end",
                {
                    "status": "completed",
                    "usage": usage.model_dump(mode="json"),
                    "message_count": len(message_ids),
                },
                request=request,
                run_id=run_id,
                turn_id=turn_id,
                duration_ms=duration,
            )
            await self._emit_hook(
                "run.end",
                {
                    "status": "completed",
                    "usage": usage.model_dump(mode="json"),
                    "duration_ms": duration,
                },
                request=request,
                run_id=run_id,
                turn_id=turn_id,
                duration_ms=duration,
            )
            await emit(
                AgentEventType.RUN_COMPLETED,
                result.model_dump(mode="json"),
            )
            await emit(AgentEventType.DONE, {"run_id": run_id, "status": "completed"})
            logger.info(
                "%s",
                summary_block(
                    "Agent 完成",
                    [
                        f"status=completed | 耗时={duration}ms | "
                        f"消息={len(message_ids)} | LLM={llm_calls} 次 | "
                        f"工具轮={tool_rounds}",
                        f"tokens: 输入={usage.input_tokens} "
                        f"输出={usage.output_tokens} "
                        f"缓存输入={usage.cached_input_tokens} "
                        f"总计={usage.total_tokens}",
                        f"warnings={len(warnings)}",
                    ],
                ),
            )
            return result
        except AgentRuntimeError as exc:
            duration = int((utcnow() - started).total_seconds() * 1000)
            status = _run_status_for_error(exc.error.code)
            result = AgentResult(
                run_id=run_id,
                conversation_id=conversation_id,
                turn_id=turn_id,
                status=status,
                user_message_id=(
                    user_stored.message.id if user_stored is not None else UUID(int=0)
                ),
                assistant_message=final_assistant,
                message_ids=message_ids,
                usage=usage,
                warnings=[_public_warning(item) for item in warnings],
                error=exc.error,
                duration_ms=duration,
            )
            logger.warning(
                "%s",
                summary_block(
                    "Agent 失败",
                    [
                        f"status={status.value} | 耗时={duration}ms | "
                        f"error={exc.error.code.value}",
                        f"消息={len(message_ids)} | LLM={llm_calls} 次 | "
                        f"工具轮={tool_rounds}",
                        f"tokens: 输入={usage.input_tokens} "
                        f"输出={usage.output_tokens}",
                    ],
                ),
            )
            turn_status = (
                TurnStatus.CANCELLED
                if status is RunStatus.CANCELLED
                else TurnStatus.FAILED
            )
            await emit_turn_terminal(
                turn_status,
                error=exc.error.model_dump(mode="json"),
                usage=usage.model_dump(mode="json"),
            )
            await self._emit_hook(
                "turn.error",
                {
                    "status": status.value,
                    "error": exc.error.model_dump(mode="json"),
                    "usage": usage.model_dump(mode="json"),
                },
                request=request,
                run_id=run_id,
                turn_id=turn_id,
                duration_ms=duration,
            )
            await self._emit_hook(
                "run.end",
                {
                    "status": status.value,
                    "error": exc.error.model_dump(mode="json"),
                    "usage": usage.model_dump(mode="json"),
                    "duration_ms": duration,
                },
                request=request,
                run_id=run_id,
                turn_id=turn_id,
                duration_ms=duration,
            )
            event = (
                AgentEventType.RUN_CANCELLED
                if status is RunStatus.CANCELLED
                else AgentEventType.RUN_FAILED
            )
            await emit(event, exc.error.model_dump(mode="json"))
            await emit(AgentEventType.DONE, {"run_id": run_id, "status": status.value})
            return result
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - 防御性收敛，不把栈泄漏到 API
            logger.exception("Agent 运行未处理异常")
            duration = int((utcnow() - started).total_seconds() * 1000)
            error = agent_error(
                AgentErrorCode.INTERNAL,
                "Agent 运行失败",
                cause=str(exc),
            )
            result = AgentResult(
                run_id=run_id,
                conversation_id=conversation_id,
                turn_id=turn_id,
                status=RunStatus.FAILED,
                user_message_id=(
                    user_stored.message.id if user_stored is not None else UUID(int=0)
                ),
                assistant_message=final_assistant,
                message_ids=message_ids,
                usage=usage,
                warnings=[_public_warning(item) for item in warnings],
                error=error,
                duration_ms=duration,
            )
            logger.error(
                "%s",
                summary_block(
                    "Agent 失败",
                    [
                        f"status=failed | 耗时={duration}ms | "
                        f"error={error.code.value}",
                        f"消息={len(message_ids)} | LLM={llm_calls} 次 | "
                        f"工具轮={tool_rounds}",
                        f"tokens: 输入={usage.input_tokens} "
                        f"输出={usage.output_tokens}",
                    ],
                ),
            )
            await emit_turn_terminal(
                TurnStatus.FAILED,
                error=error.model_dump(mode="json"),
                usage=usage.model_dump(mode="json"),
            )
            await self._emit_hook(
                "turn.error",
                {"status": "failed", "error": error.model_dump(mode="json")},
                request=request,
                run_id=run_id,
                turn_id=turn_id,
                duration_ms=duration,
            )
            await self._emit_hook(
                "run.end",
                {
                    "status": "failed",
                    "error": error.model_dump(mode="json"),
                    "duration_ms": duration,
                },
                request=request,
                run_id=run_id,
                turn_id=turn_id,
                duration_ms=duration,
            )
            await emit(AgentEventType.RUN_FAILED, error.model_dump(mode="json"))
            await emit(AgentEventType.DONE, {"run_id": run_id, "status": "failed"})
            return result
        finally:
            if heartbeat_task is not None:
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass
            # 只有成功 acquire 的 run 才释放；release 在适配器层幂等。
            try:
                final_status = "failed"
                if result is not None:
                    if result.status is RunStatus.COMPLETED:
                        final_status = "completed"
                    elif result.status is RunStatus.CANCELLED:
                        final_status = "cancelled"
                elif cancel_event.is_set():
                    # 任务被强制取消时没有 result；取消标记仍是权威信号。
                    final_status = "cancelled"
                await self.deps.leases.release(
                    conversation_id,
                    run_id,
                    final_status,
                )
            except Exception:  # noqa: BLE001 - 释放失败不能覆盖 result
                logger.warning("释放 run 租约失败: %s", run_id, exc_info=True)
            # 收尾日志也保留 run/conv 上下文，最后再恢复。
            reset_run_context(log_tokens)

    async def _replay_existing_turn(
        self,
        existing: StoredMessage,
        request: AgentRequest,
        *,
        run_id: str,
        turn_id: str,
    ):
        """幂等命中时从已落库消息重建终态结果。

        只有在用户消息内容一致、且助手回合已经完整落库（工具调用都有对应
        ToolMessage、最后一条 assistant 已完成）时才允许重放；否则返回 None，
        由调用方转成 ``IDEMPOTENCY_CONFLICT``，避免把半截回合伪装成成功。
        """
        from agent.public import AgentResult

        message = existing.message
        if not isinstance(message, UserMessage):
            return None
        if not user_payload_matches(
            message,
            content=request.content,
            images=request.images,
            files=request.files,
        ):
            return None
        following = await self.deps.messages.load_after(
            message.conversation_id,
            existing.seq_no,
        )
        turn_messages: list = []
        for item in following:
            # 下一条用户消息代表新的回合，不能并入本次重放结果。
            if isinstance(item.message, UserMessage):
                break
            turn_messages.append(item.message)
        assistants = [
            item for item in turn_messages if isinstance(item, AssistantMessage)
        ]
        if not assistants:
            return None
        final = assistants[-1]
        if final.status is not MessageStatus.COMPLETED:
            return None
        tool_call_ids = {
            item.tool_call_id for item in turn_messages if isinstance(item, ToolMessage)
        }
        # 有 tool_calls 却没有对应 ToolMessage，说明原回合在工具轮中崩溃。
        if any(call.call_id not in tool_call_ids for call in final.tool_calls):
            return None
        generation_usage = (
            final.generation.usage
            if final.generation is not None and final.generation.usage is not None
            else LLMUsage()
        )
        return AgentResult(
            run_id=run_id,
            conversation_id=message.conversation_id,
            turn_id=turn_id,
            status=RunStatus.COMPLETED,
            user_message_id=message.id,
            assistant_message=final,
            message_ids=[message.id, *(item.id for item in turn_messages)],
            usage=generation_usage,
            warnings=[],
            duration_ms=0,
        )

    async def _heartbeat_loop(
        self,
        conversation_id: UUID,
        run_id: str,
        cancel_event: asyncio.Event,
        lease_lost: asyncio.Event,
    ) -> None:
        """周期续租；租约丢失时置位 lease_lost，主循环在 Step 边界停止。"""
        interval = self.config.policy.run_heartbeat_interval_s
        while not cancel_event.is_set():
            await asyncio.sleep(interval)
            try:
                ok = await self.deps.leases.heartbeat(
                    conversation_id,
                    run_id,
                    self.config.policy.run_lease_seconds,
                )
            except Exception:  # noqa: BLE001 - 续租失败按租约丢失处理
                logger.warning("run 续租失败，停止运行: %s", run_id, exc_info=True)
                lease_lost.set()
                cancel_event.set()
                return
            if not ok:
                lease_lost.set()
                cancel_event.set()
                return

    def _schedule_background(
        self,
        name: str,
        coro: Coroutine,
        *,
        run_id: str,
    ) -> None:
        """投递后台任务；runner 缺失/已关闭时只记录，不影响主回合。"""
        runner = self.deps.background_tasks
        if runner is None:
            coro.close()
            return
        try:
            runner.submit(name, coro, run_id=run_id)
        except Exception:  # noqa: BLE001 - 后台调度失败不能阻断回复
            logger.warning("后台任务投递失败: %s", name, exc_info=True)
            coro.close()

    async def _mark_memory_ready(self, conversation_id: UUID) -> None:
        """回合成功终态后把本会话的暂存行置 ready，供记忆 worker 领取。"""
        assert self.deps.memory_queue is not None
        await self.deps.memory_queue.mark_ready(conversation_id)

    async def _emit_hook(
        self,
        event: str,
        payload: dict | None = None,
        *,
        request: AgentRequest,
        run_id: str,
        turn_id: str,
        call_id: str | None = None,
        duration_ms: int | None = None,
    ) -> None:
        """发送观测事件；Hook 失败只记录，不影响主循环。"""
        if self.deps.hooks is None:
            return
        try:
            await self.deps.hooks.emit(
                event,
                payload or {},
                trace_id=request.request_id,
                run_id=run_id,
                turn_id=turn_id,
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                call_id=call_id,
                duration_ms=duration_ms,
            )
        except Exception:  # noqa: BLE001 - 观测失败不能阻断业务
            logger.warning("Agent Hook 发送失败: %s", event, exc_info=True)

    def _limits(self, request: AgentRequest) -> AgentLimits:
        """按请求收紧限制；硬上限仍来自 AgentPolicy。"""
        policy = self.config.policy
        max_rounds = min(
            policy.max_tool_rounds,
            request.max_tool_rounds or policy.max_tool_rounds,
        )
        return AgentLimits(
            deadline=utcnow() + timedelta(seconds=policy.turn_timeout_s),
            max_tool_rounds=max_rounds,
            max_llm_calls=policy.max_llm_calls_per_run,
            max_parallel_tools=policy.max_parallel_tools,
        )

    async def _tool_permissions(
        self,
        permissions: frozenset[str],
        conversation_id: UUID,
    ) -> frozenset[str]:
        """按会话摘要次数决定是否授予轮次召回工具的权限。

        工具注册表启动后只读；动态能力通过“运行时追加权限码”控制暴露，
        避免为单个会话修改全局注册表。
        """
        code = self.deps.recall_round_permission
        if not code or self.deps.memory is None:
            return permissions
        loader = getattr(self.deps.memory, "compression_count", None)
        if not callable(loader):
            return permissions
        try:
            count = int(await loader(conversation_id))
        except Exception:  # noqa: BLE001 - 摘要计数失败不阻断对话
            logger.warning("读取会话压缩次数失败，跳过轮次召回工具", exc_info=True)
            return permissions
        if count < max(0, int(self.deps.recall_round_min_compressions)):
            return permissions
        return permissions | {code}

    @staticmethod
    def _check_runtime(
        limits: AgentLimits,
        cancel_event: asyncio.Event,
        lease_lost: asyncio.Event,
    ) -> None:
        """在每个 Step 边界检查租约、取消与 deadline。"""
        if lease_lost.is_set():
            raise AgentRuntimeError(
                agent_error(AgentErrorCode.LEASE_LOST, "会话 run 租约已丢失")
            )
        if cancel_event.is_set():
            raise AgentRuntimeError(agent_error(AgentErrorCode.CANCELLED, "运行已取消"))
        if utcnow() >= limits.deadline:
            raise AgentRuntimeError(
                agent_error(AgentErrorCode.TIMEOUT, "回合超时", retryable=True)
            )

    def _llm_request(
        self,
        request: AgentRequest,
        *,
        run_id: str,
        turn_id: str,
        messages: list[LLMMessage],
        tools: list[LLMToolDescriptor],
        deadline,
    ) -> LLMRequest:
        """通过 PromptBuilder 构造 LLM 应用层请求。"""
        return self.prompt_builder.build(
            trace_id=request.request_id,
            conversation_id=request.conversation_id,
            turn_id=turn_id,
            user_id=request.user_id,
            messages=messages,
            tools=tools,
            model_id=request.model_id or self.deps.model_id,
            tool_choice=request.tool_choice,
            purpose=ModelPurpose.CHAT,
            deadline=deadline,
            metadata={**self.deps.metadata, **request.metadata},
            stream=False,
        )

    async def _injected_rag_section(
        self,
        request: AgentRequest,
    ) -> ContextSection | None:
        """INJECTED 模式：检索 KB 并渲染成 RAG section；失败降级不阻断。"""
        mode = getattr(request.rag_mode, "value", request.rag_mode)
        if mode != "injected" or self.deps.rag is None:
            return None
        raw_ids = request.metadata.get("kb_ids") or []
        if isinstance(raw_ids, (str, bytes)):
            raw_ids = [raw_ids]
        kb_ids: list[UUID] = []
        for item in raw_ids:
            try:
                kb_ids.append(UUID(str(item)))
            except (TypeError, ValueError):
                continue
        if not kb_ids or not request.content.strip():
            logger.debug("INJECTED RAG 缺少 kb_ids 或 query，跳过注入")
            return None
        top_k = request.metadata.get("top_k")
        try:
            # 检索事件由 Retriever 的 RagEventSink 统一发出；Agent 只消费结果。
            result = await self.deps.rag.retrieve(
                query=request.content,
                kb_ids=kb_ids,
                user_id=request.user_id,
                permissions=request.permissions,
                top_k=top_k,
                trace_id=request.request_id,
            )
        except Exception:  # noqa: BLE001 - 注入式 RAG 可降级
            logger.warning("INJECTED RAG 检索失败，按无 RAG 继续", exc_info=True)
            return None
        chunks = list(getattr(result, "chunks", []) or [])
        if not chunks:
            return None
        lines: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            title = getattr(chunk, "doc_name", None) or str(chunk.doc_id)
            heading = getattr(chunk, "heading", None)
            label = f"{title} / {heading}" if heading else title
            lines.append(f"[RAG {index}] {label}\n{chunk.content}".strip())
        return ContextSection(
            source="rag",
            name="injected_context",
            content="\n\n".join(lines),
            priority=60,
            token_estimate=0,
        )

    async def _assemble_context(
        self,
        history: Sequence[StoredMessage],
        request: AgentRequest,
        tools: list[LLMToolDescriptor],
        *,
        run_id: str,
        turn_id: str,
    ) -> list[LLMMessage]:
        """加载 memory/预算并组装 Prompt；未注入 assembler 时走旧窗口逻辑。"""
        if self.deps.context is None:
            return await self._llm_messages(history, request)
        base_messages = await self._llm_messages(
            history,
            request,
            include_system=False,
        )
        memory_text: str | None = None
        # 画像最多占上下文的 1/4，避免长期记忆挤占当前对话与工具结果。
        memory_budget = max(
            256,
            int(getattr(self.deps.context, "max_context_tokens", 8192)) // 4,
        )
        if self.deps.memory is not None:
            try:
                memory_text = await self.deps.memory.load_profile(
                    request.user_id,
                    token_budget=memory_budget,
                )
            except Exception:  # noqa: BLE001 - 画像读取失败不阻断对话
                logger.warning("加载用户画像失败", exc_info=True)
        sections: list[ContextSection] = []
        rag_section = await self._injected_rag_section(request)
        if rag_section is not None:
            sections.append(rag_section)
        if memory_text:
            sections.append(
                ContextSection(
                    source="memory",
                    name="long_term_profile",
                    content=memory_text,
                    priority=80,
                    token_estimate=0,
                )
            )
        session_text: str | None = None
        if self.deps.memory is not None:
            try:
                session_text = await self.deps.memory.load_session_summary(
                    request.conversation_id,
                    token_budget=memory_budget,
                )
            except Exception:  # noqa: BLE001 - 摘要读取失败不阻断对话
                logger.warning("加载会话摘要失败", exc_info=True)
        if session_text:
            sections.append(
                ContextSection(
                    source="summary",
                    name="session_summary",
                    content=session_text,
                    priority=70,
                    token_estimate=0,
                )
            )
        if self.deps.facts is not None and request.content.strip():
            try:
                hits = await self.deps.facts.recall(
                    user_id=request.user_id,
                    query=request.content,
                    top_k=self.deps.fact_top_k,
                    min_confidence=self.deps.fact_min_confidence,
                )
            except Exception:  # noqa: BLE001 - 事实召回失败不阻断
                logger.warning("长期事实召回失败", exc_info=True)
                hits = []
            fact_lines = [
                str(item.get("content") or "").strip()
                for item in hits
                if str(item.get("content") or "").strip()
            ]
            if fact_lines:
                sections.append(
                    ContextSection(
                        source="facts",
                        name="long_term_facts",
                        content="\n".join(f"- {line}" for line in fact_lines),
                        priority=65,
                        token_estimate=0,
                    )
                )
        await self._emit_hook(
            "context.assemble.start",
            {
                "history_count": len(base_messages),
                "section_count": len(sections),
                "tool_count": len(tools),
            },
            request=request,
            run_id=run_id,
            turn_id=turn_id,
        )
        plan = self.deps.context.assemble(
            system_prompt=self.deps.system_prompt,
            sections=sections,
            messages=base_messages,
            tool_tokens=sum(len(item.name) + len(item.description) for item in tools)
            // 4,
        )
        await self._emit_hook(
            "context.assemble.end",
            {
                "token_estimate": plan.token_estimate,
                "sections": [item.name for item in plan.sections],
                "dropped": plan.dropped_sections,
                "truncated": plan.truncated,
            },
            request=request,
            run_id=run_id,
            turn_id=turn_id,
        )
        return plan.messages

    async def _llm_messages(
        self,
        history: Sequence[StoredMessage],
        request: AgentRequest,
        *,
        include_system: bool = True,
    ) -> list[LLMMessage]:
        """把持久化消息投影为 LLM wire 消息；RawMessage 跳过。"""
        messages: list[LLMMessage] = []
        if include_system and self.deps.system_prompt:
            messages.append(
                LLMMessage(
                    role=LLMRole.SYSTEM,
                    content=[TextPart(text=self.deps.system_prompt)],
                )
            )
        for stored in history:
            message = stored.message
            if isinstance(message, RawMessage):
                continue
            if isinstance(message, UserMessage) and message.files:
                converted = await self._user_message_with_attachments(message, request)
            else:
                converted = _to_llm_message(message)
            if converted is not None:
                messages.append(converted)
        return messages

    async def _user_message_with_attachments(
        self,
        message: UserMessage,
        request: AgentRequest,
    ) -> LLMMessage:
        """解析 UserMessage.files；没有解析端口时明确失败，不静默丢附件。"""
        parts: list = []
        if message.content:
            parts.append(TextPart(text=message.content))
        for image in message.images:
            parts.append(
                # 图片已有 URL，直接映射；file_id 图片由后续附件端口扩展。
                _to_llm_image(image)
            )
        for file in message.files:
            if self.deps.attachments is None:
                raise AgentRuntimeError(
                    agent_error(
                        AgentErrorCode.FILE_NOT_READY,
                        "附件解析端口未配置，无法发送 file_id 附件",
                        details={"file_id": str(file.file_id)},
                    )
                )
            try:
                part = await self.deps.attachments.resolve(
                    file.file_id,
                    conversation_id=message.conversation_id,
                    user_id=request.user_id,
                )
            except AgentRuntimeError:
                raise
            except Exception as exc:  # noqa: BLE001 - 端口异常统一收敛
                raise AgentRuntimeError(
                    agent_error(
                        AgentErrorCode.FILE_NOT_READY,
                        "附件尚未就绪或解析失败",
                        retryable=True,
                        details={"file_id": str(file.file_id)},
                        cause=str(exc),
                    )
                ) from exc
            if not isinstance(part, FilePart):
                raise AgentRuntimeError(
                    agent_error(
                        AgentErrorCode.FILE_NOT_READY,
                        "附件解析结果不是 FilePart",
                        details={"file_id": str(file.file_id)},
                    )
                )
            if not (part.url or part.text):
                raise AgentRuntimeError(
                    agent_error(
                        AgentErrorCode.FILE_NOT_READY,
                        "附件解析结果缺少 url/text",
                        details={"file_id": str(file.file_id)},
                    )
                )
            parts.append(part)
        return LLMMessage(role=LLMRole.USER, content=parts)

    @staticmethod
    def _tool_descriptors(selection) -> list[LLMToolDescriptor]:
        """ToolSelection -> LLM 工具描述；无选择时返回空列表。"""
        if selection is None:
            return []
        return [
            LLMToolDescriptor(
                name=item.name,
                description=item.description,
                parameters=item.parameters,
                strict=item.strict,
            )
            for item in selection.descriptors
        ]

    async def _stream_response(
        self,
        llm_request: LLMRequest,
        *,
        request: AgentRequest,
        run_id: str,
        turn_id: str,
        assistant_id: UUID,
        publisher: AgentEventPublisher,
        deadline,
        cancel_event: asyncio.Event,
    ) -> LLMResponse:
        """消费应用层流式事件，转发 delta 并拼装完整 LLMResponse。"""
        started = perf_counter()
        first_token_ms: int | None = None
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        usage = LLMUsage()
        finish_reason = FinishReason.STOP
        iterator = self.deps.llm.stream(llm_request).__aiter__()
        while True:
            remaining = max(0.01, (deadline - utcnow()).total_seconds())
            try:
                event = await asyncio.wait_for(iterator.__anext__(), timeout=remaining)
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError as exc:
                raise AgentRuntimeError(
                    agent_error(
                        AgentErrorCode.TIMEOUT,
                        "流式调用超过回合 deadline",
                        retryable=True,
                    )
                ) from exc
            if cancel_event.is_set():
                raise AgentRuntimeError(
                    agent_error(AgentErrorCode.CANCELLED, "运行已取消")
                )
            if utcnow() >= deadline:
                raise AgentRuntimeError(
                    agent_error(AgentErrorCode.TIMEOUT, "流式调用超时", retryable=True)
                )
            if event.type in {"text_delta", "thinking_delta"}:
                if first_token_ms is None:
                    first_token_ms = int((perf_counter() - started) * 1000)
                if event.type == "text_delta":
                    text_parts.append(event.text)
                else:
                    thinking_parts.append(event.text)
                await publisher.emit(
                    AgentEventType.ASSISTANT_DELTA,
                    {
                        "channel": "thinking"
                        if event.type == "thinking_delta"
                        else "text",
                        "text": event.text,
                        "index": event.index,
                    },
                    message_id=assistant_id,
                )
            elif event.type == "tool_call_start":
                tool_calls[event.index] = {
                    "call_id": event.call_id or f"call_{event.index}",
                    "name": event.name or "",
                    "arguments": {},
                    "raw_arguments": None,
                }
            elif event.type == "tool_call_delta":
                item = tool_calls.setdefault(
                    event.index,
                    {
                        "call_id": event.call_id or f"call_{event.index}",
                        "name": event.name or "",
                        "arguments": {},
                        "raw_arguments": None,
                    },
                )
                item["raw_arguments"] = (item["raw_arguments"] or "") + (
                    event.arguments_delta or ""
                )
            elif event.type == "tool_call_end":
                item = tool_calls.setdefault(
                    event.index,
                    {
                        "call_id": event.call_id or f"call_{event.index}",
                        "name": event.name or "",
                        "arguments": {},
                        "raw_arguments": None,
                    },
                )
                item["arguments"] = dict(event.arguments or {})
                item["raw_arguments"] = event.raw_arguments or item["raw_arguments"]
            elif event.type == "usage" and event.usage is not None:
                usage = event.usage
            elif event.type == "end":
                finish_reason = event.finish_reason or finish_reason
                if event.usage is not None:
                    usage = event.usage
            elif event.type == "error":
                raise AgentRuntimeError(
                    agent_error(
                        AgentErrorCode.MODEL_UNAVAILABLE,
                        "模型流式调用失败",
                        retryable=True,
                        details={
                            "model_id": llm_request.model_id,
                            "error_code": event.error_code,
                        },
                        cause=event.error_message,
                    )
                )

        content_parts: list = []
        if text_parts:
            content_parts.append(TextPart(text="".join(text_parts)))
        if thinking_parts:
            content_parts.append(ThinkingPart(text="".join(thinking_parts)))
        calls = [
            LLMToolCall(
                call_id=item["call_id"],
                name=item["name"],
                arguments=item["arguments"],
                raw_arguments=item["raw_arguments"],
                index=index,
            )
            for index, item in sorted(tool_calls.items())
        ]
        latency_ms = int((perf_counter() - started) * 1000)
        return LLMResponse(
            request_id=llm_request.request_id,
            provider=self.deps.model_id,
            model=llm_request.model_id,
            message=LLMMessage(
                role=LLMRole.ASSISTANT,
                content=content_parts,
                tool_calls=calls,
            ),
            finish_reason=finish_reason,
            usage=usage,
            latency_ms=latency_ms,
            first_token_ms=first_token_ms,
            phase=LLMStreamPhase.COMPLETED,
        )

    def _assistant_message(
        self,
        response: LLMResponse,
        *,
        conversation_id: UUID,
        assistant_id: UUID,
    ) -> AssistantMessage:
        """LLMResponse -> AssistantMessage；空响应转 EMPTY_RESPONSE。"""
        content = "".join(
            part.text for part in response.message.content if isinstance(part, TextPart)
        )
        thinking = [
            Thinking(
                text=part.text,
                signature=part.signature,
                redacted=part.redacted,
            )
            for part in response.message.content
            if isinstance(part, ThinkingPart)
        ]
        tool_calls = [
            ToolCall(
                call_id=call.call_id,
                name=call.name,
                arguments=dict(call.arguments or {}),
                raw_arguments=call.raw_arguments,
            )
            for call in response.message.tool_calls
        ]
        status = _message_status(response.finish_reason)
        generation = GenerationInfo(
            provider=response.provider,
            model=response.model,
            request_id=response.request_id,
            response_id=response.response_id,
            finish_reason=response.finish_reason,
            usage=response.usage,
            latency_ms=response.latency_ms,
        )
        try:
            return AssistantMessage(
                id=assistant_id,
                conversation_id=conversation_id,
                content=content,
                thinking=thinking,
                tool_calls=tool_calls,
                generation=generation,
                status=status,
            )
        except ValueError as exc:
            raise AgentRuntimeError(
                agent_error(
                    AgentErrorCode.EMPTY_RESPONSE,
                    "模型返回空响应或非法消息",
                    cause=str(exc),
                )
            ) from exc

    async def _execute_tools(
        self,
        assistant: AssistantMessage,
        *,
        request: AgentRequest,
        permissions: frozenset[str],
        run_id: str,
        turn_id: str,
        conversation_id: UUID,
        deadline,
        publisher: AgentEventPublisher,
        cancel_event: asyncio.Event,
    ) -> list[ToolMessage]:
        """执行一轮工具并转换成 ToolMessage；失败也返回占位消息。"""
        if self.deps.tools is None:
            raise AgentRuntimeError(
                agent_error(
                    AgentErrorCode.TOOL_EXECUTION_FAILED,
                    "模型请求了工具，但 Agent 未注入 ToolPort",
                )
            )
        for call in assistant.tool_calls:
            await publisher.emit(
                AgentEventType.TOOL_STARTED,
                {
                    "call_id": call.call_id,
                    "name": call.name,
                    "arguments_preview": _safe_arguments(call.arguments),
                },
                call_id=call.call_id,
            )
            await self._emit_hook(
                "tool.call.start",
                {"call_id": call.call_id, "tool_name": call.name},
                request=request,
                run_id=run_id,
                turn_id=turn_id,
                call_id=call.call_id,
            )
        context = ToolContext(
            user_id=request.user_id,
            permissions=permissions,
            conversation_id=conversation_id,
            turn_id=turn_id,
            run_id=run_id,
            call_id="",
            tool_name="",
            trace_id=request.request_id,
            deadline=deadline,
            cancel_token=cancel_event,
            services=self.deps.services,
        )
        try:
            results = await self.deps.tools.execute_batch(
                assistant.tool_calls,
                context,
                max_concurrency=self.config.policy.max_parallel_tools,
            )
        except Exception as exc:  # noqa: BLE001 - 工具端口失败补占位
            logger.warning("工具批次执行失败: %s", exc)
            results = [
                ToolResult.failure(
                    call_id=call.call_id,
                    tool_name=call.name,
                    error=_tool_error(call),
                )
                for call in assistant.tool_calls
            ]
        # 端口返回缺项时补失败占位，保证每个 call_id 恰好一条 ToolMessage。
        by_call = {result.call_id: result for result in results}
        results = [
            by_call.get(
                call.call_id,
                ToolResult.failure(
                    call_id=call.call_id,
                    tool_name=call.name,
                    error=_tool_error(call),
                ),
            )
            for call in assistant.tool_calls
        ]
        messages: list[ToolMessage] = []
        for result in results:
            await publisher.emit(
                AgentEventType.TOOL_RESULT,
                {
                    "call_id": result.call_id,
                    "name": result.tool_name,
                    "status": result.status.value,
                    "error_code": (result.error.code.value if result.error else None),
                    "truncated": result.truncated,
                },
                call_id=result.call_id,
            )
            await self._emit_hook(
                "tool.result",
                {
                    "call_id": result.call_id,
                    "tool_name": result.tool_name,
                    "status": result.status.value,
                    "error_code": (result.error.code.value if result.error else None),
                    "truncated": result.truncated,
                },
                request=request,
                run_id=run_id,
                turn_id=turn_id,
                call_id=result.call_id,
            )
            messages.append(_tool_message(result, conversation_id=conversation_id))
        return messages


def _to_llm_image(image) -> ImagePart:
    """ImageRef -> LLM 图片 part；当前只支持 URL 图片。"""
    return ImagePart(url=image.url, detail=image.detail.value)


def _to_llm_message(message) -> LLMMessage | None:
    """领域消息 -> LLMMessage；未知类型返回 None。"""
    role = getattr(message, "role", None)
    if role == "system":
        return LLMMessage(
            role=LLMRole.SYSTEM,
            content=[TextPart(text=message.content)],
        )
    if role == "user":
        parts = []
        if message.content:
            parts.append(TextPart(text=message.content))
        for image in message.images:
            parts.append(_to_llm_image(image))
        for file in message.files:
            parts.append(FilePart(file_id=file.file_id, filename=file.filename))
        return LLMMessage(role=LLMRole.USER, content=parts)
    if role == "assistant":
        parts = []
        if message.content:
            parts.append(TextPart(text=message.content))
        parts.extend(
            ThinkingPart(
                text=item.text,
                signature=item.signature,
                redacted=item.redacted,
            )
            for item in message.thinking
        )
        return LLMMessage(
            role=LLMRole.ASSISTANT,
            content=parts,
            tool_calls=[_to_llm_tool_call(call) for call in message.tool_calls],
            prefix=message.is_prefix,
        )
    if role == "tool":
        return LLMMessage(
            role=LLMRole.TOOL,
            content=[TextPart(text=message.content)] if message.content else [],
            tool_call_id=message.tool_call_id,
            name=message.name,
        )
    return None


def _to_llm_tool_call(call: ToolCall):
    """领域 ToolCall -> LLM wire ToolCall。"""
    return LLMToolCall(
        call_id=call.call_id,
        name=call.name,
        arguments=dict(call.arguments),
        raw_arguments=call.raw_arguments,
    )


def _run_status_for_error(code: AgentErrorCode) -> RunStatus:
    """错误码 -> run 终态；冲突/租约丢失/取消各自保留独立语义。"""
    if code is AgentErrorCode.LEASE_CONFLICT:
        return RunStatus.CONFLICT
    if code is AgentErrorCode.LEASE_LOST:
        return RunStatus.EXPIRED
    if code is AgentErrorCode.CANCELLED:
        return RunStatus.CANCELLED
    return RunStatus.FAILED


def _message_status(reason: FinishReason) -> MessageStatus:
    """按 finish_reason 映射 assistant 消息状态。"""
    if reason in (FinishReason.LENGTH, FinishReason.CANCELLED):
        return MessageStatus.PARTIAL
    if reason in (FinishReason.ERROR, FinishReason.CONTENT_FILTER):
        return MessageStatus.FAILED
    return MessageStatus.COMPLETED


def _tool_message(result: ToolResult, *, conversation_id: UUID) -> ToolMessage:
    """ToolResult -> ToolMessage；失败时保证 content/json 至少有一个。"""
    content = result.content or (
        result.error.message if result.error is not None else "工具执行失败"
    )
    json_data = result.data
    if result.artifacts:
        # ToolMessage 没有独立 artifacts 字段；按工具文档 §7.5 放进 json.artifacts。
        artifacts = [item.model_dump(mode="json") for item in result.artifacts]
        if isinstance(json_data, dict):
            json_data = {**json_data, "artifacts": artifacts}
        elif json_data is None:
            json_data = {"artifacts": artifacts}
        else:
            json_data = {"data": json_data, "artifacts": artifacts}
    if json_data is None and result.error is not None:
        json_data = {"error": result.error.model_dump(mode="json")}
    return ToolMessage(
        conversation_id=conversation_id,
        tool_call_id=result.call_id,
        name=result.tool_name,
        content=content,
        # Pydantic 构造签名按别名 ``json`` 生成；与存储协议保持一致。
        json=json_data,
        files=(
            [*result.files, result.overflow_file]
            if result.overflow_file is not None
            else list(result.files)
        ),
        images=list(result.images),
        status=(
            MessageStatus.COMPLETED
            if result.status is ToolResultStatus.SUCCEEDED
            else MessageStatus.FAILED
        ),
    )


def _tool_error(call: ToolCall):
    """工具端口异常时生成可回喂 LLM 的占位 ToolError。"""
    from agent.tools.models import ToolError
    from exception.tool import ToolErrorCode

    return ToolError(
        code=ToolErrorCode.UPSTREAM_ERROR,
        message="工具执行失败",
        details={"tool": call.name},
    )


def _add_usage(left: LLMUsage, right: LLMUsage) -> LLMUsage:
    """累加两次 LLM 用量。"""
    return LLMUsage(
        input_tokens=left.input_tokens + right.input_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
        total_tokens=left.total_tokens + right.total_tokens,
        cached_input_tokens=left.cached_input_tokens + right.cached_input_tokens,
        reasoning_tokens=left.reasoning_tokens + right.reasoning_tokens,
    )


def _public_warning(item: AgentWarningData):
    """内部告警 -> 公共 AgentWarning。"""
    from agent.public import AgentWarning

    return AgentWarning(code=item.code, message=item.message, details=item.details)


def _safe_arguments(arguments: dict) -> dict:
    """事件只放参数键/短预览，不把完整敏感参数推到前端。"""
    return {
        "keys": sorted(arguments.keys()),
        "preview": str(arguments)[:200],
    }


def _content_bytes(message: object) -> int:
    """消息正文的 UTF-8 字节数；RawMessage 的 dict 正文按 0 计。"""
    content = getattr(message, "content", "")
    return len(content.encode("utf-8")) if isinstance(content, str) else 0
