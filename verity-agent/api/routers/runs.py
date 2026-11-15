"""运行状态、取消与 SSE 事件路由。"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent import AgentEvent, CancelReason
from agent.runtime.errors import AgentRuntimeError
from api.routers.auth import get_current_user
from api.routers.conversations import _agent
from config import settings
from exception import NotFoundError
from store.pg.models.user import User
from store.pg.repositories import ConversationRepository

router = APIRouter(prefix="/runs", tags=["运行"])


class CancelRequest(BaseModel):
    """取消运行入参。"""

    reason: str = Field(default="user")


def _sse(event: AgentEvent) -> str:
    """把 AgentEvent 编码为 SSE 帧。"""
    data = json.dumps(event.model_dump(mode="json"), ensure_ascii=False)
    return f"id: {event.id}\nevent: {event.event_type.value}\ndata: {data}\n\n"


async def _event_stream(
    stream,
    *,
    heartbeat_interval_s: float,
) -> AsyncIterator[str]:
    """消费 AgentStream，并在空闲时发送 SSE heartbeat。"""
    try:
        while True:
            try:
                event = await asyncio.wait_for(
                    stream.__anext__(),
                    timeout=heartbeat_interval_s,
                )
            except asyncio.TimeoutError:
                yield (
                    f"event: heartbeat\ndata: "
                    f"{json.dumps({'run_id': stream.run_id})}\n\n"
                )
                continue
            yield _sse(event)
    except StopAsyncIteration:
        return
    except asyncio.CancelledError:
        raise
    except AgentRuntimeError:
        return


@router.get("/{run_id}")
async def get_run(
    run_id: str,
    user: User = Depends(get_current_user),
    agent=Depends(_agent),
) -> dict[str, Any]:
    """读取 run 视图；只能读取自己会话的 run。"""
    view = await agent.get_run(run_id)
    if view is None:
        raise NotFoundError("运行", run_id)
    await ConversationRepository().get_for_user(user.id, view.conversation_id)
    return view.model_dump(mode="json")


@router.post("/{run_id}/cancel")
async def cancel_run(
    run_id: str,
    payload: CancelRequest,
    user: User = Depends(get_current_user),
    agent=Depends(_agent),
) -> dict[str, Any]:
    """幂等取消 run；不存在/已终态也返回 202。"""
    view = await agent.get_run(run_id)
    if view is not None:
        await ConversationRepository().get_for_user(user.id, view.conversation_id)
    try:
        reason = CancelReason(payload.reason)
    except ValueError:
        reason = CancelReason.USER
    await agent.cancel(run_id, reason)
    return {"run_id": run_id, "status": "cancelling"}


@router.get("/{run_id}/events")
async def run_events(
    run_id: str,
    request: Request,
    user: User = Depends(get_current_user),
    agent=Depends(_agent),
) -> StreamingResponse:
    """订阅 run 事件；支持 Last-Event-ID 断线重放。"""
    view = await agent.get_run(run_id)
    if view is None:
        raise NotFoundError("运行", run_id)
    await ConversationRepository().get_for_user(user.id, view.conversation_id)
    last_event_id = request.headers.get("Last-Event-ID")
    try:
        stream = agent.attach(run_id, last_event_id=last_event_id)
    except AgentRuntimeError as exc:
        raise NotFoundError("运行", run_id) from exc
    return StreamingResponse(
        _event_stream(
            stream,
            heartbeat_interval_s=settings.agent.stream.heartbeat_interval_s,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
