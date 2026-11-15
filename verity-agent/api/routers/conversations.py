"""会话与消息 HTTP 路由。

HTTP 层只做鉴权、DTO 投影和调用 Agent Facade；不在这里拼 Prompt、
不执行工具，也不直接读写工具/记忆存储。
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field, model_validator

from agent import AgentRequest
from agent.message.models import FileRef, ImageRef
from api.routers.auth import get_current_user
from exception import UnavailableError, ValidationError
from store.pg.models.user import User
from store.pg.repositories import (
    ConversationMessageRepository,
    ConversationRepository,
    PermissionRepository,
)
from utils.pagination import decode_cursor, encode_cursor, normalize_limit

router = APIRouter(prefix="/conversations", tags=["会话"])


class CreateConversationRequest(BaseModel):
    """创建会话入参。"""

    title: str = Field(default="", max_length=300)


class SendMessageRequest(BaseModel):
    """发送消息入参；HTTP DTO 与 AgentRequest 分离，避免把公开契约直接当传输层。"""

    client_message_id: UUID
    content: str = ""
    images: list[dict[str, Any]] = Field(default_factory=list)
    files: list[dict[str, Any]] = Field(default_factory=list)
    stream: bool = True
    model_id: str | None = None
    rag_mode: str | None = None
    tool_choice: dict[str, Any] | None = None
    max_tool_rounds: int | None = Field(default=None, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _has_payload(self) -> "SendMessageRequest":
        """content/images/files 至少一个非空。"""
        if not self.content.strip() and not self.images and not self.files:
            raise ValueError("content/images/files 至少有一个非空")
        return self


def _agent(request: Request):
    """取 lifespan 装配的 Agent；缺失时返回 503。"""
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise UnavailableError("Agent 尚未初始化")
    return agent


def _decision_payload(message: Any) -> dict[str, Any]:
    """把领域消息投影为前端可读结构。"""
    role = getattr(message, "role", None)
    content = getattr(message, "content", "")
    thinking = [
        item.model_dump(mode="json")
        for item in getattr(message, "thinking", []) or []
    ]
    tool_calls = [
        item.model_dump(mode="json")
        for item in getattr(message, "tool_calls", []) or []
    ]
    json_data = getattr(message, "json_data", None)
    if json_data is None:
        json_data = getattr(message, "json", None)
    return {
        "id": str(getattr(message, "id", "")),
        "seq_no": None,
        "role": role.value if hasattr(role, "value") else str(role),
        "status": (
            message.status.value
            if hasattr(getattr(message, "status", None), "value")
            else str(getattr(message, "status", ""))
        ),
        "content": content,
        "thinking": thinking,
        "tool_calls": tool_calls,
        "json": json_data,
        "created_at": (
            message.created_at.isoformat()
            if getattr(message, "created_at", None) is not None
            else None
        ),
    }


@router.post("", status_code=201)
async def create_conversation(
    payload: CreateConversationRequest,
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """创建当前用户的会话。"""
    conversation = await ConversationRepository().create(
        user.id,
        title=payload.title,
    )
    return {
        "id": str(conversation.id),
        "title": conversation.title,
        "status": conversation.status.value,
        "created_at": conversation.created_at.isoformat(),
    }


@router.get("")
async def list_conversations(
    limit: int = 50,
    cursor: str | None = None,
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """按最近消息时间倒序列出当前用户会话。"""
    limit = normalize_limit(limit)
    offset = 0
    if cursor:
        offset = int(decode_cursor(cursor).get("offset", 0))
    items = await ConversationRepository().list_by_user(
        user.id,
        limit=limit,
        offset=offset,
    )
    next_cursor = (
        encode_cursor({"offset": offset + limit}) if len(items) == limit else None
    )
    return {
        "items": [
            {
                "id": str(item.id),
                "title": item.title,
                "status": item.status.value,
                "last_message_at": (
                    item.last_message_at.isoformat()
                    if item.last_message_at is not None
                    else None
                ),
                "turn_count": item.turn_count,
            }
            for item in items
        ],
        "next_cursor": next_cursor,
    }


@router.get("/{conversation_id}/messages")
async def list_messages(
    conversation_id: UUID,
    limit: int = 50,
    cursor: str | None = None,
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """按 seq_no 升序读取消息；cursor 为上一页最后一个 seq。"""
    await ConversationRepository().get_for_user(user.id, conversation_id)
    limit = normalize_limit(limit)
    after_seq = 0
    if cursor:
        after_seq = int(decode_cursor(cursor).get("seq", 0))
    rows = await ConversationMessageRepository().list_range(
        conversation_id,
        after_seq=after_seq,
        limit=limit,
    )
    items: list[dict[str, Any]] = []
    for row in rows:
        message = _decode_message(row)
        if message is None or getattr(message, "role", None) == "system":
            continue
        payload = _decision_payload(message)
        payload["seq_no"] = row.seq_no
        items.append(payload)
    next_cursor = (
        encode_cursor({"seq": rows[-1].seq_no}) if len(rows) == limit else None
    )
    return {"items": items, "next_cursor": next_cursor}


def _decode_message(row: Any):
    """解码一行消息；坏行不阻断列表，只跳过。"""
    from agent.message.codec import decode_stored_row

    try:
        return decode_stored_row(row.model_dump()).message
    except Exception:  # noqa: BLE001 - 历史坏行不应阻断列表
        return None


@router.post("/{conversation_id}/messages")
async def send_message(
    conversation_id: UUID,
    payload: SendMessageRequest,
    request: Request,
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """发送消息：流式返回 run 句柄，非流式直接等待终态。"""
    await ConversationRepository().get_for_user(user.id, conversation_id)
    permissions = await PermissionRepository().list_codes_for_user(user.id)
    agent = _agent(request)
    agent_request = AgentRequest(
        user_id=user.id,
        conversation_id=conversation_id,
        client_message_id=payload.client_message_id,
        content=payload.content,
        images=[ImageRef.model_validate(item) for item in payload.images],
        files=[FileRef.model_validate(item) for item in payload.files],
        permissions=frozenset(permissions),
        stream=payload.stream,
        model_id=payload.model_id,
        rag_mode=payload.rag_mode,
        tool_choice=payload.tool_choice,
        max_tool_rounds=payload.max_tool_rounds,
        metadata=payload.metadata,
    )
    if payload.stream:
        stream = agent.stream(agent_request)
        return {
            "message_id": None,
            "client_message_id": str(payload.client_message_id),
            "run_id": stream.run_id,
            "turn_id": stream.turn_id,
            "status": "queued",
            "events_url": f"/runs/{stream.run_id}/events",
            "stream_url": f"/runs/{stream.run_id}/events",
        }
    result = await agent.run(agent_request)
    if result.status.value in {"conflict", "expired"}:
        raise ValidationError(
            "会话已有活跃 run 或租约已过期",
            details=result.error.model_dump(mode="json") if result.error else {},
        )
    return {
        "message_id": str(result.user_message_id),
        "run_id": result.run_id,
        "turn_id": result.turn_id,
        "status": result.status.value,
        "result": result.model_dump(mode="json"),
    }
