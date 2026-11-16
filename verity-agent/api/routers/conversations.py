"""会话与消息 HTTP 路由。

HTTP 层只做鉴权、DTO 投影和调用 Agent Facade；不在这里拼 Prompt、
不执行工具，也不直接读写工具/记忆存储。
"""

from __future__ import annotations

from typing import Any, NoReturn
from uuid import UUID

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

from agent import AgentRequest, RAGMode
from agent.llm import ToolChoice
from agent.message.models import FileRef, ImageRef
from agent.runtime.enums import AgentErrorCode, RunStatus
from agent.runtime.errors import AgentRuntimeError
from api.routers.auth import get_current_user
from config import settings
from exception import (
    ConflictError,
    IdempotencyConflictError,
    InternalError,
    LeaseConflictError,
    NotFoundError,
    UnavailableError,
    ValidationError,
)
from store.pg.models.user import User
from store.pg.repositories import (
    ConversationMessageRepository,
    ConversationRepository,
    PermissionRepository,
)
from utils.ids import run_id_for
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
        """content/images/files 至少一个非空，且不超过消息模块上限。"""
        if not self.content.strip() and not self.images and not self.files:
            raise ValueError("content/images/files 至少有一个非空")
        limits = settings.message
        if len(self.content) > limits.max_text_chars:
            raise ValueError(f"content 超过 {limits.max_text_chars} 字符")
        if len(self.images) > limits.max_images:
            raise ValueError(f"images 最多 {limits.max_images} 个")
        if len(self.files) > limits.max_files:
            raise ValueError(f"files 最多 {limits.max_files} 个")
        return self


def _agent(request: Request):
    """取 lifespan 装配的 Agent；缺失时返回 503。"""
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise UnavailableError("Agent 尚未初始化")
    return agent


def _dump_items(value: Any) -> list[Any]:
    """把消息内的附件/产物模型列表投影成 JSON。"""
    return [
        item.model_dump(mode="json") if hasattr(item, "model_dump") else item
        for item in (value or [])
    ]


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
    # ToolMessage 的字段别名是 json，但 Pydantic 的 ``model.json`` 是方法；
    # 只能取 ``json_data`` 字段，不能 getattr("json")，否则响应里会出现 bound method。
    json_data = getattr(message, "json_data", None)
    status = getattr(message, "status", None)
    return {
        "id": str(getattr(message, "id", "")),
        "seq_no": None,
        "role": str(getattr(role, "value", role) or ""),
        "status": str(getattr(status, "value", status) or ""),
        "content": content,
        "thinking": thinking,
        "tool_calls": tool_calls,
        "json": json_data,
        "name": getattr(message, "name", None),
        "tool_call_id": getattr(message, "tool_call_id", None),
        "images": _dump_items(getattr(message, "images", None)),
        "files": _dump_items(getattr(message, "files", None)),
        "artifacts": _dump_items(getattr(message, "artifacts", None)),
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
    """创建会话；已有空会话时复用，避免堆积多个空对话。"""
    repository = ConversationRepository()
    conversation = await repository.find_empty_by_user(user.id)
    if conversation is not None:
        # 复用空会话时允许用新标题（例如首条消息生成的标题）。
        if payload.title.strip() and payload.title != conversation.title:
            conversation = await repository.update_title(
                conversation.id, payload.title
            )
    else:
        conversation = await repository.create(user.id, title=payload.title)
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
    # 历史数据可能残留多个空会话；列表只展示最近一个，避免“无限空对话”观感。
    visible: list[Any] = []
    empty_seen = False
    for item in items:
        if item.turn_count == 0 and item.last_message_at is None:
            if empty_seen:
                continue
            empty_seen = True
        visible.append(item)
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
            for item in visible
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
) -> Any:
    """发送消息：流式返回 run 句柄，非流式直接等待终态。"""
    await ConversationRepository().get_for_user(user.id, conversation_id)
    permissions = await PermissionRepository().list_codes_for_user(user.id)
    agent = _agent(request)
    try:
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
            rag_mode=RAGMode(payload.rag_mode.lower()) if payload.rag_mode else None,
            tool_choice=(
                ToolChoice.model_validate(payload.tool_choice)
                if payload.tool_choice
                else None
            ),
            max_tool_rounds=payload.max_tool_rounds,
            metadata=payload.metadata,
        )
    except ValueError as exc:
        raise ValidationError("消息参数不合法", details={"cause": str(exc)}) from exc
    if payload.stream:
        # 先按稳定 run id 判断幂等命中：已有运行中 run 直接返回句柄让客户端 attach；
        # 已有终态结果直接返回结果，避免重复执行 LLM/工具。
        run_id = run_id_for(conversation_id, payload.client_message_id)
        existing = await agent.get_run(run_id)
        if existing is not None:
            return JSONResponse(
                status_code=200 if existing.result is not None else 202,
                content=_run_handle(
                    run_id=existing.run_id,
                    turn_id=existing.turn_id,
                    message_id=existing.user_message_id,
                    client_message_id=payload.client_message_id,
                    status=(
                        existing.result.status.value
                        if existing.result is not None
                        else "queued"
                    ),
                    result=existing.result,
                ),
            )
        try:
            stream = agent.stream(agent_request)
        except AgentRuntimeError as exc:
            _raise_agent_error(exc.error)
        return JSONResponse(
            status_code=202,
            content=_run_handle(
                run_id=stream.run_id,
                turn_id=stream.turn_id,
                message_id=None,
                client_message_id=payload.client_message_id,
                status="queued",
            ),
        )
    result = await agent.run(agent_request)
    _raise_for_result(result)
    return {
        "message_id": str(result.user_message_id),
        "client_message_id": str(payload.client_message_id),
        "run_id": result.run_id,
        "turn_id": result.turn_id,
        "status": result.status.value,
        "result": result.model_dump(mode="json"),
    }


def _run_handle(
    *,
    run_id: str,
    turn_id: str,
    message_id: UUID | None,
    client_message_id: UUID,
    status: str,
    result=None,
) -> dict[str, Any]:
    """构造发送消息的 run 句柄；流式与非流式共用一个形状。"""
    payload: dict[str, Any] = {
        "message_id": str(message_id) if message_id is not None else None,
        "client_message_id": str(client_message_id),
        "run_id": run_id,
        "turn_id": turn_id,
        "status": status,
        "events_url": f"/runs/{run_id}/events",
        "stream_url": f"/runs/{run_id}/events",
    }
    if result is not None:
        payload["result"] = result.model_dump(mode="json")
    return payload


def _raise_agent_error(error) -> NoReturn:
    """把 Agent 公开错误码映射为 HTTP 领域异常（仅用于请求入口）。"""
    code = error.code
    if code is AgentErrorCode.LEASE_CONFLICT:
        raise LeaseConflictError(error.message, details=error.details)
    if code is AgentErrorCode.IDEMPOTENCY_CONFLICT:
        raise IdempotencyConflictError(error.message, details=error.details)
    if code is AgentErrorCode.RUN_NOT_FOUND:
        raise NotFoundError("运行", error.details.get("run_id"))
    if code is AgentErrorCode.FILE_NOT_READY:
        raise ConflictError(error.message, details=error.details)
    if code in {
        AgentErrorCode.MODEL_UNAVAILABLE,
        AgentErrorCode.TIMEOUT,
        AgentErrorCode.DEPENDENCY_UNAVAILABLE,
    }:
        raise UnavailableError(error.message, details=error.details)
    if code in {
        AgentErrorCode.MODEL_CONTENT_FILTERED,
        AgentErrorCode.CONTEXT_LENGTH_EXCEEDED,
    }:
        raise ValidationError(error.message, details=error.details)
    if code in {
        AgentErrorCode.INVALID_CONFIG,
        AgentErrorCode.INTERNAL,
        AgentErrorCode.PERSISTENCE_ERROR,
    }:
        raise InternalError(error.message, details=error.details)
    # 其他终态（如 EMPTY_RESPONSE、工具失败）保持 409，调用方可读 result 决定重试。
    raise ConflictError(error.message, details=error.details)


def _raise_for_result(result) -> None:
    """非流式入口：把冲突/依赖类终态转成对应 HTTP 错误。"""
    if result.status in {RunStatus.CONFLICT, RunStatus.EXPIRED}:
        code = (
            result.error.code
            if result.error is not None
            else AgentErrorCode.LEASE_CONFLICT
        )
        if code is AgentErrorCode.IDEMPOTENCY_CONFLICT:
            raise IdempotencyConflictError(
                result.error.message if result.error else "幂等冲突",
                details=result.error.details if result.error else {},
            )
        raise LeaseConflictError(
            result.error.message if result.error else "会话已有活跃运行",
            details=result.error.details if result.error else {},
        )
    if result.error is not None:
        _raise_agent_error(result.error)
