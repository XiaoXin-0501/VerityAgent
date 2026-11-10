"""消息 content 编解码（v2 写入，v1 兼容读取）。

写入严格：未知字段/枚举/缺必填由 Pydantic 拒绝；读取宽容：未知版本或
协议损坏降级为只读 ``RawMessage``，不让单条坏消息拖垮会话列表。
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import ValidationError as PydanticValidationError

from exception import (
    MessageDecodeError,
    MessageEncodeError,
    UnknownMessageRoleError,
    UnsupportedMessageVersionError,
)
from utils.time import utcnow

from .enum import (
    ArtifactKind,
    ImageDetail,
    MessageContentVersion,
    MessageRole,
    MessageStatus,
    SystemOrigin,
    ToolCallStatus,
)
from .models import (
    Artifact,
    AssistantMessage,
    BaseMessage,
    FileRef,
    ImageRef,
    RawMessage,
    StoredMessage,
    SystemMessage,
    Thinking,
    ToolCall,
    ToolFile,
    ToolImage,
    ToolMessage,
    UserMessage,
)

__all__ = [
    "MESSAGE_CONTENT_VERSION",
    "encode_content",
    "decode_content",
    "decode_row",
    "decode_stored_row",
    "content_version",
]

MESSAGE_CONTENT_VERSION = int(MessageContentVersion.V2)

_COMMON_KEYS = {"id", "conversation_id", "role", "status", "created_at", "content"}


def content_version(payload: Mapping[str, Any]) -> int:
    value = payload.get("v", MessageContentVersion.V1)
    if not isinstance(value, int) or isinstance(value, bool):
        raise MessageDecodeError(
            "content.v 必须是整数", detail={"v": repr(value)}
        )
    return value


# ——————————————————— 写入 ———————————————————

def encode_content(message: BaseMessage) -> dict[str, Any]:
    """领域消息 -> content JSONB（固定写 v2）。"""
    if isinstance(message, RawMessage):
        raise MessageEncodeError("RawMessage 是只读降级模型，不能写回存储")

    payload = message.model_dump(mode="json", by_alias=True)
    for key in _COMMON_KEYS:
        payload.pop(key, None)
    payload["v"] = MESSAGE_CONTENT_VERSION
    payload["text"] = message.content

    if isinstance(message, ToolMessage):
        # 存储协议使用 ``json``；字段名是 json_data，by_alias 已输出 json
        if "json" not in payload:
            payload["json"] = payload.pop("json_data", None)
    elif isinstance(message, SystemMessage):
        payload["origin"] = message.origin.value
    elif isinstance(message, UserMessage):
        payload["images"] = [
            {"url": item.url, "detail": item.detail.value} for item in message.images
        ]
        payload["files"] = [
            {
                "file_id": str(item.file_id),
                **({"filename": item.filename} if item.filename else {}),
            }
            for item in message.files
        ]
    elif isinstance(message, AssistantMessage):
        payload["thinking"] = [
            {
                "text": item.text,
                "signature": item.signature,
                "redacted": item.redacted,
            }
            for item in message.thinking
        ]
        payload["tool_calls"] = [
            {
                "call_id": call.call_id,
                "name": call.name,
                "arguments": call.arguments,
                "raw_arguments": call.raw_arguments,
                "status": call.status.value,
            }
            for call in message.tool_calls
        ]
        payload["artifacts"] = [
            {
                "kind": item.kind.value,
                "url": item.url,
                **({"filename": item.filename} if item.filename else {}),
            }
            for item in message.artifacts
        ]
        payload["is_prefix"] = message.is_prefix
        payload["generation"] = (
            message.generation.model_dump(mode="json", by_alias=True)
            if message.generation is not None
            else None
        )

    # 稳定键顺序，便于 diff/调试
    return {key: payload[key] for key in sorted(payload)}


# ——————————————————— 读取 ———————————————————

def decode_content(
    role: MessageRole | str,
    payload: Mapping[str, Any],
    *,
    message_id: UUID,
    conversation_id: UUID,
    status: MessageStatus,
    created_at: datetime,
) -> BaseMessage:
    """content JSONB -> 领域消息；协议错误抛明确异常。"""
    try:
        role_enum = MessageRole(role)
    except ValueError as exc:
        raise UnknownMessageRoleError(role) from exc
    if not isinstance(payload, Mapping):
        raise MessageDecodeError(
            "content 必须是 JSON 对象",
            detail={"type": type(payload).__name__},
        )

    version = content_version(payload)
    if version not in (MessageContentVersion.V1, MessageContentVersion.V2):
        raise UnsupportedMessageVersionError(version)
    data = _normalize_payload(role_enum, payload, version)

    common = {
        "id": message_id,
        "conversation_id": conversation_id,
        "status": status,
        "created_at": created_at,
    }
    try:
        if role_enum is MessageRole.SYSTEM:
            return SystemMessage(
                **common,
                content=data.get("text", ""),
                name=data.get("name", "default"),
                origin=_system_origin(data.get("origin")),
            )
        if role_enum is MessageRole.USER:
            return UserMessage(
                **common,
                content=data.get("text", ""),
                images=[_image_ref(item) for item in data.get("images", [])],
                files=[_file_ref(item) for item in data.get("files", [])],
            )
        if role_enum is MessageRole.ASSISTANT:
            return AssistantMessage(
                **common,
                content=data.get("text", ""),
                thinking=[_thinking(item) for item in data.get("thinking", [])],
                tool_calls=[_tool_call(item) for item in data.get("tool_calls", [])],
                artifacts=[_artifact(item) for item in data.get("artifacts", [])],
                is_prefix=bool(data.get("is_prefix", False)),
                generation=data.get("generation"),
            )
        if role_enum is MessageRole.TOOL:
            return ToolMessage(
                **common,
                content=data.get("text", ""),
                tool_call_id=data["tool_call_id"],
                name=data["name"],
                json=data.get("json"),
                files=[ToolFile.model_validate(item) for item in data.get("files", [])],
                images=[
                    ToolImage.model_validate(item) for item in data.get("images", [])
                ],
            )
    except KeyError as exc:
        raise MessageDecodeError(
            f"content 缺少必填字段: {exc.args[0]}",
            detail={"role": role_enum.value, "v": version},
        ) from exc
    except PydanticValidationError as exc:
        raise MessageDecodeError(
            "content 不符合消息协议",
            detail={
                "role": role_enum.value,
                "v": version,
                "errors": exc.errors(include_url=False),
            },
        ) from exc

    raise UnknownMessageRoleError(role_enum)


def decode_row(
    row: Mapping[str, Any],
    *,
    status: MessageStatus | None = None,
) -> BaseMessage | RawMessage:
    """数据库行 -> 领域消息；任何协议问题降级为 RawMessage。"""
    content = row.get("content")
    if isinstance(content, str):
        import json

        try:
            content = json.loads(content)
        except ValueError:
            return _raw(row, content={}, parse_error="content 不是合法 JSON")
    if not isinstance(content, Mapping):
        return _raw(
            row,
            content={} if content is None else {"raw": str(content)},
            parse_error="content 不是 JSON 对象",
        )

    raw_status = row.get("status", status or MessageStatus.COMPLETED)
    try:
        status_enum = MessageStatus(raw_status)
    except (TypeError, ValueError):
        status_enum = MessageStatus.FAILED

    try:
        return decode_content(
            row.get("role", ""),
            content,
            message_id=UUID(str(row["id"])),
            conversation_id=UUID(str(row["conversation_id"])),
            status=status_enum,
            created_at=row.get("created_at") or utcnow(),
        )
    except (UnsupportedMessageVersionError, MessageDecodeError, UnknownMessageRoleError,
            KeyError, ValueError, TypeError) as exc:
        try:
            version = content.get("v")
        except AttributeError:
            version = None
        return _raw(
            row,
            content=dict(content),
            v=version if isinstance(version, int) else None,
            parse_error=str(exc),
        )


def decode_stored_row(row: Mapping[str, Any]) -> StoredMessage:
    message = decode_row(row)
    seq_no = int(row["seq_no"])
    return StoredMessage(seq_no=seq_no, message=message)


# ——————————————————— 内部：版本归一 ———————————————————

def _normalize_payload(
    role: MessageRole, payload: Mapping[str, Any], version: int
) -> dict[str, Any]:
    data = dict(payload)
    if version is MessageContentVersion.V1:
        data = _upgrade_v1(role, data)
    # v2 读取宽容：单个 thinking 对象也接受（防上游半升级）
    if role is MessageRole.ASSISTANT and isinstance(data.get("thinking"), Mapping):
        data["thinking"] = [data["thinking"]]
    return data


def _upgrade_v1(role: MessageRole, data: dict[str, Any]) -> dict[str, Any]:
    """v1 -> v2 的读取兼容；不改变 v1 写入方。"""
    upgraded = dict(data)
    upgraded["v"] = int(MessageContentVersion.V2)
    if role is MessageRole.SYSTEM:
        upgraded["origin"] = _system_origin(upgraded.get("origin")).value
    if role is MessageRole.ASSISTANT:
        thinking = upgraded.get("thinking")
        if isinstance(thinking, Mapping):
            upgraded["thinking"] = [thinking]
        elif thinking is None:
            upgraded["thinking"] = []
        upgraded.setdefault("artifacts", [])
        upgraded.setdefault("is_prefix", False)
        upgraded.setdefault("generation", None)
        calls = upgraded.get("tool_calls", [])
        upgraded["tool_calls"] = [
            {
                "call_id": item.get("call_id") or item.get("id"),
                "name": item.get("name"),
                "arguments": item.get("arguments") or {},
                "raw_arguments": item.get("raw_arguments"),
                "status": item.get("status", ToolCallStatus.PENDING.value),
            }
            for item in calls
        ]
    if role is MessageRole.TOOL:
        if "json" not in upgraded and "json_data" in upgraded:
            upgraded["json"] = upgraded.pop("json_data")
    return upgraded


def _system_origin(value: Any) -> SystemOrigin:
    if value is None:
        return SystemOrigin.BASE
    # 旧值 tool_policy 已废弃；读取时归到 configured，保证旧消息可展示
    if value == "tool_policy":
        return SystemOrigin.CONFIGURED
    try:
        return SystemOrigin(value)
    except ValueError:
        return SystemOrigin.BASE


def _image_ref(item: Any) -> ImageRef:
    if isinstance(item, ImageRef):
        return item
    if isinstance(item, Mapping):
        detail = item.get("detail", ImageDetail.AUTO)
        try:
            detail_enum = ImageDetail(detail)
        except (TypeError, ValueError):
            detail_enum = ImageDetail.AUTO
        return ImageRef(url=str(item.get("url", "")), detail=detail_enum)
    raise MessageDecodeError("images 项必须是对象")


def _file_ref(item: Any) -> FileRef:
    if isinstance(item, FileRef):
        return item
    if isinstance(item, Mapping):
        return FileRef(
            file_id=UUID(str(item.get("file_id"))),
            filename=item.get("filename"),
        )
    raise MessageDecodeError("files 项必须是对象")


def _thinking(item: Any) -> Thinking:
    if isinstance(item, Thinking):
        return item
    if isinstance(item, Mapping):
        return Thinking(
            text=item.get("text", ""),
            signature=item.get("signature"),
            redacted=bool(item.get("redacted", False)),
        )
    raise MessageDecodeError("thinking 项必须是对象")


def _tool_call(item: Any) -> ToolCall:
    if isinstance(item, ToolCall):
        return item
    if isinstance(item, Mapping):
        status = item.get("status", ToolCallStatus.PENDING)
        try:
            status_enum = ToolCallStatus(status)
        except (TypeError, ValueError):
            status_enum = ToolCallStatus.PENDING
        return ToolCall(
            call_id=str(item.get("call_id") or item.get("id") or ""),
            name=str(item.get("name") or ""),
            arguments=item.get("arguments") or {},
            raw_arguments=item.get("raw_arguments"),
            status=status_enum,
        )
    raise MessageDecodeError("tool_calls 项必须是对象")


def _artifact(item: Any) -> Artifact:
    if isinstance(item, Artifact):
        return item
    if isinstance(item, Mapping):
        kind = item.get("kind", ArtifactKind.OTHER)
        try:
            kind_enum = ArtifactKind(kind)
        except (TypeError, ValueError):
            kind_enum = ArtifactKind.OTHER
        return Artifact(
            kind=kind_enum,
            url=str(item.get("url", "")),
            filename=item.get("filename"),
        )
    raise MessageDecodeError("artifacts 项必须是对象")


def _raw(
    row: Mapping[str, Any],
    *,
    content: dict[str, Any],
    v: int | None = None,
    parse_error: str,
) -> RawMessage:
    status_raw = row.get("status", MessageStatus.FAILED)
    try:
        status = MessageStatus(status_raw)
    except (TypeError, ValueError):
        status = MessageStatus.FAILED
    created_at = row.get("created_at")
    if isinstance(created_at, str):
        try:
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except ValueError:
            created_at = None
    return RawMessage(
        id=UUID(str(row["id"])) if row.get("id") else UUID(int=0),
        conversation_id=UUID(str(row["conversation_id"]))
        if row.get("conversation_id")
        else UUID(int=0),
        role=row.get("role", "unknown"),
        status=status,
        created_at=created_at or utcnow(),
        content=content,
        v=v,
        parse_error=parse_error,
    )
