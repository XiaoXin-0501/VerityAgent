"""消息存储视图的编解码。

- :func:`encode_content`：领域消息 → ``conversation_messages.content`` JSONB；
- :func:`decode_content`：content JSONB + 行公共字段 → 领域消息；
- :func:`decode_row` / :func:`decode_stored_row`：数据库行 → 消息 / ``StoredMessage``。

约定：

- content 根节点带 ``v``（当前为 1）；未知版本抛
  :class:`~exception.message.UnsupportedMessageVersionError`；
- 读取时只取协议内已知字段，未知扩展字段被忽略（读宽容）；
- 公共字段（id / conversation_id / role / status / created_at）来自表列，
  不重复写进 content；``seq_no`` 只进入 ``StoredMessage``。
"""

from __future__ import annotations

import json
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

from .enum import MessageRole, MessageStatus
from .models import (
    AssistantMessage,
    BaseMessage,
    StoredMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)

MESSAGE_CONTENT_VERSION = 1
_SUPPORTED_VERSIONS = frozenset({MESSAGE_CONTENT_VERSION})

# content 中除公共 text 外，各 role 需要额外写入/读出的字段。
# ToolMessage 的 Python 属性是 json_data，存储协议 key 是 json。
_ROLE_FIELDS: dict[MessageRole, tuple[str, ...]] = {
    MessageRole.SYSTEM: ("name", "origin"),
    MessageRole.USER: ("images", "files"),
    MessageRole.ASSISTANT: ("thinking", "tool_calls", "artifacts", "is_prefix"),
    MessageRole.TOOL: ("tool_call_id", "name", "json", "files", "images"),
}

_ROLE_CLASSES: dict[MessageRole, type[BaseMessage]] = {
    MessageRole.SYSTEM: SystemMessage,
    MessageRole.USER: UserMessage,
    MessageRole.ASSISTANT: AssistantMessage,
    MessageRole.TOOL: ToolMessage,
}


def encode_content(message: BaseMessage) -> dict[str, Any]:
    """把领域消息编码成 ``conversation_messages.content``。

    只编码角色私有字段 + ``text``；id / conversation_id / role / status /
    created_at 由表列承载，不在这里重复。
    """
    role = message.role
    fields = _ROLE_FIELDS.get(role)
    if fields is None:
        raise UnknownMessageRoleError(role)

    try:
        dumped = message.model_dump(mode="json", by_alias=True)
    except (TypeError, ValueError) as exc:
        raise MessageEncodeError(
            "消息 content 序列化失败",
            detail={"role": str(role), "error": str(exc)},
        ) from exc

    payload: dict[str, Any] = {
        "v": MESSAGE_CONTENT_VERSION,
        "text": dumped.get("content") or "",
    }
    for name in fields:
        payload[name] = dumped.get(name)
    return payload


def decode_content(
    role: MessageRole | str,
    payload: Mapping[str, Any],
    *,
    message_id: UUID | str,
    conversation_id: UUID | str,
    status: MessageStatus | str = MessageStatus.COMPLETED,
    created_at: datetime | str,
) -> BaseMessage:
    """把 content JSONB 还原成对应的领域消息。

    公共字段必须由调用方从表列传入；content 里只有角色私有字段和 text。
    """
    if not isinstance(payload, Mapping):
        raise MessageDecodeError(
            "消息 content 必须是 JSON 对象",
            detail={"type": type(payload).__name__},
        )

    try:
        role_enum = role if isinstance(role, MessageRole) else MessageRole(role)
    except ValueError as exc:
        raise UnknownMessageRoleError(role) from exc

    version = payload.get("v", MESSAGE_CONTENT_VERSION)
    if version not in _SUPPORTED_VERSIONS:
        raise UnsupportedMessageVersionError(version)

    data = {
        name: payload[name]
        for name in _ROLE_FIELDS[role_enum]
        if name in payload
    }
    base: dict[str, Any] = {
        "id": message_id,
        "conversation_id": conversation_id,
        "status": status,
        "created_at": created_at,
        "content": payload.get("text") or "",
    }

    message_cls = _ROLE_CLASSES[role_enum]
    try:
        return message_cls(**base, **data)
    except PydanticValidationError as exc:
        raise MessageDecodeError(
            "消息 content 不符合协议",
            detail={
                "role": role_enum.value,
                "version": version,
                "errors": exc.errors(
                    include_url=False,
                    include_context=False,
                    include_input=False,
                ),
            },
        ) from exc


def decode_row(row: Mapping[str, Any]) -> BaseMessage:
    """把 ``conversation_messages`` 行还原成领域消息。

    row 需要包含：id / conversation_id / role / status / created_at / content。
    """
    try:
        message_id = row["id"]
        conversation_id = row["conversation_id"]
        role = row["role"]
        status = row.get("status", MessageStatus.COMPLETED)
        created_at = row["created_at"]
        content = row["content"]
    except KeyError as exc:
        raise MessageDecodeError(
            "消息行缺少字段",
            detail={"missing": str(exc)},
        ) from exc

    if isinstance(content, str):
        try:
            content = json.loads(content)
        except json.JSONDecodeError as exc:
            raise MessageDecodeError(
                "消息 content 不是合法 JSON",
                detail={"error": str(exc)},
            ) from exc

    return decode_content(
        role,
        content,
        message_id=message_id,
        conversation_id=conversation_id,
        status=status,
        created_at=created_at,
    )


def decode_stored_row(row: Mapping[str, Any]) -> StoredMessage:
    """把带 ``seq_no`` 的数据库行还原成 ``StoredMessage``。"""
    seq_no = row.get("seq_no")
    if isinstance(seq_no, bool) or not isinstance(seq_no, int) or seq_no <= 0:
        raise MessageDecodeError(
            "消息行 seq_no 非法",
            detail={"seq_no": seq_no},
        )
    return StoredMessage(seq_no=seq_no, message=decode_row(row))


__all__ = [
    "MESSAGE_CONTENT_VERSION",
    "encode_content",
    "decode_content",
    "decode_row",
    "decode_stored_row",
]
