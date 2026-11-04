"""消息领域异常。

消息模型负责把“调用方写错的消息”与“存储里的消息协议损坏”区分开：

- :class:`MessageValidationError`：业务校验失败（HTTP 400）；
- :class:`MessageEncodeError`：领域消息无法序列化成存储协议（HTTP 500）；
- :class:`MessageDecodeError`：存储协议无法还原成领域消息（HTTP 500）；
- :class:`UnsupportedMessageVersionError` / :class:`UnknownMessageRoleError`：
  解码时的两种常见协议错误，带机器可读 detail。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import VerityError


class MessageError(VerityError):
    """消息相关异常基类。"""

    code = "message_error"
    http_status = 400
    default_message = "消息数据不合法"


class MessageValidationError(MessageError):
    """消息内容不满足业务约束。"""

    code = "message_validation_error"
    default_message = "消息数据不合法"


class MessageEncodeError(MessageError):
    """领域消息序列化为存储协议失败。"""

    code = "message_encode_error"
    http_status = 500
    default_message = "消息序列化失败"


class MessageDecodeError(MessageError):
    """存储协议反序列化为领域消息失败。"""

    code = "message_decode_error"
    http_status = 500
    default_message = "消息反序列化失败"


class UnsupportedMessageVersionError(MessageDecodeError):
    """content 的协议版本不在支持范围内。"""

    code = "unsupported_message_version"
    default_message = "不支持的消息版本"

    def __init__(
        self,
        version: object,
        *,
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        merged: dict[str, Any] = {"version": version}
        merged.update(detail or {})
        super().__init__(f"不支持的消息版本: {version}", detail=merged)


class UnknownMessageRoleError(MessageDecodeError):
    """role 不在 user / assistant / tool / system 范围内。"""

    code = "unknown_message_role"
    default_message = "未知的消息角色"

    def __init__(
        self,
        role: object,
        *,
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        merged: dict[str, Any] = {"role": str(role)}
        merged.update(detail or {})
        super().__init__(f"未知的消息角色: {role}", detail=merged)


__all__ = [
    "MessageError",
    "MessageValidationError",
    "MessageEncodeError",
    "MessageDecodeError",
    "UnsupportedMessageVersionError",
    "UnknownMessageRoleError",
]
