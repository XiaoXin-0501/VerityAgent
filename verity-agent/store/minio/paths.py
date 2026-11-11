"""MinIO 对象 key 约定。

对应《MinIO 对象存储数据结构设计》：

- 两个 bucket：知识库长期对象、会话临时对象；
- ``source/{scope_id}/{file_id}.{ext}`` 存原件；
- ``parse/{scope_id}/{file_id}/`` 存解析产物；
- 会话溢出/产物固定放在 ``agent-session`` 下；
- key 里不放 user_id，归属与权限走 PG / 预签名 URL。
"""

from __future__ import annotations

from typing import Any

from exception import ValidationError

__all__ = [
    "sanitize_extension",
    "kb_source_key",
    "kb_parse_prefix",
    "kb_artifact_key",
    "session_source_key",
    "session_parse_prefix",
    "session_overflow_key",
    "session_artifact_key",
    "source_key",
    "parse_prefix",
]


def _segment(value: Any, field: str) -> str:
    text = str(value).strip()
    if not text or "/" in text or "\\" in text:
        raise ValidationError(
            f"{field} 不能为空且不能包含路径分隔符",
            details={"field": field, "value": text},
        )
    return text


def sanitize_extension(extension: str | None, *, default: str = "bin") -> str:
    """扩展名统一为小写、不带点；禁止路径字符。"""
    if not extension:
        return default
    text = extension.strip().lower().lstrip(".")
    if not text:
        return default
    if "/" in text or "\\" in text or "." in text:
        raise ValidationError("扩展名不合法", details={"extension": extension})
    return text[:20]


def source_key(scope_id: Any, file_id: Any, extension: str | None) -> str:
    """``source/{scope_id}/{file_id}.{ext}``。"""
    return (
        f"source/{_segment(scope_id, 'scope_id')}/"
        f"{_segment(file_id, 'file_id')}.{sanitize_extension(extension)}"
    )


def parse_prefix(scope_id: Any, file_id: Any) -> str:
    """``parse/{scope_id}/{file_id}/``；所有解析产物共用这个前缀。"""
    return f"parse/{_segment(scope_id, 'scope_id')}/{_segment(file_id, 'file_id')}/"


def kb_source_key(kb_id: Any, doc_id: Any, extension: str | None) -> str:
    return source_key(kb_id, doc_id, extension)


def kb_parse_prefix(kb_id: Any, doc_id: Any) -> str:
    return parse_prefix(kb_id, doc_id)


def kb_artifact_key(kb_id: Any, artifact_id: Any, extension: str | None) -> str:
    return (
        f"artifact/{_segment(kb_id, 'kb_id')}/"
        f"{_segment(artifact_id, 'artifact_id')}.{sanitize_extension(extension)}"
    )


def session_source_key(
    conversation_id: Any, file_id: Any, extension: str | None
) -> str:
    return source_key(conversation_id, file_id, extension)


def session_parse_prefix(conversation_id: Any, file_id: Any) -> str:
    return parse_prefix(conversation_id, file_id)


def session_overflow_key(
    conversation_id: Any, call_id: Any, extension: str = "json"
) -> str:
    return (
        f"overflow/{_segment(conversation_id, 'conversation_id')}/"
        f"{_segment(call_id, 'call_id')}.{sanitize_extension(extension, default='json')}"
    )


def session_artifact_key(
    conversation_id: Any, artifact_id: Any, extension: str | None
) -> str:
    return (
        f"artifact/{_segment(conversation_id, 'conversation_id')}/"
        f"{_segment(artifact_id, 'artifact_id')}.{sanitize_extension(extension)}"
    )
