"""JSON / 值规范化工具。

统一约定：

- 落库、缓存、日志里的 JSON 使用 UTF-8、紧凑分隔符、``ensure_ascii=False``；
- Pydantic 模型按字段别名（``by_alias=True``）序列化，以匹配文档里的
  camelCase 结构；
- datetime / UUID / Enum / Decimal 的转换只在这一层处理，业务模块不要
  各自重复写 ``default=``。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel

from .time import ensure_utc, isoformat_z

__all__ = [
    "to_jsonable",
    "json_dumps",
    "json_loads",
    "canonical_json",
    "deep_merge",
    "without_none",
]


def to_jsonable(value: Any) -> Any:
    """递归转换成 JSON 原生类型；未知对象抛 ``TypeError``。"""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, BaseModel):
        return to_jsonable(value.model_dump(mode="python", by_alias=True))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return isoformat_z(ensure_utc(value))
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(item) for item in value]
    raise TypeError(f"无法序列化为 JSON 的类型: {type(value).__name__}")


def json_dumps(value: Any, *, sort_keys: bool = False) -> str:
    """序列化为 UTF-8 紧凑 JSON；默认保留非 ASCII 字符。"""
    return json.dumps(
        to_jsonable(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=sort_keys,
    )


def json_loads(value: str | bytes | bytearray) -> Any:
    """解析 JSON 文本/字节；异常直接抛给调用方。"""
    return json.loads(value)


def canonical_json(value: Any) -> str:
    """稳定 JSON：键排序、紧凑输出；用于哈希与缓存 key。"""
    return json_dumps(value, sort_keys=True)


def deep_merge(base: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    """递归合并，``patch`` 覆盖 ``base``；不修改入参。"""
    result = dict(base)
    for key, value in patch.items():
        current = result.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            result[key] = deep_merge(current, value)
        else:
            result[key] = value
    return result


def without_none(value: Mapping[str, Any]) -> dict[str, Any]:
    """去掉值为 None 的键（PATCH 语义）。"""
    return {key: item for key, item in value.items() if item is not None}
