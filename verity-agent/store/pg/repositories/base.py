"""仓库层公共能力。

- 统一借连接 / 开事务，所有方法支持传入外部 ``conn``，以便跨仓库
  在同一事务里完成双写（例如 conversation_messages + user_messages）；
- 统一把 ``psycopg.Error`` 翻译成 ``exception.pg`` 里的类型；
- 统一 JSONB 写入（``Jsonb``）与行 -> 模型转换；
- 统一分页、状态迁移等入参校验。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from string import hexdigits
from typing import Any, TypeVar, cast
from uuid import UUID

import psycopg
from psycopg import AsyncConnection
from psycopg.types.json import Jsonb
from pydantic import BaseModel

from exception import (
    InvalidStateTransitionError,
    PGCheckViolationError,
    PGConnectionError,
    PGError,
    PGForeignKeyViolationError,
    PGIntegrityError,
    PGNotNullViolationError,
    PGQueryError,
    PGSerializationError,
    PGUniqueViolationError,
    ValidationError,
)

from ..connection import Row, get_pg

_UNSET: Any = object()
"""用于 UPDATE 的"未提供该字段"哨兵；区别于显式传 None。"""


# ——————————————————— 错误翻译 ———————————————————

_SQLSTATE_ERRORS: dict[str, type[PGError]] = {
    "23505": PGUniqueViolationError,
    "23503": PGForeignKeyViolationError,
    "23502": PGNotNullViolationError,
    "23514": PGCheckViolationError,
    "40001": PGSerializationError,
    "40P01": PGSerializationError,
}


def translate_pg_error(exc: psycopg.Error) -> PGError:
    """把 psycopg 异常翻译成项目自定义异常。"""
    diag = getattr(exc, "diag", None)
    details: dict[str, Any] = {}
    if diag is not None:
        for key, attr in (
            ("sqlstate", "sqlstate"),
            ("constraint", "constraint_name"),
            ("table", "table_name"),
            ("column", "column_name"),
            ("detail", "message_detail"),
        ):
            value = getattr(diag, attr, None) or getattr(exc, key, None)
            if value:
                details[key] = value

    if isinstance(exc, (psycopg.OperationalError, psycopg.InterfaceError)):
        return PGConnectionError(str(exc), details=details)

    error_cls = _SQLSTATE_ERRORS.get(
        getattr(exc, "sqlstate", "") or "",
        PGIntegrityError if isinstance(exc, psycopg.IntegrityError) else PGQueryError,
    )
    return error_cls(str(exc), details=details)


# ——————————————————— JSONB / 类型适配 ———————————————————

def _json_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", by_alias=True)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Decimal):
        return float(value)
    raise TypeError(f"无法序列化为 JSON 的类型: {type(value).__name__}")


def to_jsonb(value: Any) -> Jsonb:
    """把 Pydantic 模型 / dict / list 包装成 psycopg 可写入 JSONB 的参数。"""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", by_alias=True)
    return Jsonb(
        value,
        dumps=lambda obj: json.dumps(
            obj, ensure_ascii=False, separators=(",", ":"), default=_json_default
        ),
    )


def normalize_message_content(content: Any) -> dict[str, Any]:
    """校验并规范化消息 content（只做存储层的最低限度检查）。

    消息协议的完整校验在 ``agent.message.codec``（v2；v1 兼容读取）。
    仓储层允许调用方直接传字符串、领域消息或已序列化的 dict，避免存储层
    反向依赖消息领域模型而破坏分层。
    """
    # 领域消息走统一 codec，保证落库一定是 v2 协议
    try:
        from agent.message import BaseMessage, encode_content

        if isinstance(content, BaseMessage):
            return encode_content(content)
    except ImportError:  # pragma: no cover - 消息模块总是可用
        pass

    if isinstance(content, BaseModel):
        content = content.model_dump(mode="json", by_alias=True)
    if isinstance(content, str):
        content = {"v": 2, "text": content}
    if not isinstance(content, Mapping):
        raise ValidationError(
            "消息 content 必须是对象或字符串",
            details={"type": type(content).__name__},
        )

    payload = dict(content)
    version = payload.setdefault("v", 2)
    if not isinstance(version, int) or version < 1:
        raise ValidationError(
            "消息 content 缺少合法的协议版本 v",
            details={"v": version},
        )
    text = payload.get("text", "")
    if text is None:
        text = ""
    if not isinstance(text, str):
        raise ValidationError(
            "消息 content.text 必须是字符串",
            details={"type": type(text).__name__},
        )
    payload["text"] = text
    return payload


def normalize_extension(extension: str) -> str:
    """统一扩展名为小写、不带点。"""
    value = extension.strip().lower().lstrip(".")
    if not value:
        raise ValidationError("extension 不能为空")
    if len(value) > 20:
        raise ValidationError("extension 长度不能超过 20")
    return value


def normalize_sha256(sha256: str | None) -> str | None:
    """校验 64 位十六进制内容哈希。"""
    if sha256 is None:
        return None
    value = sha256.strip().lower()
    if len(value) != 64 or any(ch not in hexdigits for ch in value):
        raise ValidationError("sha256 必须是 64 位十六进制字符串")
    return value


def require_text(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{field} 不能为空")
    return value


def validate_title(title: str, *, max_length: int = 300) -> str:
    """会话标题：允许为空字符串（后续自动生成），但不能超长。"""
    if not isinstance(title, str):
        raise ValidationError("title 必须是字符串")
    if len(title) > max_length:
        raise ValidationError(f"title 长度不能超过 {max_length}")
    return title


def pagination(limit: int, offset: int = 0) -> tuple[int, int]:
    if not isinstance(limit, int) or limit <= 0:
        raise ValidationError("limit 必须是正整数")
    if not isinstance(offset, int) or offset < 0:
        raise ValidationError("offset 不能为负数")
    return limit, offset


def as_uuid(value: UUID | str) -> UUID:
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"非法 UUID: {value!r}") from exc


def ensure_transition(
    current: str | Enum,
    target: str | Enum,
    transitions: Mapping[Any, frozenset[Any]],
    *,
    resource: str,
) -> None:
    """校验状态机迁移；同状态视为幂等，直接放行。"""
    if current == target:
        return
    allowed = transitions.get(current, frozenset())
    if target not in allowed:
        raise InvalidStateTransitionError(resource, current, target)


def _param(value: Any) -> Any:
    """把枚举转成字符串，其余原样传给 psycopg。"""
    if isinstance(value, Enum):
        return value.value
    return value


# ——————————————————— 基类 ———————————————————

_ModelT = TypeVar("_ModelT", bound=BaseModel)


class BaseRepository:
    """所有仓库的基类。

    ``db`` 不传时使用进程内单例（``get_pg()``）。方法上的 ``conn`` 参数用于
    把多个仓库的写操作合并进同一个事务::

        async with pg.transaction() as conn:
            msg = await msg_repo.append(..., conn=conn)
            await user_msg_repo.enqueue(..., conn=conn)
    """

    def __init__(self, db: Any | None = None) -> None:
        self._db = db

    @property
    def db(self):
        if self._db is None:
            self._db = get_pg()
        return self._db

    @asynccontextmanager
    async def _transaction(
        self, conn: AsyncConnection[Row] | None = None
    ) -> AsyncIterator[AsyncConnection[Row]]:
        """有外部连接时复用（不提交），否则自开事务。"""
        if conn is not None:
            yield conn
            return
        async with self.db.transaction() as own_conn:
            yield own_conn

    async def _fetch_one(
        self,
        conn: AsyncConnection[Row],
        sql: str,
        params: Sequence[Any] | Mapping[str, Any] | None = None,
    ) -> Row | None:
        try:
            cursor = await conn.execute(cast(Any, sql), params)
            return await cursor.fetchone()
        except psycopg.Error as exc:
            raise translate_pg_error(exc) from exc

    async def _fetch_all(
        self,
        conn: AsyncConnection[Row],
        sql: str,
        params: Sequence[Any] | Mapping[str, Any] | None = None,
    ) -> list[Row]:
        try:
            cursor = await conn.execute(cast(Any, sql), params)
            return list(await cursor.fetchall())
        except psycopg.Error as exc:
            raise translate_pg_error(exc) from exc

    async def _fetch_val(
        self,
        conn: AsyncConnection[Row],
        sql: str,
        params: Sequence[Any] | Mapping[str, Any] | None = None,
    ) -> Any:
        row = await self._fetch_one(conn, sql, params)
        if not row:
            return None
        return next(iter(row.values()))

    async def _execute(
        self,
        conn: AsyncConnection[Row],
        sql: str,
        params: Sequence[Any] | Mapping[str, Any] | None = None,
    ) -> int:
        try:
            cursor = await conn.execute(cast(Any, sql), params)
            return cursor.rowcount
        except psycopg.Error as exc:
            raise translate_pg_error(exc) from exc

    @staticmethod
    def _model(model_cls: type[_ModelT], row: Row | None) -> _ModelT | None:
        return None if row is None else model_cls.model_validate(row)

    @staticmethod
    def _models(model_cls: type[_ModelT], rows: Sequence[Row]) -> list[_ModelT]:
        return [model_cls.model_validate(row) for row in rows]
