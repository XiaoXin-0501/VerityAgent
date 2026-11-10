"""时间工具。

项目统一使用带时区的 UTC 时间（D-60）：

- 数据库列使用 ``TIMESTAMPTZ``；
- Python 侧只产生 aware datetime；
- 对外序列化固定为 UTC ISO8601，带 ``Z`` 后缀；
- 展示时区转换属于 API/前端，不进入领域模型。
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

__all__ = [
    "UTC",
    "utcnow",
    "ensure_utc",
    "to_utc",
    "isoformat_z",
    "parse_datetime",
    "today_utc",
    "seconds_since",
    "after",
]

UTC = timezone.utc


def utcnow() -> datetime:
    """当前 UTC 时间（aware）。"""
    return datetime.now(UTC)


def ensure_utc(value: datetime) -> datetime:
    """把 naive datetime 视为 UTC 并补齐时区；aware 值转换到 UTC。"""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def to_utc(value: datetime | date) -> datetime:
    """date 视为当天 UTC 零点；datetime 走 :func:`ensure_utc`。"""
    if isinstance(value, datetime):
        return ensure_utc(value)
    return datetime.combine(value, time.min, tzinfo=UTC)


def isoformat_z(value: datetime) -> str:
    """UTC ISO8601，例如 ``2026-11-08T10:00:00Z``。"""
    normalized = ensure_utc(value).replace(microsecond=0)
    return normalized.isoformat().replace("+00:00", "Z")


def parse_datetime(value: str) -> datetime:
    """解析 ISO8601；``Z`` 后缀兼容 Python 3.12 的 ``fromisoformat``。"""
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return ensure_utc(datetime.fromisoformat(text))


def today_utc() -> date:
    return utcnow().date()


def seconds_since(value: datetime, *, now: datetime | None = None) -> float:
    """距离 ``value`` 过去的秒数；未来时间返回负数。"""
    reference = ensure_utc(now or utcnow())
    return (reference - ensure_utc(value)).total_seconds()


def after(seconds: float, *, now: datetime | None = None) -> datetime:
    """``now + seconds``，用于租约到期时间。"""
    return ensure_utc(now or utcnow()) + timedelta(seconds=seconds)
