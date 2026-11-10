"""跨模块通用工具。

只放**无业务语义**的纯函数/小数据类型；与某个模块强相关的规则继续留在该
模块（例如消息协议在 ``agent.message``、缓存 key 在 ``store.redis.keys``）。
"""

from .hashing import (
    join_digests,
    sha256_bytes,
    sha256_file,
    sha256_json,
    sha256_text,
    short_digest,
    stable_digest,
)
from .ids import (
    new_idempotency_key,
    new_request_id,
    new_run_id,
    new_token,
    new_turn_id,
    new_uuid,
    uuid7,
)
from .pagination import (
    DEFAULT_LIMIT,
    MAX_LIMIT,
    Page,
    PageRequest,
    decode_cursor,
    encode_cursor,
    normalize_limit,
)
from .serialization import (
    canonical_json,
    deep_merge,
    json_dumps,
    json_loads,
    to_jsonable,
    without_none,
)
from .text import (
    ensure_prefix,
    mask_secret,
    normalize_whitespace,
    strip_control_chars,
    truncate,
)
from .time import (
    UTC,
    after,
    ensure_utc,
    isoformat_z,
    parse_datetime,
    seconds_since,
    today_utc,
    to_utc,
    utcnow,
)
from .tokens import estimate_tokens, fits_budget, truncate_to_tokens

__all__ = [
    # ids
    "uuid7",
    "new_uuid",
    "new_token",
    "new_request_id",
    "new_run_id",
    "new_turn_id",
    "new_idempotency_key",
    # time
    "UTC",
    "utcnow",
    "ensure_utc",
    "to_utc",
    "isoformat_z",
    "parse_datetime",
    "today_utc",
    "seconds_since",
    "after",
    # serialization
    "to_jsonable",
    "json_dumps",
    "json_loads",
    "canonical_json",
    "deep_merge",
    "without_none",
    # hashing
    "sha256_bytes",
    "sha256_text",
    "sha256_json",
    "sha256_file",
    "stable_digest",
    "short_digest",
    "join_digests",
    # tokens
    "estimate_tokens",
    "fits_budget",
    "truncate_to_tokens",
    # text
    "normalize_whitespace",
    "strip_control_chars",
    "truncate",
    "ensure_prefix",
    "mask_secret",
    # pagination
    "Page",
    "PageRequest",
    "DEFAULT_LIMIT",
    "MAX_LIMIT",
    "normalize_limit",
    "encode_cursor",
    "decode_cursor",
]
