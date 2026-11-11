"""稳定采样。

同一 ``key``（通常是 ``trace_id + hook_name``）在相同 rate 下结果稳定，
避免同一链路一部分 Hook 执行、一部分不执行。
"""

from __future__ import annotations

import hashlib

__all__ = ["stable_sample"]

_BUCKETS = 10_000


def stable_sample(key: str, rate: float) -> bool:
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % _BUCKETS
    return bucket < int(rate * _BUCKETS)
