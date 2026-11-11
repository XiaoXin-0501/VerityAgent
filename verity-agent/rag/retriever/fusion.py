"""混合召回融合（RRF）。

输入是按 score 降序排列的候选列表；相同 chunk_id 在不同通道出现会累加
``1 / (k + rank)``，无需把 BM25/向量分数归一化到同一量纲。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, TypeVar

__all__ = ["rrf_fuse"]

_T = TypeVar("_T")


def rrf_fuse(
    rankings: Iterable[Sequence[Any]],
    *,
    key,
    k: int = 60,
) -> list[tuple[Any, float]]:
    """返回按 RRF 分数降序的 ``(item, score)`` 列表。

    ``rankings`` 中每个序列已按各自通道的相关性降序排列；重复 item 以首次
    出现对象为准。
    """
    if k < 0:
        raise ValueError("rrf k 不能为负数")
    scores: dict[Any, float] = {}
    items: dict[Any, Any] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            identity = key(item)
            items.setdefault(identity, item)
            scores[identity] = scores.get(identity, 0.0) + 1.0 / (k + rank)
    ordered = sorted(scores.items(), key=lambda pair: (-pair[1], str(pair[0])))
    return [(items[identity], score) for identity, score in ordered]
