"""混合召回融合（RRF）。

输入是按 score 降序排列的候选列表；相同 chunk_id 在不同通道出现会累加
``1 / (k + rank)``，无需把 BM25/向量分数归一化到同一量纲。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, TypeVar

__all__ = ["rrf_fuse"]

_T = TypeVar("_T")


def rrf_fuse(
    rankings: Iterable[Sequence[Any]],
    *,
    key,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[Any, float]]:
    """返回按 RRF 分数降序的 ``(item, score)`` 列表。

    ``rankings`` 中每个序列已按各自通道的相关性降序排列；重复 item 以首次
    出现对象为准。``weights`` 可给每个通道加权（例如 BM25/向量按
    ``HybridPolicy.alpha`` 加权），为空时所有通道等权；权重按位置与
    ``rankings`` 一一对应，长度不一致直接报错，避免静默错配。
    """
    if k < 0:
        raise ValueError("rrf k 不能为负数")
    channels = list(rankings)
    if weights is None:
        channel_weights = [1.0] * len(channels)
    else:
        channel_weights = [float(item) for item in weights]
        if len(channel_weights) != len(channels):
            raise ValueError("RRF weights 与 rankings 数量不一致")
        if any(item < 0 for item in channel_weights):
            raise ValueError("RRF 通道权重不能为负数")

    scores: dict[Any, float] = {}
    items: dict[Any, Any] = {}
    for ranking, weight in zip(channels, channel_weights, strict=True):
        if weight == 0:
            continue
        for rank, item in enumerate(ranking, start=1):
            identity = key(item)
            items.setdefault(identity, item)
            scores[identity] = (
                scores.get(identity, 0.0) + weight * (1.0 / (k + rank))
            )
    ordered = sorted(scores.items(), key=lambda pair: (-pair[1], str(pair[0])))
    return [(items[identity], score) for identity, score in ordered]
