"""UserFact Weaviate 适配器（只读召回）。"""

from __future__ import annotations

import logging
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from store.weaviate import WeaviateStore
from store.pg.models.enums import MAX_USER_FACTS
from utils.text import normalize_whitespace
from utils.time import utcnow

__all__ = ["WeaviateFactStore"]

logger = logging.getLogger(__name__)


class WeaviateFactStore:
    """按用户 + status=active 做纯 BM25 事实召回。"""

    def __init__(self, store: WeaviateStore | None = None) -> None:
        """保存 Weaviate 门面；不主动建立连接。"""
        self.store = store or WeaviateStore()

    async def recall(
        self,
        *,
        user_id: Any,
        query: str,
        top_k: int = 5,
        min_confidence: float | None = None,
    ) -> list[dict[str, Any]]:
        """返回事实属性与 BM25 分数；集合不存在时抛给调用方降级。"""
        from weaviate.classes.query import Filter, MetadataQuery

        where = Filter.by_property("user_id").equal(str(user_id)) & (
            Filter.by_property("status").equal("active")
        )
        collection = self.store.user_facts
        response = await self.store.maybe_to_thread(
            collection.query.bm25,
            query=query,
            query_properties=["content"],
            limit=max(1, int(top_k)),
            filters=where,
            return_metadata=MetadataQuery(score=True),
        )
        hits: list[dict[str, Any]] = []
        for item in getattr(response, "objects", []) or []:
            props = dict(getattr(item, "properties", {}) or {})
            confidence = float(props.get("confidence") or 0.0)
            if min_confidence is not None and confidence < min_confidence:
                continue
            metadata = getattr(item, "metadata", None)
            hits.append(
                {
                    "id": str(getattr(item, "uuid", "")),
                    "content": str(props.get("content") or ""),
                    "category": props.get("category"),
                    "confidence": confidence,
                    "score": float(getattr(metadata, "score", 0.0) or 0.0),
                    "source_msg_ids": props.get("source_msg_ids") or [],
                }
            )
        hits.sort(key=lambda item: item["score"], reverse=True)
        return hits

    @staticmethod
    def fact_id_for(user_id: Any, content: str) -> str:
        """返回事实确定性 id；写入/冲突判定共用。"""
        normalized = normalize_whitespace(content, keep_newlines=False).lower()
        return str(uuid5(NAMESPACE_URL, f"{user_id}{chr(0)}{normalized}"))

    async def find_similar(
        self,
        *,
        user_id: Any,
        content: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """按新事实内容 BM25 召回当前用户 active 候选。"""
        from weaviate.classes.query import Filter, MetadataQuery

        collection = self.store.user_facts
        where = Filter.by_property("user_id").equal(str(user_id)) & (
            Filter.by_property("status").equal("active")
        )
        response = await self.store.maybe_to_thread(
            collection.query.bm25,
            query=content,
            query_properties=["content"],
            limit=max(1, int(limit)),
            filters=where,
            return_metadata=MetadataQuery(score=True),
        )
        candidates: list[dict[str, Any]] = []
        for item in getattr(response, "objects", []) or []:
            props = dict(getattr(item, "properties", {}) or {})
            metadata = getattr(item, "metadata", None)
            candidates.append(
                {
                    "id": str(getattr(item, "uuid", "")),
                    "content": str(props.get("content") or ""),
                    "category": props.get("category"),
                    "confidence": float(props.get("confidence") or 0.0),
                    "context": props.get("context"),
                    "scope": props.get("scope"),
                    "score": float(getattr(metadata, "score", 0.0) or 0.0),
                }
            )
        return candidates

    async def supersede(
        self,
        *,
        user_id: Any,
        old_fact_id: str,
        new_fact_id: str,
    ) -> bool:
        """把旧事实标记为 superseded，指向新事实 id。"""
        collection = self.store.user_facts
        existing = await self.store.maybe_to_thread(
            collection.query.fetch_object_by_id,
            old_fact_id,
        )
        if existing is None or str(
            (getattr(existing, "properties", {}) or {}).get("user_id")
        ) != str(user_id):
            return False
        now = utcnow().isoformat()
        await self.store.maybe_to_thread(
            collection.data.update,
            uuid=old_fact_id,
            properties={
                "status": "superseded",
                "superseded_by": new_fact_id,
                "deprecated_at": now,
                "updated_at": now,
            },
        )
        return True

    async def upsert_facts(
        self,
        *,
        user_id: Any,
        facts: list[dict[str, Any]],
        source_msg_ids: list[str],
    ) -> int:
        """按确定性 id 写入/更新 active 事实；返回处理条数。"""
        if not facts:
            return 0
        collection = self.store.user_facts
        now = utcnow().isoformat()
        written = 0
        for fact in facts:
            content = normalize_whitespace(str(fact.get("content") or ""), keep_newlines=False)
            if not content:
                continue
            fact_id = uuid5(NAMESPACE_URL, f"{user_id}{chr(0)}{content.lower()}")
            properties = _fact_properties(
                fact,
                content=content,
                user_id=user_id,
                source_msg_ids=source_msg_ids,
                now=now,
            )
            try:
                existing = await self.store.maybe_to_thread(
                    collection.query.fetch_object_by_id,
                    fact_id,
                )
            except Exception:  # noqa: BLE001 - 不存在通常以异常返回
                existing = None
            if existing is None:
                await self.store.maybe_to_thread(
                    collection.data.insert,
                    properties=properties,
                    uuid=fact_id,
                )
            else:
                old = dict(getattr(existing, "properties", {}) or {})
                merged_sources = sorted(
                    {
                        *(str(item) for item in old.get("source_msg_ids") or []),
                        *source_msg_ids,
                    }
                )
                properties.update(
                    {
                        "revision": int(old.get("revision") or 0) + 1,
                        "source_msg_ids": merged_sources,
                        "confirmation_count": len(merged_sources),
                        "created_at": old.get("created_at") or now,
                    }
                )
                await self.store.maybe_to_thread(
                    collection.data.update,
                    uuid=fact_id,
                    properties=properties,
                )
            written += 1
        return written

    async def deprecate_facts(
        self,
        *,
        user_id: Any,
        fact_ids: list[str],
    ) -> int:
        """软删除事实；not-found 视为成功。"""
        collection = self.store.user_facts
        now = utcnow().isoformat()
        changed = 0
        for fact_id in fact_ids:
            try:
                existing = await self.store.maybe_to_thread(
                    collection.query.fetch_object_by_id,
                    fact_id,
                )
                if existing is None or str(
                    (getattr(existing, "properties", {}) or {}).get("user_id")
                ) != str(user_id):
                    continue
                await self.store.maybe_to_thread(
                    collection.data.update,
                    uuid=fact_id,
                    properties={
                        "status": "deprecated",
                        "deprecated_at": now,
                        "updated_at": now,
                    },
                )
                changed += 1
            except Exception:  # noqa: BLE001 - 删除幂等，缺失按成功
                logger.debug("UserFact 软删除跳过: %s", fact_id)
        return changed

    async def enforce_fact_limit(
        self,
        *,
        user_id: Any,
        max_facts: int = MAX_USER_FACTS,
    ) -> int:
        """超过上限时淘汰最久未确认的 active 事实；返回淘汰数量。"""
        from weaviate.classes.query import Filter, Sort

        collection = self.store.user_facts
        where = Filter.by_property("user_id").equal(str(user_id)) & (
            Filter.by_property("status").equal("active")
        )
        active = await self.count_active(user_id=user_id)
        overflow = max(0, active - max(1, int(max_facts)))
        if overflow == 0:
            return 0
        response = await self.store.maybe_to_thread(
            collection.query.fetch_objects,
            filters=where,
            limit=overflow,
            sort=Sort.by_property("last_confirmed_at", ascending=True),
        )
        ids = [
            str(getattr(item, "uuid", ""))
            for item in getattr(response, "objects", []) or []
        ]
        return await self.deprecate_facts(user_id=user_id, fact_ids=ids)

    async def penalize_facts(
        self,
        *,
        user_id: Any,
        fact_ids: list[str],
        delta: float = 0.1,
        disputed_threshold: float = 0.4,
    ) -> int:
        """下调事实置信度；低于阈值置 disputed，返回修改条数。"""
        collection = self.store.user_facts
        changed = 0
        now = utcnow().isoformat()
        for fact_id in fact_ids:
            existing = await self.store.maybe_to_thread(
                collection.query.fetch_object_by_id,
                fact_id,
            )
            props = dict(getattr(existing, "properties", {}) or {}) if existing else {}
            if not existing or str(props.get("user_id")) != str(user_id):
                continue
            old = float(props.get("confidence") or 0.0)
            confidence = max(0.0, old - max(0.0, float(delta)))
            status = "disputed" if confidence < disputed_threshold else props.get("status", "active")
            await self.store.maybe_to_thread(
                collection.data.update,
                uuid=fact_id,
                properties={
                    "confidence": confidence,
                    "status": status,
                    "updated_at": now,
                },
            )
            changed += 1
        return changed

    async def count_active(self, *, user_id: Any) -> int:
        """统计用户 active 事实数；用于 PG fact_count 对账。"""
        from weaviate.classes.query import Filter

        collection = self.store.user_facts
        where = Filter.by_property("user_id").equal(str(user_id)) & (
            Filter.by_property("status").equal("active")
        )
        result = await self.store.maybe_to_thread(
            collection.aggregate.over_all,
            total_count=True,
            filters=where,
        )
        return int(getattr(result, "total_count", 0) or 0)


def _fact_properties(
    fact: dict[str, Any],
    *,
    content: str,
    user_id: Any,
    source_msg_ids: list[str],
    now: str,
) -> dict[str, Any]:
    """把抽取 fact 规范成 UserFact properties。"""
    confidence = float(fact.get("confidence") or 0.5)
    confidence = min(1.0, max(0.0, confidence))
    category = str(fact.get("category") or "history")
    if category not in {"work", "personal", "preference", "constraint", "goal", "history"}:
        category = "history"
    sources = sorted({str(item) for item in source_msg_ids if str(item)})
    return {
        "user_id": str(user_id),
        "revision": 0,
        "content": content,
        "category": category,
        "confidence": confidence,
        "confirmation_count": len(sources),
        "last_confirmed_at": now,
        "source_msg_ids": sources,
        "context": fact.get("context"),
        "scope": fact.get("scope"),
        "source_type": fact.get("source_type") or "user_inferred",
        "status": "active",
        "created_at": now,
        "updated_at": now,
    }
