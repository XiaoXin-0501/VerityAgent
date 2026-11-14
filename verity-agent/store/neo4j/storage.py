"""Neo4j 图写入/删除/一跳扩展门面。

只提供 RAG 建图与检索会用到的通用操作；关系抽取、白名单过滤等业务规则
仍由 RAG 模块负责。
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from exception import Neo4jQueryError

from .connection import Neo4jConnectionManager, get_neo4j

__all__ = ["GraphStore", "relation_key"]

_SEP = "\x1f"


def relation_key(subject: str, relation: str, obj: str, kb_id: Any) -> str:
    """确定性关系键：subject\\x1frelation\\x1fobject\\x1fkb_id。"""
    return _SEP.join([subject, relation, obj, str(kb_id)])


class GraphStore:
    def __init__(self, manager: Neo4jConnectionManager | None = None) -> None:
        self._manager = manager

    @property
    def manager(self) -> Neo4jConnectionManager:
        return self._manager or get_neo4j()

    async def _write(self, query: str, parameters: Mapping[str, Any]) -> None:
        async def work(tx):
            result = await tx.run(query, **dict(parameters))
            await result.consume()

        await self.manager.execute_write(work)

    async def _read(self, query: str, parameters: Mapping[str, Any]) -> list[dict[str, Any]]:
        async def work(tx):
            result = await tx.run(query, **dict(parameters))
            return [record.data() async for record in result]

        return await self.manager.execute_read(work)

    async def upsert_chunk_ref(self, chunk_id: str, doc_id: str) -> None:
        await self._write(
            "MERGE (c:ChunkRef {chunk_id: $chunk_id}) SET c.doc_id = $doc_id",
            {"chunk_id": chunk_id, "doc_id": doc_id},
        )

    async def link_keywords(
        self,
        *,
        chunk_id: str,
        doc_id: str,
        kb_id: str,
        keywords: Iterable[str],
    ) -> None:
        values = [item for item in dict.fromkeys(keywords) if item]
        if not values:
            return
        await self._write(
            """
            MERGE (c:ChunkRef {chunk_id: $chunk_id})
            SET c.doc_id = $doc_id
            WITH c
            UNWIND $keywords AS kw
            MERGE (k:Keyword {name: kw, kb_id: $kb_id})
            MERGE (k)-[:MENTIONED_IN]->(c)
            """,
            {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "kb_id": kb_id,
                "keywords": values,
            },
        )

    async def upsert_relations(
        self,
        *,
        chunk_id: str,
        doc_id: str,
        kb_id: str,
        relations: Iterable[Mapping[str, Any]],
    ) -> None:
        values: list[dict[str, Any]] = []
        for item in relations:
            subject = str(item["subject"])
            relation = str(item["relation"])
            obj = str(item["object"])
            values.append(
                {
                    "subject": subject,
                    "relation": relation,
                    "object": obj,
                    "key": item.get("key")
                    or relation_key(subject, relation, obj, kb_id),
                    "confidence": item.get("confidence"),
                }
            )
        if not values:
            return
        await self._write(
            """
            MERGE (c:ChunkRef {chunk_id: $chunk_id})
            SET c.doc_id = $doc_id
            WITH c
            UNWIND $relations AS r
            MERGE (rel:Relation {key: r.key})
            ON CREATE SET rel.evidence_count = 0
            SET rel.relation = r.relation,
                rel.kb_id = $kb_id,
                rel.confidence = coalesce(r.confidence, rel.confidence)
            MERGE (s:Keyword {name: r.subject, kb_id: $kb_id})
            MERGE (o:Keyword {name: r.object, kb_id: $kb_id})
            MERGE (s)-[:SUBJECT_OF]->(rel)
            MERGE (o)-[:OBJECT_OF]->(rel)
            MERGE (s)-[:MENTIONED_IN]->(c)
            MERGE (o)-[:MENTIONED_IN]->(c)
            MERGE (c)-[:EVIDENCE_FOR]->(rel)
            ON CREATE SET rel.evidence_count = coalesce(rel.evidence_count, 0) + 1
            """,
            {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "kb_id": kb_id,
                "relations": values,
            },
        )

    async def delete_document(self, doc_id: str) -> None:
        """删文档图数据并清理孤儿；三步均幂等。"""
        await self._write(
            "MATCH (c:ChunkRef {doc_id: $doc_id}) DETACH DELETE c",
            {"doc_id": doc_id},
        )
        await self._write(
            """
            MATCH (rel:Relation)
            WHERE NOT (rel)<-[:EVIDENCE_FOR]-(:ChunkRef)
            DETACH DELETE rel
            """,
            {},
        )
        await self._write(
            """
            MATCH (k:Keyword)
            WHERE NOT (k)-[:MENTIONED_IN]->(:ChunkRef)
            DETACH DELETE k
            """,
            {},
        )

    async def expand_next_hops(
        self,
        *,
        seed: str,
        kb_id: str,
        allowed_relations: Iterable[str],
        min_evidence: int = 2,
        limit: int = 20,
        hub_as_intermediate: bool = False,
    ) -> list[dict[str, Any]]:
        """1 跳强关系扩展；返回下一跳实体，不返回 chunk。

        ``hub_as_intermediate=False``（默认）时排除 ``is_hub=true`` 的实体，
        避免“犬/猫/药物”这类高频 hub 把图扩展变成泛化召回。
        """
        return await self._read(
            """
            MATCH (k:Keyword {kb_id: $kb_id, name: $seed})
            MATCH (k)-[:SUBJECT_OF|OBJECT_OF]-(rel:Relation)
            WHERE rel.kb_id = $kb_id
              AND rel.relation IN $allowed_relations
              AND coalesce(rel.evidence_count, 0) >= $min_evidence
            MATCH (rel)-[:SUBJECT_OF|OBJECT_OF]-(n:Keyword)
            WHERE n.kb_id = $kb_id
              AND n <> k
              AND ($hub_as_intermediate OR coalesce(n.is_hub, false) = false)
            RETURN k.name AS seed_name,
                   rel.relation AS relation,
                   n.name AS next_hop_name,
                   rel.evidence_count AS evidence_count
            ORDER BY evidence_count DESC
            LIMIT $limit
            """,
            {
                "seed": seed,
                "kb_id": kb_id,
                "allowed_relations": list(allowed_relations),
                "min_evidence": min_evidence,
                "limit": limit,
                "hub_as_intermediate": bool(hub_as_intermediate),
            },
        )
