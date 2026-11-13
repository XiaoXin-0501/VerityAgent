"""图存储接口与 Neo4j 实现。

RAG 只依赖 ``GraphIndexStore``；Neo4j 的 MERGE/Cypher 细节留在
``store.neo4j.GraphStore``，这里做数据类型转换与错误收敛。
"""

from __future__ import annotations

from typing import Protocol

from exception import Neo4jError
from rag.shared.models import GraphRelation
from store.neo4j import GraphStore

__all__ = ["GraphIndexStore", "Neo4jGraphIndexStore"]


class GraphIndexStore(Protocol):
    """建图所需的三个幂等操作。"""

    async def upsert_chunk(
        self,
        chunk_id: str,
        doc_id: str,
        kb_id: str,
        keywords: list[str],
    ) -> None:
        """写 ChunkRef 并连接关键词（MENTIONED_IN）。"""
        ...

    async def upsert_relations(
        self,
        chunk_id: str,
        doc_id: str,
        kb_id: str,
        relations: list[GraphRelation],
    ) -> None:
        """写关系三元组与证据边（SUBJECT_OF/OBJECT_OF/EVIDENCE_FOR）。"""
        ...

    async def delete_document(self, doc_id: str) -> None:
        """按 doc_id 删除图数据并清理孤儿节点，幂等。"""
        ...


class Neo4jGraphIndexStore:
    """``GraphStore`` 的 RAG 适配器。"""

    def __init__(self, store: GraphStore | None = None) -> None:
        """保存 Neo4j 门面；默认使用全局连接。"""
        self.store = store or GraphStore()

    async def upsert_chunk(
        self,
        chunk_id: str,
        doc_id: str,
        kb_id: str,
        keywords: list[str],
    ) -> None:
        """写入 chunk 引用与关键词；失败转成 Neo4jError 由上层处理。"""
        try:
            await self.store.link_keywords(
                chunk_id=chunk_id,
                doc_id=doc_id,
                kb_id=kb_id,
                keywords=keywords,
            )
        except Neo4jError:
            raise
        except Exception as exc:  # pragma: no cover - 网络/驱动异常兜底
            raise Neo4jError(
                "Neo4j 关键词写入失败",
                details={"cause": str(exc), "chunk_id": chunk_id},
            ) from exc

    async def upsert_relations(
        self,
        chunk_id: str,
        doc_id: str,
        kb_id: str,
        relations: list[GraphRelation],
    ) -> None:
        """写入关系三元组；空列表直接跳过。"""
        if not relations:
            return
        try:
            await self.store.upsert_relations(
                chunk_id=chunk_id,
                doc_id=doc_id,
                kb_id=kb_id,
                relations=[
                    # GraphStore 只认 Mapping，这里显式转换 Pydantic 模型。
                    {
                        "subject": item.subject,
                        "relation": item.relation,
                        "object": item.object,
                        "confidence": item.confidence,
                    }
                    for item in relations
                ],
            )
        except Neo4jError:
            raise
        except Exception as exc:  # pragma: no cover - 网络/驱动异常兜底
            raise Neo4jError(
                "Neo4j 关系写入失败",
                details={"cause": str(exc), "chunk_id": chunk_id},
            ) from exc

    async def delete_document(self, doc_id: str) -> None:
        """删除文档的图数据；幂等。"""
        try:
            await self.store.delete_document(doc_id)
        except Neo4jError:
            raise
        except Exception as exc:  # pragma: no cover - 网络/驱动异常兜底
            raise Neo4jError(
                "Neo4j 文档图删除失败",
                details={"cause": str(exc), "doc_id": doc_id},
            ) from exc
