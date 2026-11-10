"""Neo4j 约束/索引初始化（《Neo4j 数据结构设计》§6）。"""

from __future__ import annotations

from typing import Any

from exception import Neo4jSchemaError

from .connection import Neo4jConnectionManager

__all__ = ["GRAPH_SCHEMA_VERSION", "SCHEMA_STATEMENTS", "init_schema", "check_schema"]

GRAPH_SCHEMA_VERSION = "2026-11-09"

# 所有语句都使用 IF NOT EXISTS，可重复执行
SCHEMA_STATEMENTS: tuple[str, ...] = (
    # 概念节点唯一：同一概念在不同 KB 是不同节点
    "CREATE CONSTRAINT keyword_name_kb IF NOT EXISTS "
    "FOR (k:Keyword) REQUIRE (k.name, k.kb_id) IS UNIQUE",
    # 来源 chunk 指针唯一
    "CREATE CONSTRAINT chunk_ref_id IF NOT EXISTS "
    "FOR (c:ChunkRef) REQUIRE c.chunk_id IS UNIQUE",
    # 按 KB 圈定概念节点
    "CREATE INDEX keyword_kb IF NOT EXISTS "
    "FOR (k:Keyword) ON (k.kb_id)",
    # 按文档删除/对账
    "CREATE INDEX chunk_ref_doc IF NOT EXISTS "
    "FOR (c:ChunkRef) ON (c.doc_id)",
    # 关系节点唯一键：subject \x1f relation \x1f object \x1f kb_id
    "CREATE CONSTRAINT relation_key IF NOT EXISTS "
    "FOR (r:Relation) REQUIRE r.key IS UNIQUE",
    # 按 KB + 关系类型做扩展过滤
    "CREATE INDEX relation_kb_type IF NOT EXISTS "
    "FOR (r:Relation) ON (r.kb_id, r.relation)",
)


async def init_schema(manager: Neo4jConnectionManager) -> None:
    """幂等创建约束与索引；失败抛 ``Neo4jSchemaError``。"""
    try:
        for statement in SCHEMA_STATEMENTS:
            await manager.run(statement)
    except Exception as exc:
        raise Neo4jSchemaError(
            "Neo4j schema 初始化失败",
            detail={"cause": str(exc)},
        ) from exc


async def check_schema(manager: Neo4jConnectionManager) -> dict[str, Any]:
    """返回约束/索引存在性摘要，供启动日志/健康检查使用。"""
    try:
        constraints = await manager.run("SHOW CONSTRAINTS")
        indexes = await manager.run("SHOW INDEXES")
    except Exception as exc:
        raise Neo4jSchemaError(
            "Neo4j schema 检查失败", detail={"cause": str(exc)}
        ) from exc
    return {
        "version": GRAPH_SCHEMA_VERSION,
        "constraints": [item.get("name") for item in constraints],
        "indexes": [item.get("name") for item in indexes],
    }
