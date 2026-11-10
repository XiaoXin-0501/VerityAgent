"""Neo4j 图数据库异常。"""

from __future__ import annotations

from .base import VerityError

__all__ = [
    "Neo4jError",
    "Neo4jConnectionError",
    "Neo4jQueryError",
    "Neo4jSchemaError",
]


class Neo4jError(VerityError):
    """Neo4j 相关异常基类。"""

    code = "NEO4J_ERROR"
    http_status = 500
    default_message = "图数据库操作失败"


class Neo4jConnectionError(Neo4jError):
    """连接不上 Neo4j / 连接中断 / 超时。"""

    code = "NEO4J_CONNECTION_ERROR"
    http_status = 503
    default_message = "图数据库连接失败"
    retryable = True


class Neo4jQueryError(Neo4jError):
    """Cypher 执行失败。"""

    code = "NEO4J_QUERY_ERROR"
    http_status = 500
    default_message = "图数据库查询失败"


class Neo4jSchemaError(Neo4jError):
    """约束/索引创建失败或与预期不一致。"""

    code = "NEO4J_SCHEMA_ERROR"
    http_status = 500
    default_message = "图数据库 schema 初始化失败"
    user_visible = False
