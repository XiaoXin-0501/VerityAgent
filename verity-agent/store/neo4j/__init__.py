"""Neo4j 图数据库模块。

- ``connection.py``：AsyncDriver 生命周期与事务封装；
- ``schema.py``：约束/索引初始化；
- ``storage.py``：建图、删除、1 跳扩展。
"""

from .connection import (
    Neo4jConnectionManager,
    close_neo4j,
    get_neo4j,
    init_neo4j,
    set_neo4j,
)
from .schema import (
    GRAPH_SCHEMA_VERSION,
    SCHEMA_STATEMENTS,
    check_schema,
    init_schema,
)
from .storage import GraphStore, relation_key

__all__ = [
    "Neo4jConnectionManager",
    "get_neo4j",
    "set_neo4j",
    "init_neo4j",
    "close_neo4j",
    "GRAPH_SCHEMA_VERSION",
    "SCHEMA_STATEMENTS",
    "init_schema",
    "check_schema",
    "GraphStore",
    "relation_key",
]
