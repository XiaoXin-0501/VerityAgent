"""Weaviate 向量库模块。

- ``connection.py``：client 生命周期；
- ``schema.py``：import JSON 校验与 dev/test 创建（D-97）；
- ``storage.py``：collection 级通用操作。
"""

from .connection import (
    WeaviateConnectionManager,
    close_weaviate,
    get_weaviate,
    init_weaviate,
    set_weaviate,
)
from .schema import SchemaCheckResult, SchemaManager, load_schema_import
from .storage import WeaviateStore

__all__ = [
    "WeaviateConnectionManager",
    "get_weaviate",
    "set_weaviate",
    "init_weaviate",
    "close_weaviate",
    "SchemaCheckResult",
    "SchemaManager",
    "load_schema_import",
    "WeaviateStore",
]
