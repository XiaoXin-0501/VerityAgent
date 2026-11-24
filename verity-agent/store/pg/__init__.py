"""PostgreSQL 存储模块。

结构：

- ``connection.py``  ：异步连接池（psycopg_pool）与事务上下文；
- ``schema.py``      ：11 张表的幂等 DDL；
- ``models/``        ：行模型（Pydantic）+ 消息/摘要/画像 JSONB 协议；
- ``repositories/``  ：按表划分的仓库，支持跨仓库同事务组合。

启动示例::

    from store.pg import init_pg, close_pg, init_schema

    await init_pg()
    await init_schema()
    ...
    await close_pg()
"""

from .connection import (
    PGConnectionManager,
    close_pg,
    get_pg,
    init_pg,
    set_pg,
)
from .repositories import (
    BaseRepository,
    ConversationMessageRepository,
    ConversationRepository,
    ConversationSummaryRepository,
    DocumentRepository,
    KnowledgeBaseRepository,
    PermissionRepository,
    SessionFileRepository,
    UserMemoryRepository,
    UserMessageRepository,
    UserRepository,
)
from .schema import (
    DEFAULT_PERMISSIONS,
    DEFAULT_USER_PERMISSIONS,
    DEFAULT_USERS,
    MIGRATIONS_SQL,
    SCHEMA_SQL,
    init_schema,
    schema_statements,
    seed_default_permissions,
    seed_default_users,
)

__all__ = [
    # 连接
    "PGConnectionManager",
    "get_pg",
    "set_pg",
    "init_pg",
    "close_pg",
    # 建表
    "SCHEMA_SQL",
    "MIGRATIONS_SQL",
    "DEFAULT_USERS",
    "DEFAULT_PERMISSIONS",
    "DEFAULT_USER_PERMISSIONS",
    "init_schema",
    "seed_default_users",
    "seed_default_permissions",
    "schema_statements",
    # 仓库
    "BaseRepository",
    "UserRepository",
    "PermissionRepository",
    "KnowledgeBaseRepository",
    "DocumentRepository",
    "ConversationRepository",
    "ConversationMessageRepository",
    "ConversationSummaryRepository",
    "UserMemoryRepository",
    "UserMessageRepository",
    "SessionFileRepository",
]
