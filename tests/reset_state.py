"""把本地开发/测试环境重置到“刚初始化”的状态。

清空范围：

- PostgreSQL：用户、权限、知识库、文档、会话、消息、摘要、长期记忆、
  用户消息队列、会话文件；清空后重新写入默认账号（xiaoxin / xiaobai）
  与默认权限（xiaoxin 全量、xiaobai 只读基础）。
- Weaviate：删除并重建 DocChunk / UserFact collection。
- Neo4j：删除所有节点与关系。
- MinIO：清空知识库 bucket 与会话 bucket 中的全部对象。
- Redis：清空当前 namespace 下的检索/工具/会话缓存与 run 状态。

运行方式：

```bash
# 先停止后端，避免运行中的 worker 继续写入
.venv/Scripts/python tests/reset_state.py
```

安全约束：

- ``APP__ENV=prod`` 时直接拒绝执行；
- 只清空业务数据，不删除表结构、bucket 或数据库。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# 允许直接 ``python tests/reset_state.py`` 运行。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "verity-agent"))

from config import settings
from config.app import Environment
from store.minio import MinioStorage, close_minio, get_minio, init_minio
from store.neo4j import close_neo4j, get_neo4j, init_neo4j
from store.pg import close_pg, get_pg, init_pg, init_schema
from store.pg.schema import seed_default_permissions, seed_default_users
from store.redis import close_redis, get_redis, init_redis
from store.weaviate import (
    SchemaManager,
    close_weaviate,
    get_weaviate,
    init_weaviate,
)

# 业务表；按依赖关系列出，TRUNCATE ... CASCADE 会处理外键。
APP_TABLES: tuple[str, ...] = (
    "user_permissions",
    "permissions",
    "session_files",
    "user_messages",
    "user_memory",
    "conversation_summaries",
    "conversation_messages",
    "conversations",
    "documents",
    "knowledge_bases",
    "users",
)


async def _reset_postgres() -> None:
    """重建/确认表结构，清空业务表并恢复默认账号与权限。"""
    await init_pg()
    await init_schema()

    async with get_pg().transaction() as conn:
        cursor = await conn.execute(
            """
            SELECT tablename
            FROM pg_tables
            WHERE schemaname = 'public'
            """
        )
        existing = {str(row["tablename"]) for row in await cursor.fetchall()}
        targets = [name for name in APP_TABLES if name in existing]
        if targets:
            await conn.execute(
                f"TRUNCATE TABLE {', '.join(targets)} "
                "RESTART IDENTITY CASCADE"
            )
        await seed_default_users(conn)
        await seed_default_permissions(conn)

    print(f"[postgres] 已清空: {', '.join(targets) or '（无匹配业务表）'}")
    print("[postgres] 已恢复默认账号与权限: xiaoxin(admin, 全部权限), xiaobai(user, 只读基础权限)")


async def _reset_weaviate() -> None:
    """删除并重建 DocChunk / UserFact collection。"""
    if not settings.weaviate.enabled:
        print("[weaviate] 未启用，跳过")
        return
    # 先加载 schema 再删除 collection：schema 路径/内容错误应在破坏性操作
    # 之前失败，避免“collection 已删、schema 没加载、环境半重置”。
    manager = SchemaManager(settings.weaviate)
    await init_weaviate()
    client = get_weaviate().client
    existing = await asyncio.to_thread(client.collections.list_all, False)
    names = (
        settings.weaviate.doc_chunk_collection,
        settings.weaviate.user_fact_collection,
    )
    for name in names:
        if name in existing:
            await asyncio.to_thread(client.collections.delete, name)
            print(f"[weaviate] 已删除 collection: {name}")
    await manager.ensure(
        client,
        expected_dimensions=settings.rag.indexer.embedding_dimensions,
    )
    print(f"[weaviate] 已重建 collection: {', '.join(names)}")


async def _reset_neo4j() -> None:
    """删除 Neo4j 中全部节点和关系。"""
    if not settings.neo4j.enabled:
        print("[neo4j] 未启用，跳过")
        return
    await init_neo4j()

    async def clear(tx) -> None:
        """在写事务中执行全图删除。"""
        result = await tx.run("MATCH (n) DETACH DELETE n")
        await result.consume()

    await get_neo4j().execute_write(clear)
    print("[neo4j] 已删除全部节点与关系")


async def _reset_minio() -> None:
    """清空知识库 bucket 与会话 bucket。"""
    if not settings.minio.enabled:
        print("[minio] 未启用，跳过")
        return
    await init_minio()
    storage = MinioStorage(get_minio())
    kb_removed = await storage.remove_prefix(storage.kb_bucket, "")
    session_removed = await storage.remove_prefix(storage.session_bucket, "")
    print(
        f"[minio] 已清空 kb={kb_removed} objects, "
        f"session={session_removed} objects"
    )


async def _reset_redis() -> None:
    """清空当前 namespace 下的 Redis key（检索/工具/会话缓存与 run 状态）。"""
    await init_redis()
    redis = get_redis()
    prefix = f"{settings.redis.namespace}:"
    cursor: str | int = 0
    removed = 0
    while True:
        cursor, keys = await redis.execute_command(
            "SCAN", cursor, "MATCH", f"{prefix}*", "COUNT", 500
        )
        if keys:
            removed += int(await redis.delete(*keys))
        if str(cursor) == "0":
            break
    print(f"[redis] 已清空 namespace={settings.redis.namespace}: {removed} keys")


async def _run() -> None:
    """按外部存储 -> Redis -> PostgreSQL 的顺序重置，最后统一释放连接。"""
    try:
        await _reset_weaviate()
        await _reset_neo4j()
        await _reset_minio()
        await _reset_redis()
        await _reset_postgres()
        print("[reset] 完成：环境已恢复到刚初始化状态")
    finally:
        await close_neo4j()
        await close_weaviate()
        await close_minio()
        await close_redis()
        await close_pg()


def main() -> None:
    """同步入口：生产环境拒绝执行，其余环境直接重置。"""
    if settings.app.env is Environment.PROD:
        print("拒绝执行：APP__ENV=prod 时不允许使用测试重置脚本。")
        raise SystemExit(2)

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(_run())


if __name__ == "__main__":
    main()
