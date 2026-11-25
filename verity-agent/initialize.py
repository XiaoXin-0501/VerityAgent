"""初始化并探测本地/测试环境的外部存储。

顺序：PostgreSQL → Redis → Neo4j → MinIO → Weaviate。

本模块既可作为根目录 ``init.py`` 的实现，也被应用启动链路调用
（``ensure_initialized``）：启动时幂等补齐缺失的表/bucket/collection，
不删除、不覆盖业务数据。

- PostgreSQL：建立/升级表结构；dev/test 写入默认账号与权限，prod 只建结构；
- Redis：PING，并做一次带 TTL 的 set/get/delete；
- Neo4j：建立约束/索引，检查 schema，查询节点数；
- MinIO：确保 bucket 存在，并上传/读取/删除一个临时探针对象；
- Weaviate：按 ``read/weaviate_schema_import.json`` 创建缺失 collection、
  补齐 property，然后执行 schema check。

用法（仓库根目录）：

```bash
.venv/Scripts/python init.py
.venv/Scripts/python init.py --only pg redis
.venv/Scripts/python init.py --weaviate-check-only
```
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

# 作为源码模块被根目录 init.py / 启动链路导入；独立运行时也补齐源码路径。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "verity-agent"))

from config import settings  # noqa: E402
from store.minio import (  # noqa: E402
    MinioStorage,
    close_minio,
    init_minio,
)
from store.neo4j import (  # noqa: E402
    check_schema as check_neo4j_schema,
    close_neo4j,
    init_neo4j,
    init_schema as init_neo4j_schema,
)
from store.pg import (  # noqa: E402
    close_pg,
    init_pg,
    init_schema as init_pg_schema,
)
from store.redis import close_redis, init_redis  # noqa: E402
from store.weaviate import (  # noqa: E402
    SchemaManager,
    close_weaviate,
    init_weaviate,
)

__all__ = ["ServiceResult", "main", "run"]

SERVICES = ("pg", "redis", "neo4j", "minio", "weaviate")


@dataclass(slots=True)
class ServiceResult:
    """单个服务的初始化/探测结果。"""

    name: str
    status: str
    detail: str
    duration_ms: int


def _timeout(value: float, default: float) -> float:
    """把命令行超时收紧到合理范围。"""
    return max(1.0, float(value or default))


def _safe_redis_target() -> str:
    """打印 Redis 目标时去掉 URL 中的密码。"""
    parts = urlsplit(settings.redis.url)
    host = parts.hostname or "localhost"
    port = parts.port or 6379
    db = parts.path.lstrip("/") or "0"
    return f"{host}:{port}/{db}"


async def _init_pg(probe: bool, *, seed_defaults: bool = True) -> str:
    """初始化 PostgreSQL schema 并做基础查询探活。"""
    manager = await init_pg()
    if probe:
        await manager.ping()
    await init_pg_schema(seed_defaults=seed_defaults)
    async with manager.connection() as conn:
        cursor = await conn.execute(
            "SELECT current_database() AS db, current_user AS usr"
        )
        row = await cursor.fetchone()
        cursor = await conn.execute(
            """
            SELECT count(*) AS tables
            FROM information_schema.tables
            WHERE table_schema = 'public'
            """
        )
        tables = await cursor.fetchone()
    return (
        f"db={row['db']} user={row['usr']} "
        f"tables={tables['tables'] if tables else 0}"
    )


async def _init_redis(probe: bool) -> str:
    """初始化 Redis，并用临时 key 验证读写。"""
    manager = await init_redis()
    await manager.ping()
    detail = f"target={_safe_redis_target()} namespace={settings.redis.namespace}"
    if not probe:
        return detail
    key = f"{settings.redis.namespace}:init:probe:{uuid4().hex}"
    try:
        await manager.set(key, "ok", ex=60)
        value = await manager.get(key)
        if value != "ok":
            raise RuntimeError(f"Redis 探针读取不一致: {value!r}")
    finally:
        await manager.delete(key)
    return f"{detail} set/get/delete=ok"


async def _init_neo4j(probe: bool) -> str:
    """初始化 Neo4j 约束/索引并检查 schema。"""
    manager = await init_neo4j()
    if probe:
        await manager.ping()
    await init_neo4j_schema(manager)
    summary = await check_neo4j_schema(manager)
    if probe:
        rows = await manager.run("MATCH (n) RETURN count(n) AS nodes")
        nodes = int(rows[0]["nodes"]) if rows else 0
    else:
        nodes = -1
    return (
        f"uri={settings.neo4j.uri} "
        f"constraints={len(summary.get('constraints', []))} "
        f"indexes={len(summary.get('indexes', []))} "
        f"nodes={nodes if nodes >= 0 else 'skipped'}"
    )


async def _init_minio(probe: bool) -> str:
    """初始化 MinIO bucket，并用临时对象验证上传/读取/删除。"""
    manager = await init_minio()
    await manager.ensure_buckets()
    storage = MinioStorage(manager)
    kb_bucket = storage.kb_bucket
    session_bucket = storage.session_bucket
    for bucket in (kb_bucket, session_bucket):
        if not await manager.bucket_exists(bucket):
            raise RuntimeError(f"MinIO bucket 不存在: {bucket}")
    detail = (
        f"endpoint={settings.minio.endpoint} "
        f"buckets={kb_bucket},{session_bucket}"
    )
    if not probe:
        return detail
    key = f"init-probe/{uuid4().hex}.txt"
    try:
        await storage.put_bytes(
            session_bucket,
            key,
            b"verity-init-probe",
            content_type="text/plain",
        )
        data = await storage.get_bytes(session_bucket, key)
        if data != b"verity-init-probe":
            raise RuntimeError("MinIO 探针读取内容不一致")
    finally:
        try:
            await storage.remove(session_bucket, key)
        except Exception as exc:  # noqa: BLE001 - 探针清理失败不覆盖主结果
            logging.warning("MinIO 探针清理失败: %s", exc)
    return f"{detail} put/get/delete=ok"


async def _init_weaviate(probe: bool, check_only: bool) -> str:
    """初始化 Weaviate schema，或仅做 schema check。"""
    manager = await init_weaviate()
    if probe:
        await manager.ping()
    schema_manager = SchemaManager(settings.weaviate)
    if check_only:
        result = await schema_manager.check(
            manager.client,
            expected_dimensions=settings.rag.indexer.embedding_dimensions,
        )
    else:
        result = await schema_manager.ensure(
            manager.client,
            expected_dimensions=settings.rag.indexer.embedding_dimensions,
        )
    if not result.ok:
        raise RuntimeError(
            "Weaviate schema 未通过: "
            f"missing={result.missing_collections or result.missing_properties} "
            f"type={result.type_mismatches} token={result.tokenization_mismatches} "
            f"dim={result.dimension_mismatches}"
        )
    warning_text = f" warnings={'; '.join(result.warnings)}" if result.warnings else ""
    return (
        f"url={settings.weaviate.url} "
        f"collections={','.join(schema_manager.expected_collections)} "
        f"mode={'check' if check_only else 'ensure'}{warning_text}"
    )


async def _run_service(
    name: str,
    *,
    probe: bool,
    weaviate_check_only: bool,
    timeout_s: float,
) -> ServiceResult:
    """执行单个服务，并把异常收敛成结果行。"""
    started = time.perf_counter()
    try:
        if name == "pg":
            detail = await asyncio.wait_for(
                _init_pg(probe, seed_defaults=not settings.app.is_prod),
                timeout=timeout_s,
            )
        elif name == "redis":
            detail = await asyncio.wait_for(_init_redis(probe), timeout=timeout_s)
        elif name == "neo4j":
            if not settings.neo4j.enabled:
                return ServiceResult(
                    "neo4j", "SKIP", "未启用", _elapsed_ms(started)
                )
            detail = await asyncio.wait_for(_init_neo4j(probe), timeout=timeout_s)
        elif name == "minio":
            if not settings.minio.enabled:
                return ServiceResult(
                    "minio", "SKIP", "未启用", _elapsed_ms(started)
                )
            detail = await asyncio.wait_for(_init_minio(probe), timeout=timeout_s)
        elif name == "weaviate":
            if not settings.weaviate.enabled:
                return ServiceResult(
                    "weaviate", "SKIP", "未启用", _elapsed_ms(started)
                )
            detail = await asyncio.wait_for(
                _init_weaviate(probe, weaviate_check_only),
                timeout=timeout_s,
            )
        else:  # pragma: no cover - argparse 已限制
            raise ValueError(f"未知服务: {name}")
        return ServiceResult(name, "OK", detail, _elapsed_ms(started))
    except Exception as exc:  # noqa: BLE001 - 环境自检需继续跑后续服务
        return ServiceResult(
            name,
            "FAIL",
            f"{type(exc).__name__}: {exc}",
            _elapsed_ms(started),
        )


def _elapsed_ms(started: float) -> int:
    """返回从 ``started`` 到现在的毫秒数。"""
    return int((time.perf_counter() - started) * 1000)


async def run(
    *,
    only: tuple[str, ...] = SERVICES,
    probe: bool = True,
    weaviate_check_only: bool = False,
    timeout_s: float = 90.0,
) -> list[ServiceResult]:
    """按顺序初始化并测试服务；单个失败不阻断后续服务。"""
    results: list[ServiceResult] = []
    try:
        for name in only:
            print(f"[init] {name} ...", flush=True)
            result = await _run_service(
                name,
                probe=probe,
                weaviate_check_only=weaviate_check_only,
                timeout_s=timeout_s,
            )
            results.append(result)
            print(
                f"[{result.status}] {result.name} "
                f"({result.duration_ms}ms) {result.detail}",
                flush=True,
            )
    finally:
        await _close_all()
    return results


async def _optional_service(
    name: str,
    required: bool,
    factory,
    timeout_s: float,
) -> None:
    """执行可选服务初始化；非必需服务失败只告警。"""
    try:
        await asyncio.wait_for(factory(), timeout=timeout_s)
    except Exception as exc:  # noqa: BLE001 - 可选服务不阻断启动
        if required:
            raise
        logging.warning("启动初始化 %s 失败，继续启动: %s", name, exc)


async def ensure_initialized(*, timeout_s: float = 90.0) -> None:
    """应用启动前的幂等初始化：补齐缺失结构，不删除/不覆盖业务数据。

    - PostgreSQL 必定执行：建表 + 迁移；prod 不写默认账号；
    - Neo4j / MinIO / Weaviate 仅在启用时执行；非 ``required_for_startup``
      的失败只告警，由应用自身的健康检查/降级继续处理；
    - Redis 无持久 schema，不在本函数中处理。
    """
    try:
        # PostgreSQL 是权威元数据源，初始化失败直接阻断启动。
        await asyncio.wait_for(
            _init_pg(False, seed_defaults=not settings.app.is_prod),
            timeout=timeout_s,
        )
        logging.info("启动初始化：PostgreSQL schema 已就绪")
        if settings.neo4j.enabled:
            await _optional_service(
                "neo4j",
                settings.neo4j.required_for_startup,
                lambda: _init_neo4j(False),
                timeout_s,
            )
        if settings.minio.enabled:
            await _optional_service(
                "minio",
                settings.minio.required_for_startup,
                lambda: _init_minio(False),
                timeout_s,
            )
        if settings.weaviate.enabled:
            await _optional_service(
                "weaviate",
                settings.weaviate.required_for_startup,
                lambda: _init_weaviate(False, False),
                timeout_s,
            )
    finally:
        await _close_all()


async def _close_all() -> None:
    """尽力关闭所有全局连接；关闭失败只记录。"""
    for name, closer in (
        ("weaviate", close_weaviate),
        ("minio", close_minio),
        ("neo4j", close_neo4j),
        ("redis", close_redis),
        ("pg", close_pg),
    ):
        try:
            await closer()
        except Exception as exc:  # noqa: BLE001 - 关闭阶段不覆盖主结果
            logging.warning("关闭 %s 失败: %s", name, exc)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="初始化并测试 PostgreSQL / Redis / Neo4j / MinIO / Weaviate",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=SERVICES,
        default=list(SERVICES),
        help="只运行指定服务",
    )
    parser.add_argument(
        "--no-probe",
        action="store_true",
        help="只初始化，不做额外读写探针（连接初始化仍会探活）",
    )
    parser.add_argument(
        "--weaviate-check-only",
        action="store_true",
        help="Weaviate 只校验 schema，不创建缺失 collection",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=90.0,
        help="单个服务的超时时间（秒）",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """同步入口：运行初始化并返回是否全部通过。"""
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # SDK 的逐请求/逐语句 INFO 太吵；失败仍会以 WARNING/ERROR 输出。
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    print(
        f"[init] env={settings.app.env.value} "
        f"root={ROOT} services={','.join(args.only)}",
        flush=True,
    )
    results = asyncio.run(
        run(
            only=tuple(args.only),
            probe=not args.no_probe,
            weaviate_check_only=args.weaviate_check_only,
            timeout_s=_timeout(args.timeout, 90.0),
        )
    )
    failed = [item for item in results if item.status == "FAIL"]
    skipped = [item for item in results if item.status == "SKIP"]
    print(
        f"[init] 完成: ok={len(results) - len(failed) - len(skipped)} "
        f"failed={len(failed)} skipped={len(skipped)}",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
