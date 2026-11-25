"""PostgreSQL 建表语句（与《PostgreSQL 数据库结构设计》§2 的 11 张表一致）。

设计要点：

- 主键统一 UUID，默认 ``gen_random_uuid()``（PostgreSQL 13+ 内置）；
- ``TIMESTAMPTZ`` 统一由数据库 ``now()`` 生成，应用只传 UTC；
- 状态列与 ``models/enums.py`` 保持一致；
- JSONB 结构由应用层模型校验，数据库只保证是合法 JSON；
- 删除采用「软删 + 跨存储状态机」：``deleted_at`` / ``delete_attempts`` /
  ``delete_error`` / ``delete_next_retry_at``；
- 入库任务复用 ``documents`` 的 ``index_*`` 列（D-77），不新增任务表；
- ``user_memory.fact_count`` 的 150 硬上限只是最后防线。

``init_schema`` 幂等；SQL 来自 ``read/postgres_*.sql``，由根目录 ``init.py``
或启动期 ``ensure_initialized`` 执行。生产环境不写入默认账号，表结构变更
只追加 ``ALTER TABLE``，不删除已有列/数据。
"""

from __future__ import annotations

from argon2 import PasswordHasher
from psycopg import AsyncConnection

from config.base import PROJECT_ROOT
from exception import ConfigurationError

from .connection import PGConnectionManager, Row, get_pg
from .models.enums import UserRole

# 仅用于开发/首次启动；生产环境必须改密或删除。
# 角色以这里为准：xiaoxin 是管理员，xiaobai 是普通用户。
DEFAULT_USERS: tuple[tuple[str, str, UserRole], ...] = (
    ("xiaoxin", "123456", UserRole.ADMIN),
    ("xiaobai", "123456", UserRole.USER),
)

# 默认权限码目录；开发库首启/升级时写入 permissions。
# 资源级权限码（kb:read:<id> / doc:read:<id>）由业务创建资源时按需补；
# kb:read:* / doc:read:* 是部署级通配，避免每个资源都逐条授权。
DEFAULT_PERMISSIONS: tuple[tuple[str, str], ...] = (
    ("kb:read", "读取使用默认权限码的知识库"),
    ("kb:read:*", "读取所有知识库（部署级通配）"),
    ("kb:write:*", "创建、更新、删除所有知识库"),
    ("doc:read:*", "读取所有文档"),
    ("doc:write:*", "上传、更新、删除所有文档"),
    ("tool:kb:search", "使用知识库检索工具 search_kb"),
    ("tool:kb:conflict", "使用片段冲突处理工具 resolve_chunk_conflict"),
    ("tool:memory:recall", "使用轮次回溯工具 recall_round"),
    ("admin:users", "管理用户与权限"),
    ("admin:kb", "管理知识库与索引"),
)

# 普通默认用户只授予只读能力；管理员 xiaoxin 在 seed 时授予 permissions
# 表中的全部权限，确保新增默认权限后无需手工补授权。
DEFAULT_USER_PERMISSIONS: dict[str, tuple[str, ...]] = {
    "xiaobai": (
        "kb:read",
        "doc:read:*",
        "tool:kb:search",
        "tool:memory:recall",
    ),
}

_hasher = PasswordHasher()

# 建表/迁移 SQL 是数据文件，统一放在仓库 ``read/`` 下；应用启动不自动执行，
# 由根目录 init.py / 启动期 ensure_initialized（或 tests/reset_state.py）
# 显式调用 ``init_schema``。
_READ_DIR = PROJECT_ROOT / "read"


def _load_sql(name: str) -> str:
    """读取 ``read/`` 下的 SQL 文件；缺失时说明期望路径。"""
    path = _READ_DIR / name
    if not path.is_file():
        raise ConfigurationError(
            "PostgreSQL schema SQL 文件不存在",
            details={"path": str(path)},
        )
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(
            "PostgreSQL schema SQL 文件读取失败",
            details={"path": str(path), "cause": str(exc)},
        ) from exc


def migrations_sql() -> str:
    """返回旧库升级 SQL（必须在建表 SQL 之前执行）。"""
    return _load_sql("postgres_migrations.sql")


def schema_sql() -> str:
    """返回 11 张业务表的建表/索引 SQL。"""
    return _load_sql("postgres_schema.sql")



async def init_schema(
    db: PGConnectionManager | None = None,
    *,
    seed_defaults: bool = True,
) -> None:
    """幂等地升级旧列、创建全部表/索引；可选写入默认账号与权限。

    ``seed_defaults=False`` 只建结构不写种子账号，供生产启动自检使用；
    根目录 ``init.py`` / 启动期 ``ensure_initialized`` 与
    ``tests/reset_state.py`` 按环境选择是否写入种子。
    """
    manager = db or get_pg()
    async with manager.transaction() as conn:
        await conn.execute(migrations_sql())
        await conn.execute(schema_sql())
        if seed_defaults:
            await seed_default_users(conn)
            await seed_default_permissions(conn)


async def seed_default_users(conn: AsyncConnection[Row]) -> None:
    """写入默认账号；已存在时保留密码，但同步默认角色。

    默认账号只用于开发/首启；生产环境应改密或删除。角色同步让已有库中的
    ``xiaoxin`` 在升级后也获得管理员权限。
    """
    for username, password, role in DEFAULT_USERS:
        await conn.execute(
            """
            INSERT INTO users (username, password_hash, role)
            VALUES (%s, %s, %s)
            ON CONFLICT (username) DO UPDATE
            SET role = EXCLUDED.role
            """,
            (username, _hasher.hash(password), role.value),
        )


async def seed_default_permissions(conn: AsyncConnection[Row]) -> None:
    """写入默认权限码，并同步默认用户的授权。

    - ``permissions`` 按 ``permission_code`` 幂等 upsert；
    - ``xiaoxin`` 获得当前 permissions 表中的全部权限（管理员种子）；
    - ``xiaobai`` 只获得只读基础权限；
    - 资源级权限码仍由业务创建资源时按需写入，不在种子中伪造 id。
    """
    for permission_code, description in DEFAULT_PERMISSIONS:
        await conn.execute(
            """
            INSERT INTO permissions (permission_code, description)
            VALUES (%s, %s)
            ON CONFLICT (permission_code) DO UPDATE
            SET description = EXCLUDED.description
            """,
            (permission_code, description),
        )
    # 管理员：授权当前权限表中的全部权限，包括后续管理面新增的权限。
    await conn.execute(
        """
        INSERT INTO user_permissions (user_id, permission_id, permission_code)
        SELECT u.id, p.id, p.permission_code
        FROM users u
        CROSS JOIN permissions p
        WHERE u.username = %s
        ON CONFLICT (user_id, permission_id) DO NOTHING
        """,
        ("xiaoxin",),
    )
    for username, permission_codes in DEFAULT_USER_PERMISSIONS.items():
        await conn.execute(
            """
            INSERT INTO user_permissions (user_id, permission_id, permission_code)
            SELECT u.id, p.id, p.permission_code
            FROM users u
            CROSS JOIN permissions p
            WHERE u.username = %s
              AND p.permission_code = ANY(%s)
            ON CONFLICT (user_id, permission_id) DO NOTHING
            """,
            (username, list(permission_codes)),
        )


def schema_statements() -> list[str]:
    """按分号切分语句列表，方便单条执行或排查 DDL。"""
    return [stmt.strip() for stmt in schema_sql().split(";") if stmt.strip()]
