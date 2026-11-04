"""PostgreSQL 建表语句（与《PostgreSQL 数据库结构设计》§2 的 11 张表一致）。

设计要点：
- 主键统一 UUID，默认 ``gen_random_uuid()``（PostgreSQL 13+ 内置，无需 pgcrypto）；
- ``TIMESTAMPTZ`` 统一由数据库 ``now()`` 生成，避免应用节点时区不一致；
- 状态列带 CHECK 约束，和 ``models/enums.py`` 保持一致；
- JSONB 结构（消息 / 摘要 / 画像）由应用层模型校验，数据库只保证是合法 JSON；
- ``user_memory.fact_count`` 用 CHECK 落死 150 条硬上限；
- 建表后幂等插入两个默认账号（``xiaoxin`` / ``xiaobai``，密码均为 ``123456``），
  已存在则不覆盖密码。

``init_schema`` 幂等，可重复执行；单企业单 Agent 场景不做迁移框架，
表结构变更时补 ``ALTER TABLE`` 或重建。
"""

from __future__ import annotations

from argon2 import PasswordHasher
from psycopg import AsyncConnection

from .connection import PGConnectionManager, get_pg

# 初始化库时写入的默认账号：仅用于开发/首次启动，生产环境应及时改密或删除
DEFAULT_USERS: tuple[tuple[str, str], ...] = (
    ("xiaoxin", "123456"),
    ("xiaobai", "123456"),
)

_hasher = PasswordHasher()

SCHEMA_SQL = """
-- ==========================================================
-- 1. users 用户表
-- ==========================================================
CREATE TABLE IF NOT EXISTS users (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username      VARCHAR(100) NOT NULL UNIQUE,
    email         VARCHAR(255) UNIQUE,
    phone         VARCHAR(30),
    password_hash TEXT,
    role          VARCHAR(20) NOT NULL DEFAULT 'user'
                  CHECK (role IN ('admin', 'user')),
    status        VARCHAR(20) NOT NULL DEFAULT 'active'
                  CHECK (status IN ('active', 'disabled', 'locked')),
    last_login_at TIMESTAMPTZ,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ==========================================================
-- 2. permissions 权限码定义
-- ==========================================================
CREATE TABLE IF NOT EXISTS permissions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    permission_code VARCHAR(100) NOT NULL UNIQUE,
    description     TEXT NOT NULL DEFAULT '',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ==========================================================
-- 3. user_permissions 用户权限关联
--    permission_code 冗余一份，鉴权单表查询，不 join permissions
-- ==========================================================
CREATE TABLE IF NOT EXISTS user_permissions (
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    permission_id   UUID NOT NULL REFERENCES permissions(id) ON DELETE CASCADE,
    permission_code VARCHAR(100) NOT NULL,
    PRIMARY KEY (user_id, permission_id)
);
CREATE INDEX IF NOT EXISTS idx_user_permissions_permission_id
    ON user_permissions (permission_id);
CREATE INDEX IF NOT EXISTS idx_user_permissions_user_code
    ON user_permissions (user_id, permission_code);

-- ==========================================================
-- 4. knowledge_bases 知识库
-- ==========================================================
CREATE TABLE IF NOT EXISTS knowledge_bases (
    id                       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name                     VARCHAR(200) NOT NULL,
    description              TEXT,
    required_permission_code VARCHAR(100) NOT NULL,
    status                   VARCHAR(20) NOT NULL DEFAULT 'active'
                             CHECK (status IN ('active', 'disabled', 'deleting')),
    created_at               TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at               TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at               TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_knowledge_bases_active
    ON knowledge_bases (status) WHERE deleted_at IS NULL;

-- ==========================================================
-- 5. documents 知识库文档
--    completed = 解析成功 + Weaviate 写入成功 + Neo4j 引用建完
-- ==========================================================
CREATE TABLE IF NOT EXISTS documents (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    kb_id              UUID NOT NULL REFERENCES knowledge_bases(id) ON DELETE RESTRICT,
    original_filename  VARCHAR(500) NOT NULL,
    extension          VARCHAR(20) NOT NULL,
    size_bytes         BIGINT NOT NULL CHECK (size_bytes >= 0),
    sha256             CHAR(64),
    source_object_key  TEXT NOT NULL,
    parse_manifest_key TEXT,
    uploaded_by        UUID NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    status             VARCHAR(20) NOT NULL DEFAULT 'pending'
                       CHECK (status IN ('pending', 'processing', 'completed',
                                         'failed', 'deleting', 'deleted')),
    token_count        BIGINT,
    chunk_count        INTEGER,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    parsed_at          TIMESTAMPTZ,
    deleted_at         TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_documents_kb_status
    ON documents (kb_id, status) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_documents_sha256
    ON documents (sha256) WHERE sha256 IS NOT NULL;

-- ==========================================================
-- 6. conversations 会话
--    status 管会话生命周期；current_run_status 管单次运行状态
-- ==========================================================
CREATE TABLE IF NOT EXISTS conversations (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id            UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title              VARCHAR(300) NOT NULL DEFAULT '',
    status             VARCHAR(20) NOT NULL DEFAULT 'active'
                       CHECK (status IN ('active', 'archived', 'deleted')),
    current_run_status VARCHAR(20) NOT NULL DEFAULT 'idle'
                       CHECK (current_run_status IN ('idle', 'running', 'waiting_tool',
                                                     'paused', 'failed', 'completed')),
    turn_count         BIGINT NOT NULL DEFAULT 0 CHECK (turn_count >= 0),
    total_input_tokens BIGINT NOT NULL DEFAULT 0 CHECK (total_input_tokens >= 0),
    total_output_tokens BIGINT NOT NULL DEFAULT 0 CHECK (total_output_tokens >= 0),
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_message_at    TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_conversations_user_last_message
    ON conversations (user_id, last_message_at DESC NULLS LAST);

-- ==========================================================
-- 7. conversation_messages 原始消息（ground truth，append-only）
-- ==========================================================
CREATE TABLE IF NOT EXISTS conversation_messages (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    seq_no          BIGINT NOT NULL CHECK (seq_no > 0),
    role            VARCHAR(20) NOT NULL
                    CHECK (role IN ('user', 'assistant', 'tool', 'system')),
    content         JSONB NOT NULL,
    status          VARCHAR(20) NOT NULL DEFAULT 'completed'
                    CHECK (status IN ('streaming', 'completed', 'partial', 'failed')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (conversation_id, seq_no)
);

-- ==========================================================
-- 8. conversation_summaries 会话摘要（1:1，不设代理主键）
-- ==========================================================
CREATE TABLE IF NOT EXISTS conversation_summaries (
    conversation_id     UUID PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
    summary_json        JSONB NOT NULL,
    covered_through_seq BIGINT NOT NULL DEFAULT 0 CHECK (covered_through_seq >= 0),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ==========================================================
-- 9. user_memory 长期记忆画像（一人一份）
--    revision 是乐观锁版本号；fact_count 硬上限 150
-- ==========================================================
CREATE TABLE IF NOT EXISTS user_memory (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id    UUID NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
    value      JSONB NOT NULL,
    revision   BIGINT NOT NULL DEFAULT 0 CHECK (revision >= 0),
    fact_count INTEGER NOT NULL DEFAULT 0
               CHECK (fact_count >= 0 AND fact_count <= 150),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ==========================================================
-- 10. user_messages 用户消息暂存（长期记忆 worker 扫描）
--     与 conversation_messages 同一事务双写；一条消息只暂存一次
-- ==========================================================
CREATE TABLE IF NOT EXISTS user_messages (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    message_id      UUID NOT NULL UNIQUE REFERENCES conversation_messages(id) ON DELETE CASCADE,
    content         JSONB NOT NULL,
    status          VARCHAR(20) NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'processing', 'failed')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_user_messages_status_created
    ON user_messages (status, created_at);
CREATE INDEX IF NOT EXISTS idx_user_messages_user_status
    ON user_messages (user_id, status);

-- ==========================================================
-- 11. session_files 会话内上传文件
--     只在所属会话范围检索；过期由 MinIO bucket TTL 回收
-- ==========================================================
CREATE TABLE IF NOT EXISTS session_files (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id    UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    message_id         UUID REFERENCES conversation_messages(id) ON DELETE SET NULL,
    uploaded_by        UUID NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    original_filename  VARCHAR(500) NOT NULL,
    extension          VARCHAR(20) NOT NULL,
    size_bytes         BIGINT NOT NULL CHECK (size_bytes >= 0),
    sha256             CHAR(64),
    source_object_key  TEXT NOT NULL,
    parse_manifest_key TEXT,
    status             VARCHAR(20) NOT NULL DEFAULT 'uploading'
                       CHECK (status IN ('uploading', 'processing', 'available',
                                         'failed', 'deleting', 'deleted')),
    token_count        BIGINT,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_session_files_conversation_status
    ON session_files (conversation_id, status);
"""


async def init_schema(db: PGConnectionManager | None = None) -> None:
    """幂等地创建全部表、索引与约束，并写入默认账号。"""
    manager = db or get_pg()
    async with manager.transaction() as conn:
        await conn.execute(SCHEMA_SQL)
        await seed_default_users(conn)


async def seed_default_users(conn: AsyncConnection) -> None:
    """插入默认账号；用户名已存在时保持原密码不动。

    密码用 argon2id 现场哈希，每次调用盐值不同；``ON CONFLICT DO NOTHING``
    保证重复执行 ``init_schema`` 不会把用户改过的密码重置回默认值。
    """
    for username, password in DEFAULT_USERS:
        await conn.execute(
            """
            INSERT INTO users (username, password_hash)
            VALUES (%s, %s)
            ON CONFLICT (username) DO NOTHING
            """,
            (username, _hasher.hash(password)),
        )


# 按分号切分的语句列表，方便需要单条执行（或排查某条 DDL）时使用
def schema_statements() -> list[str]:
    return [stmt.strip() for stmt in SCHEMA_SQL.split(";") if stmt.strip()]
