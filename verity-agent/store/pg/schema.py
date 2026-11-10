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

``init_schema`` 幂等；首版不引入迁移框架，表结构变更时补 ``ALTER TABLE``
或重建开发库。生产环境应使用独立迁移流程。
"""

from __future__ import annotations

from argon2 import PasswordHasher
from psycopg import AsyncConnection

from .connection import PGConnectionManager, get_pg

# 仅用于开发/首次启动；生产环境必须改密或删除
DEFAULT_USERS: tuple[tuple[str, str], ...] = (
    ("xiaoxin", "123456"),
    ("xiaobai", "123456"),
)

_hasher = PasswordHasher()

# 旧库升级用的加列/约束迁移。必须在 SCHEMA_SQL 之前执行：
# 旧表已存在时 CREATE TABLE IF NOT EXISTS 不会补列，而 SCHEMA_SQL 里的索引
# 可能引用新列。全部使用 IF EXISTS / IF NOT EXISTS，可重复执行。
MIGRATIONS_SQL = """
-- users：跨存储删除
ALTER TABLE IF EXISTS users ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ;
ALTER TABLE IF EXISTS users ADD COLUMN IF NOT EXISTS delete_attempts INTEGER NOT NULL DEFAULT 0;
ALTER TABLE IF EXISTS users ADD COLUMN IF NOT EXISTS delete_error TEXT;
ALTER TABLE IF EXISTS users ADD COLUMN IF NOT EXISTS delete_next_retry_at TIMESTAMPTZ;
ALTER TABLE IF EXISTS users DROP CONSTRAINT IF EXISTS users_status_check;
ALTER TABLE IF EXISTS users ADD CONSTRAINT users_status_check
    CHECK (status IN ('active', 'disabled', 'locked', 'deleting', 'deleted'));

-- knowledge_bases：索引版本 / embedding 版本 / 删除
ALTER TABLE IF EXISTS knowledge_bases ADD COLUMN IF NOT EXISTS index_version BIGINT NOT NULL DEFAULT 0;
ALTER TABLE IF EXISTS knowledge_bases ADD COLUMN IF NOT EXISTS embedding_model_id VARCHAR(100) NOT NULL DEFAULT 'unset';
ALTER TABLE IF EXISTS knowledge_bases ADD COLUMN IF NOT EXISTS embedding_dimensions INTEGER NOT NULL DEFAULT 1024;
ALTER TABLE IF EXISTS knowledge_bases ADD COLUMN IF NOT EXISTS embedding_version VARCHAR(50) NOT NULL DEFAULT 'v1';
ALTER TABLE IF EXISTS knowledge_bases ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ;
ALTER TABLE IF EXISTS knowledge_bases ADD COLUMN IF NOT EXISTS delete_attempts INTEGER NOT NULL DEFAULT 0;
ALTER TABLE IF EXISTS knowledge_bases ADD COLUMN IF NOT EXISTS delete_error TEXT;
ALTER TABLE IF EXISTS knowledge_bases ADD COLUMN IF NOT EXISTS delete_next_retry_at TIMESTAMPTZ;

-- documents：描述 / 入库任务 / 删除
ALTER TABLE IF EXISTS documents ADD COLUMN IF NOT EXISTS description TEXT;
ALTER TABLE IF EXISTS documents ADD COLUMN IF NOT EXISTS index_job_id UUID;
ALTER TABLE IF EXISTS documents ADD COLUMN IF NOT EXISTS index_stage VARCHAR(40);
ALTER TABLE IF EXISTS documents ADD COLUMN IF NOT EXISTS index_progress REAL;
ALTER TABLE IF EXISTS documents ADD COLUMN IF NOT EXISTS index_attempts INTEGER NOT NULL DEFAULT 0;
ALTER TABLE IF EXISTS documents ADD COLUMN IF NOT EXISTS index_error TEXT;
ALTER TABLE IF EXISTS documents ADD COLUMN IF NOT EXISTS index_next_retry_at TIMESTAMPTZ;
ALTER TABLE IF EXISTS documents ADD COLUMN IF NOT EXISTS index_locked_until TIMESTAMPTZ;
ALTER TABLE IF EXISTS documents ADD COLUMN IF NOT EXISTS index_locked_by TEXT;
ALTER TABLE IF EXISTS documents ADD COLUMN IF NOT EXISTS index_version BIGINT;
ALTER TABLE IF EXISTS documents ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ;
ALTER TABLE IF EXISTS documents ADD COLUMN IF NOT EXISTS delete_attempts INTEGER NOT NULL DEFAULT 0;
ALTER TABLE IF EXISTS documents ADD COLUMN IF NOT EXISTS delete_error TEXT;
ALTER TABLE IF EXISTS documents ADD COLUMN IF NOT EXISTS delete_next_retry_at TIMESTAMPTZ;

-- conversations：run 租约 / 删除
ALTER TABLE IF EXISTS conversations ADD COLUMN IF NOT EXISTS current_run_id UUID;
ALTER TABLE IF EXISTS conversations ADD COLUMN IF NOT EXISTS run_started_at TIMESTAMPTZ;
ALTER TABLE IF EXISTS conversations ADD COLUMN IF NOT EXISTS lock_expires_at TIMESTAMPTZ;
ALTER TABLE IF EXISTS conversations ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ;
ALTER TABLE IF EXISTS conversations ADD COLUMN IF NOT EXISTS delete_attempts INTEGER NOT NULL DEFAULT 0;
ALTER TABLE IF EXISTS conversations ADD COLUMN IF NOT EXISTS delete_error TEXT;
ALTER TABLE IF EXISTS conversations ADD COLUMN IF NOT EXISTS delete_next_retry_at TIMESTAMPTZ;
ALTER TABLE IF EXISTS conversations DROP CONSTRAINT IF EXISTS conversations_status_check;
ALTER TABLE IF EXISTS conversations ADD CONSTRAINT conversations_status_check
    CHECK (status IN ('active', 'archived', 'deleting', 'deleted'));
ALTER TABLE IF EXISTS conversations DROP CONSTRAINT IF EXISTS conversations_current_run_status_check;
ALTER TABLE IF EXISTS conversations ADD CONSTRAINT conversations_current_run_status_check
    CHECK (current_run_status IN ('idle', 'running', 'waiting_tool', 'paused',
                                  'failed', 'completed', 'cancelled'));

-- conversation_messages：幂等键 / finalize 时间
ALTER TABLE IF EXISTS conversation_messages ADD COLUMN IF NOT EXISTS client_message_id UUID;
ALTER TABLE IF EXISTS conversation_messages ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT now();

-- conversation_summaries：压缩次数
ALTER TABLE IF EXISTS conversation_summaries ADD COLUMN IF NOT EXISTS compression_count BIGINT NOT NULL DEFAULT 0;

-- user_memory：事实集版本
ALTER TABLE IF EXISTS user_memory ADD COLUMN IF NOT EXISTS facts_revision BIGINT NOT NULL DEFAULT 0;

-- user_messages：ready_at / 租约 / 重试
ALTER TABLE IF EXISTS user_messages ADD COLUMN IF NOT EXISTS ready_at TIMESTAMPTZ;
ALTER TABLE IF EXISTS user_messages ADD COLUMN IF NOT EXISTS attempt_count INTEGER NOT NULL DEFAULT 0;
ALTER TABLE IF EXISTS user_messages ADD COLUMN IF NOT EXISTS last_error TEXT;
ALTER TABLE IF EXISTS user_messages ADD COLUMN IF NOT EXISTS next_retry_at TIMESTAMPTZ;
ALTER TABLE IF EXISTS user_messages ADD COLUMN IF NOT EXISTS locked_until TIMESTAMPTZ;
ALTER TABLE IF EXISTS user_messages ADD COLUMN IF NOT EXISTS locked_by TEXT;
ALTER TABLE IF EXISTS user_messages ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT now();
ALTER TABLE IF EXISTS user_messages DROP CONSTRAINT IF EXISTS user_messages_status_check;
ALTER TABLE IF EXISTS user_messages ADD CONSTRAINT user_messages_status_check
    CHECK (status IN ('pending', 'processing', 'failed', 'dead'));

-- session_files：更新时间 / 删除
ALTER TABLE IF EXISTS session_files ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT now();
ALTER TABLE IF EXISTS session_files ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ;
ALTER TABLE IF EXISTS session_files ADD COLUMN IF NOT EXISTS delete_attempts INTEGER NOT NULL DEFAULT 0;
ALTER TABLE IF EXISTS session_files ADD COLUMN IF NOT EXISTS delete_error TEXT;
ALTER TABLE IF EXISTS session_files ADD COLUMN IF NOT EXISTS delete_next_retry_at TIMESTAMPTZ;

-- 旧暂存行没有 ready_at：视为已就绪，交给 worker 重新扫描
DO $$
BEGIN
    IF to_regclass('public.user_messages') IS NOT NULL THEN
        UPDATE user_messages
        SET ready_at = created_at
        WHERE ready_at IS NULL AND status = 'pending';
    END IF;
END $$;
"""

SCHEMA_SQL = """
-- ==========================================================
-- 1. users 用户表
-- ==========================================================
CREATE TABLE IF NOT EXISTS users (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username              VARCHAR(100) NOT NULL UNIQUE,
    email                 VARCHAR(255) UNIQUE,
    phone                 VARCHAR(30),
    password_hash         TEXT,
    role                  VARCHAR(20) NOT NULL DEFAULT 'user'
                          CHECK (role IN ('admin', 'user')),
    status                VARCHAR(20) NOT NULL DEFAULT 'active'
                          CHECK (status IN ('active', 'disabled', 'locked',
                                            'deleting', 'deleted')),
    last_login_at         TIMESTAMPTZ,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at            TIMESTAMPTZ,
    delete_attempts       INTEGER NOT NULL DEFAULT 0 CHECK (delete_attempts >= 0),
    delete_error          TEXT,
    delete_next_retry_at  TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_users_deleting
    ON users (delete_next_retry_at)
    WHERE deleted_at IS NOT NULL OR status = 'deleting';

-- ==========================================================
-- 2. permissions 权限码定义
-- ==========================================================
CREATE TABLE IF NOT EXISTS permissions (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    permission_code  VARCHAR(100) NOT NULL UNIQUE,
    description      TEXT NOT NULL DEFAULT '',
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ==========================================================
-- 3. user_permissions 用户权限关联
-- ==========================================================
CREATE TABLE IF NOT EXISTS user_permissions (
    user_id          UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    permission_id    UUID NOT NULL REFERENCES permissions(id) ON DELETE CASCADE,
    permission_code  VARCHAR(100) NOT NULL,
    PRIMARY KEY (user_id, permission_id)
);
CREATE INDEX IF NOT EXISTS idx_user_permissions_permission_id
    ON user_permissions (permission_id);
CREATE INDEX IF NOT EXISTS idx_user_permissions_user_code
    ON user_permissions (user_id, permission_code);

-- ==========================================================
-- 4. knowledge_bases 知识库
--    index_version 是 KB 级缓存令牌（D-54）
--    embedding_* 是整库重灌的版本依据（D-57）
-- ==========================================================
CREATE TABLE IF NOT EXISTS knowledge_bases (
    id                       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name                     VARCHAR(200) NOT NULL,
    description              TEXT,
    required_permission_code VARCHAR(100) NOT NULL,
    index_version            BIGINT NOT NULL DEFAULT 0 CHECK (index_version >= 0),
    embedding_model_id       VARCHAR(100) NOT NULL,
    embedding_dimensions     INTEGER NOT NULL CHECK (embedding_dimensions > 0),
    embedding_version        VARCHAR(50) NOT NULL DEFAULT 'v1',
    status                   VARCHAR(20) NOT NULL DEFAULT 'active'
                             CHECK (status IN ('active', 'disabled', 'deleting')),
    created_at               TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at               TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at               TIMESTAMPTZ,
    delete_attempts          INTEGER NOT NULL DEFAULT 0 CHECK (delete_attempts >= 0),
    delete_error             TEXT,
    delete_next_retry_at     TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_knowledge_bases_active
    ON knowledge_bases (status) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_knowledge_bases_deleting
    ON knowledge_bases (delete_next_retry_at) WHERE status = 'deleting';

-- ==========================================================
-- 5. documents 知识库文档
--    completed = 解析成功 + Weaviate 写入成功 + Neo4j 建图成功（D-56）
--    index_* 是复用 documents 的入库任务状态（D-77）
-- ==========================================================
CREATE TABLE IF NOT EXISTS documents (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    kb_id                 UUID NOT NULL REFERENCES knowledge_bases(id) ON DELETE RESTRICT,
    original_filename     VARCHAR(500) NOT NULL,
    description           TEXT,
    extension             VARCHAR(20) NOT NULL,
    size_bytes            BIGINT NOT NULL CHECK (size_bytes >= 0),
    sha256                CHAR(64),
    source_object_key     TEXT NOT NULL,
    parse_manifest_key    TEXT,
    uploaded_by           UUID NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    status                VARCHAR(20) NOT NULL DEFAULT 'pending'
                          CHECK (status IN ('pending', 'processing', 'completed',
                                            'failed', 'deleting', 'deleted')),
    token_count           BIGINT,
    chunk_count           INTEGER,
    index_job_id          UUID,
    index_stage           VARCHAR(40),
    index_progress        REAL
                          CHECK (index_progress IS NULL
                                 OR (index_progress >= 0 AND index_progress <= 1)),
    index_attempts        INTEGER NOT NULL DEFAULT 0 CHECK (index_attempts >= 0),
    index_error           TEXT,
    index_next_retry_at   TIMESTAMPTZ,
    index_locked_until    TIMESTAMPTZ,
    index_locked_by       TEXT,
    index_version         BIGINT,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    parsed_at             TIMESTAMPTZ,
    deleted_at            TIMESTAMPTZ,
    delete_attempts       INTEGER NOT NULL DEFAULT 0 CHECK (delete_attempts >= 0),
    delete_error          TEXT,
    delete_next_retry_at  TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_documents_kb_status
    ON documents (kb_id, status) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_documents_sha256
    ON documents (sha256) WHERE sha256 IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_documents_job
    ON documents (index_job_id) WHERE index_job_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_documents_index_claim
    ON documents (status, index_next_retry_at)
    WHERE deleted_at IS NULL AND status IN ('pending', 'processing', 'failed');
CREATE INDEX IF NOT EXISTS idx_documents_deleting
    ON documents (delete_next_retry_at) WHERE status = 'deleting';

-- ==========================================================
-- 6. conversations 会话
--    status 管会话生命周期；current_run_status 管单次运行
--    current_run_id + lock_expires_at 是 run 租约（P0-8）
-- ==========================================================
CREATE TABLE IF NOT EXISTS conversations (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id             UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title               VARCHAR(300) NOT NULL DEFAULT '',
    status              VARCHAR(20) NOT NULL DEFAULT 'active'
                        CHECK (status IN ('active', 'archived', 'deleting', 'deleted')),
    current_run_status  VARCHAR(20) NOT NULL DEFAULT 'idle'
                        CHECK (current_run_status IN ('idle', 'running', 'waiting_tool',
                                                      'paused', 'failed', 'completed',
                                                      'cancelled')),
    current_run_id      UUID,
    run_started_at      TIMESTAMPTZ,
    lock_expires_at     TIMESTAMPTZ,
    turn_count          BIGINT NOT NULL DEFAULT 0 CHECK (turn_count >= 0),
    total_input_tokens  BIGINT NOT NULL DEFAULT 0 CHECK (total_input_tokens >= 0),
    total_output_tokens BIGINT NOT NULL DEFAULT 0 CHECK (total_output_tokens >= 0),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_message_at     TIMESTAMPTZ,
    deleted_at          TIMESTAMPTZ,
    delete_attempts     INTEGER NOT NULL DEFAULT 0 CHECK (delete_attempts >= 0),
    delete_error        TEXT,
    delete_next_retry_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_conversations_user_last_message
    ON conversations (user_id, last_message_at DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS idx_conversations_active_run
    ON conversations (lock_expires_at)
    WHERE current_run_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_conversations_deleting
    ON conversations (delete_next_retry_at) WHERE status = 'deleting';

-- ==========================================================
-- 7. conversation_messages 原始消息（ground truth）
--    逻辑 append-only；assistant 流式消息允许一次终态 finalize UPDATE
-- ==========================================================
CREATE TABLE IF NOT EXISTS conversation_messages (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id    UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    seq_no             BIGINT NOT NULL CHECK (seq_no > 0),
    client_message_id  UUID,
    role               VARCHAR(20) NOT NULL
                       CHECK (role IN ('user', 'assistant', 'tool', 'system')),
    content            JSONB NOT NULL,
    status             VARCHAR(20) NOT NULL DEFAULT 'completed'
                       CHECK (status IN ('streaming', 'completed', 'partial', 'failed')),
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (conversation_id, seq_no)
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_messages_client_id
    ON conversation_messages (conversation_id, client_message_id)
    WHERE client_message_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_messages_conversation_seq
    ON conversation_messages (conversation_id, seq_no);

-- ==========================================================
-- 8. conversation_summaries 会话摘要（1:1）
-- ==========================================================
CREATE TABLE IF NOT EXISTS conversation_summaries (
    conversation_id     UUID PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
    summary_json        JSONB NOT NULL,
    covered_through_seq BIGINT NOT NULL DEFAULT 0 CHECK (covered_through_seq >= 0),
    compression_count   BIGINT NOT NULL DEFAULT 0 CHECK (compression_count >= 0),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ==========================================================
-- 9. user_memory 长期记忆画像（一人一份）
-- ==========================================================
CREATE TABLE IF NOT EXISTS user_memory (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
    value           JSONB NOT NULL,
    revision        BIGINT NOT NULL DEFAULT 0 CHECK (revision >= 0),
    fact_count      INTEGER NOT NULL DEFAULT 0
                    CHECK (fact_count >= 0 AND fact_count <= 150),
    facts_revision  BIGINT NOT NULL DEFAULT 0 CHECK (facts_revision >= 0),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ==========================================================
-- 10. user_messages 用户消息暂存队列
--     ready_at 由回合终态置位；worker 只领取 ready_at IS NOT NULL
-- ==========================================================
CREATE TABLE IF NOT EXISTS user_messages (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    message_id      UUID NOT NULL UNIQUE
                    REFERENCES conversation_messages(id) ON DELETE CASCADE,
    content         JSONB NOT NULL,
    status          VARCHAR(20) NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'processing', 'failed', 'dead')),
    ready_at        TIMESTAMPTZ,
    attempt_count   INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    last_error      TEXT,
    next_retry_at   TIMESTAMPTZ,
    locked_until    TIMESTAMPTZ,
    locked_by       TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_user_messages_claim
    ON user_messages (user_id, status, ready_at, created_at);
CREATE INDEX IF NOT EXISTS idx_user_messages_retry
    ON user_messages (status, next_retry_at);

-- ==========================================================
-- 11. session_files 会话内上传文件
-- ==========================================================
CREATE TABLE IF NOT EXISTS session_files (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id       UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    message_id            UUID REFERENCES conversation_messages(id) ON DELETE SET NULL,
    uploaded_by           UUID NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    original_filename     VARCHAR(500) NOT NULL,
    extension             VARCHAR(20) NOT NULL,
    size_bytes            BIGINT NOT NULL CHECK (size_bytes >= 0),
    sha256                CHAR(64),
    source_object_key     TEXT NOT NULL,
    parse_manifest_key    TEXT,
    status                VARCHAR(20) NOT NULL DEFAULT 'uploading'
                          CHECK (status IN ('uploading', 'processing', 'available',
                                            'failed', 'deleting', 'deleted')),
    token_count           BIGINT,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at            TIMESTAMPTZ,
    delete_attempts       INTEGER NOT NULL DEFAULT 0 CHECK (delete_attempts >= 0),
    delete_error          TEXT,
    delete_next_retry_at  TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_session_files_conversation_status
    ON session_files (conversation_id, status);
CREATE INDEX IF NOT EXISTS idx_session_files_deleting
    ON session_files (delete_next_retry_at) WHERE status = 'deleting';
"""


async def init_schema(db: PGConnectionManager | None = None) -> None:
    """幂等地升级旧列、创建全部表/索引并写入默认账号。"""
    manager = db or get_pg()
    async with manager.transaction() as conn:
        await conn.execute(MIGRATIONS_SQL)
        await conn.execute(SCHEMA_SQL)
        await seed_default_users(conn)


async def seed_default_users(conn: AsyncConnection) -> None:
    """插入默认账号；用户名已存在时保持原密码不动。"""
    for username, password in DEFAULT_USERS:
        await conn.execute(
            """
            INSERT INTO users (username, password_hash)
            VALUES (%s, %s)
            ON CONFLICT (username) DO NOTHING
            """,
            (username, _hasher.hash(password)),
        )


def schema_statements() -> list[str]:
    """按分号切分语句列表，方便单条执行或排查 DDL。"""
    return [stmt.strip() for stmt in SCHEMA_SQL.split(";") if stmt.strip()]
