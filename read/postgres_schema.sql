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
    -- 领域画像名；NULL 表示使用 IndexerConfig.domain_profile
    domain_profile           VARCHAR(100),
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
    -- 轮次 id：一次用户提问到完整回答的全部消息共享同一个值
    round_id           UUID,
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
CREATE INDEX IF NOT EXISTS idx_messages_conversation_round
    ON conversation_messages (conversation_id, round_id, seq_no)
    WHERE round_id IS NOT NULL;

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
