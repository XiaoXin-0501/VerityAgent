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
ALTER TABLE IF EXISTS knowledge_bases ADD COLUMN IF NOT EXISTS domain_profile VARCHAR(100);
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

-- conversation_messages：幂等键 / finalize 时间 / 轮次 id
ALTER TABLE IF EXISTS conversation_messages ADD COLUMN IF NOT EXISTS client_message_id UUID;
ALTER TABLE IF EXISTS conversation_messages ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT now();
ALTER TABLE IF EXISTS conversation_messages ADD COLUMN IF NOT EXISTS round_id UUID;

-- 旧消息回填轮次：以“最近一条 user 消息”为界，user 消息自身 id 作为该轮
-- 轮次 id；第一条 user 消息之前的 system/遗留行保持 NULL。
DO $$
BEGIN
    IF to_regclass('public.conversation_messages') IS NOT NULL THEN
        UPDATE conversation_messages AS m
        SET round_id = (
            SELECT u.id
            FROM conversation_messages AS u
            WHERE u.conversation_id = m.conversation_id
              AND u.seq_no <= m.seq_no
              AND u.role = 'user'
            ORDER BY u.seq_no DESC
            LIMIT 1
        )
        WHERE m.round_id IS NULL
          AND EXISTS (
              SELECT 1
              FROM conversation_messages AS u
              WHERE u.conversation_id = m.conversation_id
                AND u.seq_no <= m.seq_no
                AND u.role = 'user'
          );
    END IF;
END $$;

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
