"""长期记忆抽取 Worker 配置。"""

from __future__ import annotations

from pydantic import Field

from .base import ConfigModel

__all__ = ["MemoryConfig"]


class MemoryConfig(ConfigModel):
    """记忆抽取 worker 的运行策略；默认关闭，避免无配置时产生 LLM 费用。"""

    # 是否启动后台记忆抽取 worker
    worker_enabled: bool = False
    # worker 标识；写入 user_messages.locked_by
    worker_id: str = "agent-memory"
    # 单批最多处理的消息数
    worker_batch_size: int = Field(default=20, ge=1)
    # 单批租约时长（秒）
    worker_lease_seconds: int = Field(default=60, ge=10)
    # 队列为空时的轮询间隔（秒）
    worker_scan_interval_s: float = Field(default=30.0, gt=0)
    # 最大尝试次数；超过进入 dead
    worker_max_attempts: int = Field(default=5, ge=1)
    # 失败退避基数（秒）
    worker_retry_backoff_s: float = Field(default=30.0, gt=0)
    # 抽取模型 id；为空使用默认 chat 模型
    extraction_model_id: str | None = None
    # 送入 LLM 的画像/消息最大 token
    extraction_max_input_tokens: int = Field(default=4000, ge=256)
    # 每次画像确认的 confidence 提升步长
    profile_confidence_step: float = Field(default=0.05, gt=0.0, le=1.0)
    # 每个画像段保留的 source_msg_ids 上限；只保留最近证据，避免无限增长
    profile_source_msg_limit: int = Field(default=20, ge=1)
    # 画像抽取输出 token 上限
    extraction_max_output_tokens: int = Field(default=1024, ge=128)

    # 是否启用回合后会话摘要压缩
    compression_enabled: bool = False
    # 压缩模型 id；为空使用默认 chat 模型
    compression_model_id: str | None = None
    # 工作窗口估算 token 超过该值才压缩
    compression_trigger_tokens: int = Field(default=6000, ge=256)
    # 压缩后保留最近消息数
    compression_keep_recent_messages: int = Field(default=10, ge=1)
    # 压缩输入消息/摘要的最大 token
    compression_max_input_tokens: int = Field(default=8000, ge=256)
    # 压缩输出 token 上限
    compression_max_output_tokens: int = Field(default=2048, ge=128)
    # 压缩 LLM 超时（秒）
    compression_timeout_s: float = Field(default=60.0, gt=0)
    # 摘要 topic/fact 的来源锚点上限；只保留最近证据，避免无限增长
    compression_source_msg_limit: int = Field(default=20, ge=1)
    # 单个 topic 保留的轮次锚点上限
    compression_topic_round_limit: int = Field(default=20, ge=1)
    # 成功压缩次数超过该值才启用 topic 遗忘
    compression_topic_forget_after: int = Field(default=5, ge=0)
    # topic 条数超过该值才触发遗忘
    compression_max_topics: int = Field(default=10, ge=1)
    # mention_count <= 该值的 topic 视为“插话”候选
    compression_interjection_max_mentions: int = Field(default=1, ge=1)
    # 最新 last_mentioned_at 的 topic 是否永不删除
    compression_protect_latest_topic: bool = True
    # 单次压缩最多遗忘的 topic 数
    compression_max_forget_per_compression: int = Field(default=5, ge=1)

    # 是否注入长期事实（Weaviate UserFact BM25）
    facts_enabled: bool = True
    # 单次召回事实数
    fact_top_k: int = Field(default=5, ge=1)
    # 事实最低置信度；为空不限制
    fact_min_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    # 是否把抽取结果写入 Weaviate UserFact
    fact_write_enabled: bool = False
    # 是否启用事实冲突判定（规则优先，必要时 LLM 兜底）
    fact_conflict_enabled: bool = False
    # 冲突判定模型 id；为空使用默认 chat
    fact_conflict_model_id: str | None = None
    # 冲突判定候选事实数
    fact_conflict_max_candidates: int = Field(default=10, ge=1)
    # 冲突判定 LLM 超时（秒）
    fact_conflict_timeout_s: float = Field(default=20.0, gt=0)
    # 冲突判定 LLM 输出 token 上限（只输出简短 JSON 决策）
    fact_conflict_max_output_tokens: int = Field(default=128, ge=16)

    # 是否启动用户跨存储删除 worker
    deletion_worker_enabled: bool = False
    # 单批删除用户数
    deletion_batch_size: int = Field(default=20, ge=1)
    # 无待删除任务时的轮询间隔（秒）
    deletion_scan_interval_s: float = Field(default=60.0, gt=0)
    # 删除失败重试间隔（秒）
    deletion_retry_backoff_s: float = Field(default=60.0, gt=0)

    # 是否启动事实数量对账 worker
    reconciliation_worker_enabled: bool = False
    # 单批对账用户数
    reconciliation_batch_size: int = Field(default=50, ge=1)
    # 对账轮询间隔（秒）
    reconciliation_interval_s: float = Field(default=300.0, gt=0)
