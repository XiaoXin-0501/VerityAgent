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
    # 压缩 LLM 超时（秒）
    compression_timeout_s: float = Field(default=60.0, gt=0)

    # 是否注入长期事实（Weaviate UserFact BM25）
    facts_enabled: bool = True
    # 单次召回事实数
    fact_top_k: int = Field(default=5, ge=1)
    # 事实最低置信度；为空不限制
    fact_min_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    # 是否把抽取结果写入 Weaviate UserFact
    fact_write_enabled: bool = False

    # 是否启动用户跨存储删除 worker
    deletion_worker_enabled: bool = False
    # 单批删除用户数
    deletion_batch_size: int = Field(default=20, ge=1)
    # 删除失败重试间隔（秒）
    deletion_retry_backoff_s: float = Field(default=60.0, gt=0)

    # 是否启动事实数量对账 worker
    reconciliation_worker_enabled: bool = False
    # 单批对账用户数
    reconciliation_batch_size: int = Field(default=50, ge=1)
    # 对账轮询间隔（秒）
    reconciliation_interval_s: float = Field(default=300.0, gt=0)
