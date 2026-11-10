"""Agent 模块配置（公共 DTO 与内部策略的启动默认值）。"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["AgentConfig", "AgentPolicy", "AgentStreamConfig"]


class AgentStreamConfig(BaseModel):
    """SSE 流式输出配置。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # SSE 心跳间隔（秒）；空闲时发送 keepalive
    heartbeat_interval_s: float = Field(default=15.0, gt=0)
    # 单连接待发送事件缓冲上限；超过按背压策略处理
    max_buffer_events: int = Field(default=512, ge=16)
    # 断线重放缓冲事件数；用于 Last-Event-ID 续传
    replay_buffer_size: int = Field(default=1024, ge=64)


class AgentPolicy(BaseModel):
    """内部运行限制；由 AgentConfig 派生，不直接暴露给调用方。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # 单次 run 最多回合数；防止无限循环
    max_turns_per_run: int = Field(default=20, ge=1)
    # 单回合最多工具轮数
    max_tool_rounds: int = Field(default=8, ge=1)
    # 单批最多并行工具数
    max_parallel_tools: int = Field(default=4, ge=1)
    # 单次 run 最多 LLM 调用次数
    max_llm_calls_per_run: int = Field(default=20, ge=1)
    # 会话 run 租约时长（秒）；过期可被新 run 接管
    run_lease_seconds: int = Field(default=120, ge=10)
    # run 续租间隔（秒）；应小于租约时长
    run_heartbeat_interval_s: float = Field(default=30.0, gt=0)
    # 单回合总超时（秒）
    turn_timeout_s: float = Field(default=900.0, gt=0)


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # 运行限制（回合/工具/LLM 次数/租约）
    policy: AgentPolicy = Field(default_factory=AgentPolicy)
    # SSE 心跳与重放缓冲
    stream: AgentStreamConfig = Field(default_factory=AgentStreamConfig)
    # 运行期状态落 Redis 时的 TTL；必须大于 run 租约
    run_state_ttl_s: int = Field(default=3600, ge=60)
    # 取消 run 时等待工具/流收尾的宽限时间（秒）
    cancel_grace_s: float = Field(default=5.0, ge=0)
