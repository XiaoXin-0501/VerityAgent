"""Hooks 模块配置。"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["HooksConfig", "HookSinkConfig"]


class HookSinkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # v1 只有日志 sink；后续可接指标/追踪后端
    log_enabled: bool = True
    # 日志 sink 的级别：DEBUG / INFO / WARNING / ERROR
    log_level: str = "INFO"
    # 采样率 0~1；1 表示全量记录
    sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)


class HooksConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # 是否启用 Hooks；关闭后所有观察者不执行
    enabled: bool = True
    # 观察者失败不阻塞主流程（D-65）
    fail_open: bool = True
    # 单个观察者派发超时（秒）
    dispatch_timeout_s: float = Field(default=1.0, gt=0)
    # 异步派发队列上限
    max_queue_size: int = Field(default=1000, ge=1)
    # 队列满时是否丢弃事件而不是阻塞主流程
    drop_on_overflow: bool = True
    # 输出 sink 配置
    sink: HookSinkConfig = Field(default_factory=HookSinkConfig)
