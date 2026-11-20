"""Agent 模块配置（公共 DTO 与内部策略的启动默认值）。"""

from __future__ import annotations

from pydantic import Field, model_validator

from .base import ConfigModel

__all__ = [
    "AgentConfig",
    "AgentPolicy",
    "AgentStreamConfig",
    "QueryPlanConfig",
    "ReflectionConfig",
]


class QueryPlanConfig(ConfigModel):
    """Query 规划（意图识别 + query 改写）配置。

    Agent 在每次 run 的上下文装配前执行一次：先用可解释规则识别明确
    不需要 RAG 的请求；其余请求交给 LLM，输入包含用户画像、当前会话
    历史、垂直领域目录与实体类型参考，输出改写 query、RAG 召回意图、
    扩写后的用户意图与检索实体。
    """

    # 是否启用 query 规划；关闭时不做意图识别/改写/RAG 决策。
    # 注意：INJECTED RAG 依赖该决策，关闭后不会自动注入检索上下文；
    # 工具模式（LLM 主动调用 search_kb）仍正常工作。
    enabled: bool = True
    # 是否调用 LLM 规划；规则只对“明确不需要 RAG”做短路，其余交给 LLM
    use_llm: bool = True
    # 规划模型 id；为空时使用 Agent 默认 chat 模型
    model_id: str | None = None
    # LLM 超时（秒）；超时保留规则结果
    timeout_s: float = Field(default=8.0, gt=0)
    # LLM 最大输出 token；输出是短 JSON
    max_output_tokens: int = Field(default=512, ge=64)
    # 注入规划提示词的历史消息条数；0 表示不注入历史
    history_messages: int = Field(default=8, ge=0)
    # 用户画像注入上限（token）
    profile_token_budget: int = Field(default=512, ge=0)
    # 会话摘要注入上限（token）；避免长摘要挤占规划提示词
    summary_token_budget: int = Field(default=1024, ge=0)
    # 垂直领域目录注入上限（字符）
    domain_catalog_chars: int = Field(default=2000, ge=0)
    # 领域实体类型参考注入上限（字符）；0 表示不注入画像类型，使用通用兜底
    domain_entity_types_chars: int = Field(default=1600, ge=0)
    # 低于该字符数的 query 不调用 LLM，直接用规则结果
    min_query_chars: int = Field(default=2, ge=1)


class ReflectionConfig(ConfigModel):
    """主循环反思配置。

    一次 run 的循环每推进 ``interval_steps`` 次，插入一次独立反思调用：
    检查当前方向是否仍在回答用户问题、工具结果是否有进展；可以继续、
    调整方向或止损并强制收敛到最终回答。
    """

    # 是否启用循环反思
    enabled: bool = True
    # 每多少次主循环迭代反思一次；1 表示每轮都反思
    interval_steps: int = Field(default=3, ge=1)
    # 反思模型 id；为空时使用 Agent 默认 chat 模型
    model_id: str | None = None
    # 反思 LLM 超时（秒）
    timeout_s: float = Field(default=15.0, gt=0)
    # 反思 LLM 最大输出 token；输出是短 JSON
    max_output_tokens: int = Field(default=512, ge=64)
    # 注入反思提示词的单个工具结果最多字符数
    max_tool_result_chars: int = Field(default=1200, ge=100)
    # 反思判定“止损”时是否清空工具并强制最终回答
    force_final_on_stop: bool = True


class AgentStreamConfig(ConfigModel):
    """SSE 流式输出配置。"""

    # SSE 心跳间隔（秒）；空闲时发送 keepalive
    heartbeat_interval_s: float = Field(default=15.0, gt=0)
    # 单连接待发送事件缓冲上限；超过按背压策略处理
    max_buffer_events: int = Field(default=512, ge=16)
    # 断线重放缓冲事件数；用于 Last-Event-ID 续传
    replay_buffer_size: int = Field(default=1024, ge=64)


class AgentPolicy(ConfigModel):
    """内部运行限制；由 AgentConfig 派生，不直接暴露给调用方。"""

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


class AgentConfig(ConfigModel):
    # 运行限制（回合/工具/LLM 次数/租约）
    policy: AgentPolicy = Field(default_factory=AgentPolicy)
    # Query 规划：规则短路 + LLM 规划（意图识别/query 改写/RAG 决策）
    query_plan: QueryPlanConfig = Field(default_factory=QueryPlanConfig)
    # 主循环反思：每 N 次循环检查方向，必要时止损
    reflection: ReflectionConfig = Field(default_factory=ReflectionConfig)
    # SSE 心跳与重放缓冲
    stream: AgentStreamConfig = Field(default_factory=AgentStreamConfig)
    # 运行期状态落 Redis 时的 TTL；必须大于 run 租约
    run_state_ttl_s: int = Field(default=3600, ge=60)
    # 工作窗口单次最多加载的消息条数（安全阀）；实际保留由
    # ContextBudgetPlanner 按 token 预算决定，不是固定召回条数。
    history_limit: int = Field(default=400, ge=1)
    # 取消 run 时等待工具/流收尾的宽限时间（秒）
    cancel_grace_s: float = Field(default=5.0, ge=0)
    # 是否将事件镜像到 Redis Stream，支持多实例 replay/resync
    event_store_enabled: bool = True
    # 跨实例事件轮询间隔（秒）；Pub/Sub 不可用时的兜底
    remote_event_poll_interval_s: float = Field(default=0.5, gt=0)
    # 跨实例取消轮询间隔（秒）；Pub/Sub 不可用时的兜底
    remote_cancel_poll_interval_s: float = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def _check_lease_heartbeat(self) -> "AgentConfig":
        """续租间隔必须短于租约，否则会在租约过期后才尝试续租。"""
        if self.policy.run_heartbeat_interval_s >= self.policy.run_lease_seconds:
            raise ValueError(
                "run_heartbeat_interval_s 必须小于 run_lease_seconds"
            )
        return self
