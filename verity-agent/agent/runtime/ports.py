"""Agent 依赖端口。

AgentLoop 只依赖这些窄协议；生产适配器在 ``agent.adapters`` 中实现，
测试用 fake 端口即可，不启动 PG/Redis/Provider。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import UUID

from collections.abc import AsyncIterator

from agent.llm import FilePart, LLMRequest, LLMResponse, LLMStreamEvent
from agent.message.models import (
    AssistantMessage,
    StoredMessage,
    ToolMessage,
    UserMessage,
)
from agent.tools.models import (
    ToolContext,
    ToolExposureContext,
    ToolResult,
    ToolSelection,
)

__all__ = [
    "LeasePort",
    "MessagePort",
    "LLMPort",
    "ToolPort",
    "HookPort",
    "AttachmentPort",
    "MemoryQueuePort",
    "MemoryPort",
    "RagPort",
    "KbCatalogPort",
    "FactPort",
    "QueryPlannerPort",
    "FlowLogPort",
    "AgentDependencies",
]


class LeasePort(Protocol):
    """会话级 run 租约；同一会话同一时刻只允许一个 run。"""

    async def acquire(
        self,
        conversation_id: UUID,
        run_id: str,
        lease_seconds: int,
    ) -> bool:
        """条件抢占；成功 True，已有活跃 run 返回 False。"""
        ...

    async def heartbeat(self, conversation_id: UUID, run_id: str, lease_seconds: int) -> bool:
        """续租；租约已丢失返回 False。"""
        ...

    async def release(
        self,
        conversation_id: UUID,
        run_id: str,
        status: str,
    ) -> None:
        """幂等释放租约并写终态投影。"""
        ...


class MessagePort(Protocol):
    """消息事实源；Agent 只做领域对象转换，不拼 SQL/seq_no。"""

    async def find_by_client_message_id(
        self,
        conversation_id: UUID,
        client_message_id: UUID,
    ) -> StoredMessage | None:
        """幂等查询；未命中返回 None。"""
        ...

    async def append_user(
        self,
        message: UserMessage,
        *,
        client_message_id: UUID,
        user_id: UUID | None = None,
        round_id: UUID | None = None,
    ) -> StoredMessage:
        """追加用户消息并分配 seq_no。

        ``user_id`` 非空时，实现方应在同一事务内同步写入长期记忆暂存行
        （``user_messages``，``ready_at`` 仍为空），避免崩溃后漏抽取。
        ``round_id`` 标识“一次提问到完整回答”的轮次，同一回合的后续
        assistant/tool 消息必须复用同一个值。
        """
        ...

    async def append_assistant(
        self,
        message: AssistantMessage,
        *,
        round_id: UUID | None = None,
    ) -> StoredMessage:
        """追加 assistant 消息并分配 seq_no。"""
        ...

    async def append_tool(
        self,
        message: ToolMessage,
        *,
        round_id: UUID | None = None,
    ) -> StoredMessage:
        """追加 ToolMessage 并分配 seq_no。"""
        ...

    async def load_window(
        self,
        conversation_id: UUID,
        *,
        limit: int = 40,
    ) -> list[StoredMessage]:
        """按 seq_no 升序读取最近消息。"""
        ...

    async def load_after(
        self,
        conversation_id: UUID,
        seq_no: int,
        *,
        limit: int = 200,
    ) -> list[StoredMessage]:
        """读取 ``seq_no`` 之后的消息；幂等命中时重建已有回合。"""
        ...

    async def load_recent_after(
        self,
        conversation_id: UUID,
        seq_no: int,
        *,
        limit: int = 400,
    ) -> list[StoredMessage]:
        """读取 ``seq_no`` 之后最近的 N 条消息，按 seq 升序返回。

        上下文工作窗口使用本方法；实现方不能返回该区间最老的 N 条。
        """
        ...


class LLMPort(Protocol):
    """LLM 应用层端口；Agent 不直接 import provider。"""

    async def chat(self, request: LLMRequest) -> LLMResponse:
        """非流式调用；失败抛异常，由 loop 转成 AgentError。"""
        ...

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        """流式调用；实现方返回应用层统一事件。"""
        ...


class ToolPort(Protocol):
    """工具选择与批量执行端口。"""

    def select(
        self,
        permissions: frozenset[str],
        *,
        exposure: ToolExposureContext | None = None,
    ) -> ToolSelection:
        """按权限与本次请求暴露策略生成工具集合。"""
        ...

    def activate_group(
        self,
        selection: ToolSelection,
        router_name: str,
        permissions: frozenset[str],
        *,
        exposure: ToolExposureContext | None = None,
    ) -> ToolSelection:
        """router 调用后展开组内工具；不支持路由的端口可不实现。"""
        ...

    async def execute_batch(
        self,
        calls: list,
        context: ToolContext,
        *,
        max_concurrency: int | None = None,
    ) -> list[ToolResult]:
        """执行一批工具；返回顺序与入参一致。"""
        ...


class HookPort(Protocol):
    """Hooks 最小端口；没有 HookScope 时可不注入。"""

    async def emit(self, event: str, payload: Any = None, **kwargs: Any) -> Any:
        """发送观测事件；失败不应影响主循环。"""
        ...


class AttachmentPort(Protocol):
    """会话附件解析端口；把 file_id 解析成模型可消费的文本/URL。"""

    async def resolve(
        self,
        file_id: UUID,
        *,
        conversation_id: UUID,
        user_id: UUID,
    ) -> FilePart:
        """返回可发送给模型的 FilePart；未就绪时抛异常。"""
        ...


class MemoryQueuePort(Protocol):
    """长期记忆队列的回合终态端口；入队由 MessagePort 同事务完成。"""

    async def mark_ready(self, conversation_id: UUID) -> None:
        """把该会话未就绪的暂存行标为可抽取。"""
        ...


class MemoryPort(Protocol):
    """读取用户长期画像/会话摘要的最小端口；不做事实检索/写入。"""

    async def load_profile(
        self,
        user_id: UUID,
        *,
        token_budget: int | None = None,
    ) -> str | None:
        """返回可注入 system 的画像文本；无画像返回 None。"""
        ...

    async def load_session_summary(
        self,
        conversation_id: UUID,
        *,
        token_budget: int | None = None,
    ) -> str | None:
        """返回可注入 system 的会话摘要文本；无摘要返回 None。"""
        ...

    async def load_session_summary_state(
        self,
        conversation_id: UUID,
        *,
        token_budget: int | None = None,
    ) -> tuple[str | None, int]:
        """返回 (摘要文本, covered_through_seq)；实现方可选提供。"""
        ...

    async def compression_count(self, conversation_id: UUID) -> int:
        """返回会话摘要成功压缩次数；无摘要返回 0。"""
        ...


class RagPort(Protocol):
    """INJECTED RAG 只读检索端口；返回 RAG 领域结果，由 Agent 渲染成 section。"""

    async def retrieve(
        self,
        *,
        query: str,
        kb_ids: list[UUID],
        user_id: UUID,
        permissions: frozenset[str],
        top_k: int | None = None,
        trace_id: str | None = None,
        original_query: str | None = None,
        rewritten_query: str | None = None,
        rewritten_queries: list[str] | None = None,
        rag_intent: str | None = None,
        entities: list[str] | None = None,
    ) -> Any:
        """执行检索；权限过滤由 RAG 流水线负责。

        ``query`` 是主召回 query；``original_query``/``rewritten_query``/
        ``rewritten_queries``/``rag_intent``/``entities`` 由 Agent 查询规划
        提供，RAG 只消费不识别。``rewritten_queries`` 用于中英文双语变体，
        与 ``rewritten_query`` 一起去重后并发召回。
        """
        ...


class KbCatalogPort(Protocol):
    """知识库可读目录：未显式给 kb_ids 时解析用户可读知识库。

    权限事实源是 PG 的 ``knowledge_bases.required_permission_code``；
    Agent 与 search_kb 共用同一实现，避免各自拼权限码。
    """

    async def list_readable(
        self,
        *,
        user_id: UUID,
        permissions: frozenset[str],
        limit: int = 20,
    ) -> list[UUID]:
        """返回用户可读的 active 知识库 id。"""
        ...


class QueryPlannerPort(Protocol):
    """查询规划端口：规则短路 + LLM 规划。"""

    async def plan(
        self,
        original_query: str,
        *,
        profile_text: str | None = None,
        session_summary: str | None = None,
        history_text: str | None = None,
        domain_catalog: str | None = None,
        trace_id: str,
        conversation_id: UUID,
        turn_id: str,
        user_id: UUID,
    ) -> Any:
        """返回带 ``plan``/``usage`` 的规划结果。"""
        ...


class FlowLogPort(Protocol):
    """结构化流程详情日志端口；按 run 绑定写入器。"""

    def bind_run(
        self,
        *,
        run_id: str,
        conversation_id: UUID | None,
        turn_id: str | None,
        user_id: UUID | None,
    ) -> Any:
        """返回带 ``event(name, **fields)`` 的 run 写入器。"""
        ...


class FactPort(Protocol):
    """长期事实 BM25 召回端口；事实写入由 memory worker 负责。"""

    async def recall(
        self,
        *,
        user_id: UUID,
        query: str,
        top_k: int = 5,
        min_confidence: float | None = None,
    ) -> list[Any]:
        """按 query 召回当前用户 active 事实。"""
        ...


@dataclass(slots=True)
class AgentDependencies:
    """Agent 运行依赖；只放端口和稳定配置，不放全局单例。"""

    # LLM 调用端口
    llm: LLMPort
    # 默认模型 id；请求可覆盖
    model_id: str
    # 消息事实源
    messages: MessagePort
    # 会话租约
    leases: LeasePort
    # 工具选择/执行；为空表示本次运行不提供工具
    tools: ToolPort | None = None
    # Hooks；为空表示不观测
    hooks: HookPort | None = None
    # 附件解析；为空表示仅支持已带 url/text 的附件
    attachments: AttachmentPort | None = None
    # 长期记忆队列入队/就绪标记；为空表示不启用记忆抽取链路
    memory_queue: MemoryQueuePort | None = None
    # 长期画像读取；为空表示不注入 memory section
    memory: MemoryPort | None = None
    # Context 组装器；为空时 loop 直接使用 system + 最近消息
    context: Any = None
    # INJECTED RAG；为空表示只走工具模式
    rag: RagPort | None = None
    # 知识库可读目录；INJECTED 未显式给 kb_ids 时用于解析默认检索范围
    kb_catalog: KbCatalogPort | None = None
    # 查询规划（规则 + LLM）：意图识别/query 改写/RAG 决策/事实召回意图
    query_planner: QueryPlannerPort | None = None
    # 垂直领域目录文本（主画像 + 子画像），注入查询规划提示词
    domain_catalog: str | None = None
    # 结构化流程详情日志；为空表示不记录
    flow_log: FlowLogPort | None = None
    # 长期事实 BM25 召回；为空表示不注入 facts section
    facts: FactPort | None = None
    # 事实召回数量/最低置信度
    fact_top_k: int = 5
    fact_min_confidence: float | None = None
    # 回合后会话摘要压缩；为空表示不压缩
    compressor: Any = None
    # Redis 事件/快照存储；为空表示只使用内存 ring buffer
    event_store: Any = None
    # 后台任务运行器；为空时 Agent 自己创建一个进程内 runner
    background_tasks: Any = None
    # 基础 system prompt；ContextAssembler 把它固定在头部，memory/RAG 等
    # 动态 section 作为末尾补充 system 注入
    system_prompt: str | None = None
    # 单次加载消息安全上限；实际工作窗口由 ContextBudgetPlanner 决定
    history_limit: int = 400
    # 传给 ToolContext 的共享服务对象（RAG/存储等由工具执行器解释）
    services: Any = None
    # 扩展元数据；不进入 Prompt
    metadata: dict[str, Any] = field(default_factory=dict)
