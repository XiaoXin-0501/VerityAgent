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
from agent.tools.models import ToolContext, ToolResult, ToolSelection

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
    "FactPort",
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
    ) -> StoredMessage:
        """追加用户消息并分配 seq_no。"""
        ...

    async def append_assistant(self, message: AssistantMessage) -> StoredMessage:
        """追加 assistant 消息并分配 seq_no。"""
        ...

    async def append_tool(self, message: ToolMessage) -> StoredMessage:
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

    def select(self, permissions: frozenset[str]) -> ToolSelection:
        """按权限生成本次暴露的工具集合。"""
        ...

    def activate_group(
        self,
        selection: ToolSelection,
        router_name: str,
        permissions: frozenset[str],
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
    """长期记忆队列的最小写入端口；实际抽取由独立 worker 完成。"""

    async def enqueue_user_message(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        message_id: UUID,
        content: dict,
    ) -> None:
        """回合终态后把用户消息放入待抽取队列（幂等）。"""
        ...

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
    ) -> Any:
        """执行检索；权限过滤由 RAG 流水线负责。"""
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
    # 可选系统提示词；ContextAssembler 尚未接入时的最小替代
    system_prompt: str | None = None
    # 读取最近消息条数；后续由 ContextBudgetPlanner 取代
    history_limit: int = 40
    # 传给 ToolContext 的共享服务对象（RAG/存储等由工具执行器解释）
    services: Any = None
    # 扩展元数据；不进入 Prompt
    metadata: dict[str, Any] = field(default_factory=dict)
