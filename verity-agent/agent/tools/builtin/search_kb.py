"""内置 ``search_kb`` 工具：把 RAG Retriever 暴露给 Agent。

工具只负责参数校验、权限上下文传递和结果格式化；KB 权限仍由检索流水线在
召回前执行，工具不会自行拼接 SQL 或直接访问 Weaviate。
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from agent.tools.enums import ToolKind, ToolSideEffect
from agent.tools.models import ToolContext, ToolError, ToolResult, ToolSpec
from agent.tools.policies import (
    CachePolicy,
    CircuitBreakerPolicy,
    RateLimitPolicy,
    ToolPolicy,
)
from agent.tools.registry import ToolRegistry
from config.tools import SearchKbToolConfig
from exception.tool import ToolErrorCode
from rag.shared.models import RetrievalRequest

__all__ = ["SearchKbArgs", "register_search_kb", "search_kb_executor"]

logger = logging.getLogger(__name__)


class SearchKbArgs(BaseModel):
    """``search_kb`` 的 LLM 参数模型。"""

    query: str = Field(min_length=1, max_length=2000)
    kb_ids: list[UUID] = Field(min_length=1, max_length=20)
    # 为空时使用 ToolsConfig.search_kb.default_top_k；上限由执行器按配置收紧
    top_k: int | None = Field(default=None, ge=1, le=50)


def _format_chunks(result: Any) -> tuple[str, dict[str, Any]]:
    """把 RetrievalResult 转成 LLM 文本 + UI 结构化数据。"""
    chunks = list(getattr(result, "chunks", []) or [])
    if not chunks:
        return "未检索到与问题相关的知识库片段。", {
            "query": getattr(result, "query", None),
            "chunks": [],
            "documents": {},
            "citations": [],
            "trace_id": getattr(getattr(result, "trace", None), "trace_id", None),
            "duration_ms": getattr(result, "duration_ms", None),
        }
    lines: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        title = getattr(chunk, "doc_name", None) or str(chunk.doc_id)
        heading = getattr(chunk, "heading", None)
        label = f"{title} / {heading}" if heading else title
        lines.append(f"[{index}] {label}\n{chunk.content}".strip())
    data = {
        "query": getattr(result, "query", None),
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": str(chunk.doc_id),
                "doc_name": getattr(chunk, "doc_name", None),
                "heading": getattr(chunk, "heading", None),
                "content": chunk.content,
                "score": getattr(chunk, "score", None),
                "citation": (
                    chunk.citation.model_dump(mode="json")
                    if getattr(chunk, "citation", None) is not None
                    else None
                ),
            }
            for chunk in chunks
        ],
        "documents": {
            str(doc_id): (
                meta.model_dump(mode="json")
                if hasattr(meta, "model_dump")
                else dict(meta)
            )
            for doc_id, meta in (
                getattr(result, "documents", {}) or {}
            ).items()
        },
        "citations": [
            citation.model_dump(mode="json")
            for citation in (getattr(result, "citations", []) or [])
        ],
        "trace_id": getattr(getattr(result, "trace", None), "trace_id", None),
        "duration_ms": getattr(result, "duration_ms", None),
    }
    return "\n\n".join(lines), data


async def search_kb_executor(
    arguments: SearchKbArgs,
    context: ToolContext,
    *,
    config: SearchKbToolConfig | None = None,
) -> ToolResult:
    """执行知识库检索；失败转 ToolResult，不把底层异常抛给主循环。"""
    retriever = getattr(context.services, "rag_retriever", None)
    if retriever is None:
        return ToolResult.failure(
            call_id=context.call_id,
            tool_name=context.tool_name,
            error=ToolError(
                code=ToolErrorCode.UPSTREAM_ERROR,
                message="知识库检索未配置",
                retryable=False,
            ),
        )
    policy = config or SearchKbToolConfig()
    # 未显式指定时使用运维配置的默认值，并统一收紧到允许上限。
    top_k = arguments.top_k if arguments.top_k is not None else policy.default_top_k
    top_k = min(top_k, policy.max_top_k)
    # 查询规划由 Agent 主循环注入：用户原话、改写 query、召回意图。
    # 工具自身选择的 query 仍是主召回 query；规划字段只补充召回/rerank。
    plan = context.metadata.get("query_plan")
    original_query = getattr(plan, "original_query", None) or arguments.query
    rewritten_query = getattr(plan, "rewritten_query", None)
    rag_intent = (
        getattr(plan, "rag_intent_label", None) if plan is not None else None
    )
    request = RetrievalRequest(
        query=arguments.query,
        original_query=original_query,
        rewritten_query=rewritten_query,
        rag_intent=rag_intent,
        kb_ids=list(arguments.kb_ids),
        user_id=context.user_id,
        permissions=context.permissions,
        top_k=top_k,
        trace_id=context.trace_id,
    )
    try:
        # RAG 检索事件由 Retriever 内部的 RagEventSink 统一发送，工具层
        # 不再重复 emit rag.retrieval.*；工具执行本身由 ToolPipeline 记录。
        result = await retriever.retrieve(request)
    except Exception as exc:  # noqa: BLE001 - 工具错误统一结构化
        logger.warning("search_kb 检索失败: %s", exc, exc_info=True)
        return ToolResult.failure(
            call_id=context.call_id,
            tool_name=context.tool_name,
            error=ToolError(
                code=ToolErrorCode.UPSTREAM_ERROR,
                message="知识库检索失败，请稍后重试",
                retryable=True,
                cause=str(exc),
            ),
        )
    content, data = _format_chunks(result)
    return ToolResult.success(
        call_id=context.call_id,
        tool_name=context.tool_name,
        content=content,
        data=data,
    )


def register_search_kb(
    registry: ToolRegistry,
    retriever: Any | None,
    *,
    config: SearchKbToolConfig | None = None,
) -> None:
    """注册 ``search_kb``；没有可用 Retriever 时保持不暴露，避免误导 LLM。"""
    if retriever is None:
        logger.info("RAG Retriever 未就绪，跳过 search_kb 注册")
        return

    policy = config or SearchKbToolConfig()

    async def _executor(
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """适配 ToolExecutor 协议：校验参数后调用业务执行器。"""
        parsed = SearchKbArgs.model_validate(arguments)
        # 让执行器拿到实际的 call_id/tool_name；ToolPipeline 会再归一化一次。
        bound = context.model_copy(
            update={"call_id": context.call_id, "tool_name": "search_kb"}
        )
        return await search_kb_executor(parsed, bound, config=policy)

    registry.register_tool(
        ToolSpec(
            name="search_kb",
            description=(
                "在指定知识库中检索相关片段。用户询问宠物疾病、疫苗、"
                "诊疗、药品或文档内容时优先使用；返回片段和引用。"
            ),
            group="kb",
            kind=ToolKind.EXECUTABLE,
            side_effect=ToolSideEffect.READ,
            policy=ToolPolicy(
                # 检索结果由 RAG 的 RetrievalCache 负责（key 已包含改写 query/
                # 召回意图/索引版本/权限）；工具层再缓存会掩盖规划字段变化，
                # 因此这里只保留熔断与限流。
                cache=CachePolicy(enabled=False),
                circuit_breaker=CircuitBreakerPolicy(enabled=True),
                rate_limit=RateLimitPolicy(
                    enabled=True,
                    max_calls=policy.rate_limit_calls,
                    window_s=policy.rate_limit_window_s,
                ),
            ),
            tags=["rag", "knowledge"],
            # 不声明权限范围缓存：工具层缓存已关闭，检索缓存由 RAG 负责。
            metadata={},
        ),
        _executor,
        argument_model=SearchKbArgs,
        source="builtin",
    )
