"""内置 ``search_kb`` 工具：把 RAG Retriever 暴露给 Agent。

工具只负责参数校验、权限上下文传递和结果格式化；KB 权限仍由检索流水线在
召回前执行，工具不会自行拼接 SQL 或直接访问 Weaviate。
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from agent.tools.declaration import ToolBuildContext, ToolDeclaration
from agent.tools.factory import SimpleToolFactory
from agent.tools.models import ToolContext, ToolError, ToolResult
from agent.tools.services import ToolServiceKey, get_tool_service
from config.tools import SearchKbToolConfig
from exception import RagError, RagErrorCode
from exception.tool import ToolErrorCode
from rag.shared.models import RetrievalRequest

__all__ = ["SearchKbArgs", "SearchKbFactory", "build_search_kb_executor", "search_kb_executor"]

logger = logging.getLogger(__name__)


class SearchKbArgs(BaseModel):
    """``search_kb`` 的 LLM 参数模型。"""

    query: str = Field(min_length=1, max_length=2000)
    # 省略时由服务端解析“当前用户可读的全部知识库”；模型不应猜测内部 id。
    kb_ids: list[UUID] = Field(default_factory=list, max_length=20)
    # 为空时使用 ToolsConfig.search_kb.default_top_k；上限由执行器按配置收紧
    top_k: int | None = Field(default=None, ge=1, le=50)


def _warning_codes(result: Any) -> list[str]:
    """提取检索告警码；保留顺序并去重。"""
    codes: list[str] = []
    for item in getattr(result, "warnings", []) or []:
        code = getattr(item, "code", None) or item
        text = str(getattr(code, "value", code) or "").strip()
        if text and text not in codes:
            codes.append(text)
    return codes


def _format_chunks(result: Any) -> tuple[str, dict[str, Any]]:
    """把 RetrievalResult 转成 LLM 文本 + UI 结构化数据。"""
    chunks = list(getattr(result, "chunks", []) or [])
    warnings = _warning_codes(result)
    if not chunks:
        content = "未检索到与问题相关的知识库片段。"
        # 告警必须进入 LLM 可见文本：embedding 403/BM25 空结果时，
        # “没有检索到”不等于“知识库没有内容”，否则模型会错误断言无资料。
        if "EMBEDDING_FAILED" in warnings:
            content += (
                "（检索服务降级：向量通道不可用，且关键词通道未返回候选；"
                "这不代表知识库一定没有相关内容。可稍后重试，或换用更具体的关键词。）"
            )
        elif warnings:
            content += f"（检索告警: {', '.join(warnings)}；可尝试换关键词或稍后重试。）"
        return content, {
            "query": getattr(result, "query", None),
            "rewritten_query": getattr(result, "rewritten_query", None),
            "rewritten_queries": list(
                getattr(result, "rewritten_queries", []) or []
            ),
            "rag_intent": getattr(result, "rag_intent", None),
            "chunks": [],
            "documents": {},
            "citations": [],
            "warnings": warnings,
            "trace_id": getattr(getattr(result, "trace", None), "trace_id", None),
            "duration_ms": getattr(result, "duration_ms", None),
        }
    lines: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        title = getattr(chunk, "doc_name", None) or str(chunk.doc_id)
        heading = getattr(chunk, "heading", None)
        label = f"{title} / {heading}" if heading else title
        # chunk_id 必须进入给 LLM 的文本：resolve_chunk_conflict 需要模型
        # 从检索结果中挑选真实片段 id；结构化 data 不会进入 ToolMessage。
        chunk_id = getattr(chunk, "chunk_id", None)
        prefix = (
            f"[{index}] chunk_id={chunk_id} | {label}"
            if chunk_id
            else f"[{index}] {label}"
        )
        lines.append(f"{prefix}\n{chunk.content}".strip())
    if "EMBEDDING_FAILED" in warnings:
        lines.append(
            "> 检索告警：向量通道不可用，本次只返回关键词/图通道候选，"
            "完整性可能受影响。"
        )
    data = {
        "query": getattr(result, "query", None),
        "rewritten_query": getattr(result, "rewritten_query", None),
        "rewritten_queries": list(getattr(result, "rewritten_queries", []) or []),
        "rag_intent": getattr(result, "rag_intent", None),
        "warnings": warnings,
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": str(chunk.doc_id),
                "doc_name": getattr(chunk, "doc_name", None),
                "heading": getattr(chunk, "heading", None),
                "content": chunk.content,
                "score": getattr(chunk, "score", None),
                # 观测字段：图 next-hop 召回与邻居扩展片段，便于排障/评估。
                "graph_recalled": bool(
                    (getattr(chunk, "metadata", {}) or {}).get("graph_recalled")
                ),
                "neighbor_of": (getattr(chunk, "metadata", {}) or {}).get(
                    "neighbor_of"
                ),
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
    retriever = get_tool_service(
        context.services, ToolServiceKey.RAG_RETRIEVER
    )
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
    # 单回合检索硬预算：提示词约束之外再兜底，防止模型在一批里并行发大量
    # 子查询导致 token/耗时失控。计数放在 run 级 metadata，由主循环注入。
    counts = context.metadata.get("tool_call_counts")
    if isinstance(counts, dict):
        used = int(counts.get("search_kb", 0))
        if used >= policy.max_calls_per_run:
            return ToolResult.failure(
                call_id=context.call_id,
                tool_name=context.tool_name,
                error=ToolError(
                    code=ToolErrorCode.RATE_LIMITED,
                    message=(
                        f"本次回答的检索次数已达上限（{policy.max_calls_per_run} 次），"
                        "请基于已返回的片段作答；如仍缺少关键信息，请明确说明缺口，"
                        "不要再调用检索工具。"
                    ),
                    retryable=False,
                    details={"used": used, "limit": policy.max_calls_per_run},
                ),
            )
        counts["search_kb"] = used + 1
    # 未显式指定时使用运维配置的默认值，并统一收紧到允许上限。
    top_k = arguments.top_k if arguments.top_k is not None else policy.default_top_k
    top_k = min(top_k, policy.max_top_k)
    # kb_ids 是可选参数：聊天请求没有“会话绑定知识库”，模型也不应看到/猜测
    # 内部 id。省略时按用户实际权限解析可读知识库，与检索流水线共用
    # knowledge_bases.required_permission_code。
    kb_ids = list(dict.fromkeys(arguments.kb_ids))
    if not kb_ids:
        catalog = get_tool_service(context.services, ToolServiceKey.KB_CATALOG)
        resolver = getattr(catalog, "list_readable", None)
        if not callable(resolver):
            return ToolResult.failure(
                call_id=context.call_id,
                tool_name=context.tool_name,
                error=ToolError(
                    code=ToolErrorCode.INVALID_ARGUMENTS,
                    message="未提供 kb_ids，且服务端未配置知识库目录",
                    retryable=False,
                ),
            )
        try:
            kb_ids = list(
                await resolver(
                    user_id=context.user_id,
                    permissions=context.permissions,
                    limit=policy.max_kb_ids,
                )
            )
        except Exception as exc:  # noqa: BLE001 - 权限目录失败按依赖错误返回
            logger.warning("search_kb 解析可读知识库失败: %s", exc, exc_info=True)
            return ToolResult.failure(
                call_id=context.call_id,
                tool_name=context.tool_name,
                error=ToolError(
                    code=ToolErrorCode.UPSTREAM_ERROR,
                    message="读取可访问知识库失败，请稍后重试",
                    retryable=True,
                    cause=str(exc),
                ),
            )
        if not kb_ids:
            return ToolResult.failure(
                call_id=context.call_id,
                tool_name=context.tool_name,
                error=ToolError(
                    code=ToolErrorCode.PERMISSION_DENIED,
                    message="当前没有可访问的知识库",
                    retryable=False,
                ),
            )
    kb_ids = kb_ids[: policy.max_kb_ids]
    # 查询规划由 Agent 主循环注入：用户原话、改写 query、召回意图。
    # 工具自身选择的 query 仍是主召回 query；规划字段只补充召回/rerank。
    plan = context.metadata.get("query_plan")
    original_query = getattr(plan, "original_query", None) or arguments.query
    rewritten_query = getattr(plan, "rewritten_query", None)
    # 中英文双语改写由 QueryPlan.retrieval_queries 统一给出；老 plan/无规划时为空。
    rewritten_queries = list(getattr(plan, "retrieval_queries", []) or [])
    rag_intent = (
        getattr(plan, "rag_intent_label", None) if plan is not None else None
    )
    entities = list(getattr(plan, "entities", []) or []) if plan is not None else []
    request = RetrievalRequest(
        query=arguments.query,
        original_query=original_query,
        rewritten_query=rewritten_query,
        rewritten_queries=rewritten_queries,
        rag_intent=rag_intent,
        entities=entities,
        kb_ids=kb_ids,
        user_id=context.user_id,
        permissions=context.permissions,
        top_k=top_k,
        trace_id=context.trace_id,
    )
    try:
        # RAG 检索事件由 Retriever 内部的 RagEventSink 统一发送，工具层
        # 不再重复 emit rag.retrieval.*；工具执行本身由 ToolPipeline 记录。
        result = await retriever.retrieve(request)
    except RagError as exc:
        # 权限/参数错误是可解释的业务终态，不能统一标成可重试的
        # UPSTREAM_ERROR，否则模型会反复重试同一个非法请求。
        if exc.error_code is RagErrorCode.PERMISSION_DENIED:
            return ToolResult.failure(
                call_id=context.call_id,
                tool_name=context.tool_name,
                error=ToolError(
                    code=ToolErrorCode.PERMISSION_DENIED,
                    message="无权访问指定知识库",
                    retryable=False,
                    details=dict(exc.details or {}),
                ),
            )
        if exc.error_code in {
            RagErrorCode.INVALID_REQUEST,
            RagErrorCode.INVALID_CONFIG,
        }:
            return ToolResult.failure(
                call_id=context.call_id,
                tool_name=context.tool_name,
                error=ToolError(
                    code=ToolErrorCode.INVALID_ARGUMENTS,
                    message=exc.message,
                    retryable=False,
                    details=dict(exc.details or {}),
                ),
            )
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


def build_search_kb_executor(config: SearchKbToolConfig | None = None):
    """构造 ``search_kb`` 执行器；Spec/策略由 tools.manifest 统一声明。"""
    policy = config or SearchKbToolConfig()

    async def _executor(
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """适配 ToolExecutor 协议：校验参数后调用业务执行器。"""
        parsed = SearchKbArgs.model_validate(arguments)
        # call_id/tool_name 由 ToolPipeline 在调用前写入 ToolContext，
        # 执行器不再复制或覆盖，避免工具名与注册声明漂移。
        return await search_kb_executor(parsed, context, config=policy)

    return _executor


class SearchKbFactory(SimpleToolFactory):
    """``search_kb`` 工厂：依赖 Retriever，执行器封装在实现模块内。"""

    argument_model = SearchKbArgs

    def create_executor(
        self,
        declaration: ToolDeclaration,
        ctx: ToolBuildContext,
    ):
        """按运行配置构造检索执行器。"""
        return build_search_kb_executor(ctx.tools_config.search_kb)
