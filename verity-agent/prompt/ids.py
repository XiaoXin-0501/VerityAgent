"""Prompt id 常量。

业务代码只引用这里的枚举值，不直接写字符串，避免模板改名后漏改调用点；
manifest/模板文件仍使用字符串 id。
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["PromptId"]


class PromptId(StrEnum):
    """当前已注册的 prompt id（与 templates/manifest.json 一一对应）。"""

    # Agent：主循环基础 system prompt
    AGENT_SYSTEM = "agent.system"
    # Agent：查询规划（意图识别/query 改写/RAG 决策；规则只短路明确无需 RAG）
    AGENT_QUERY_PLAN_SYSTEM = "agent.query_plan.system"
    # Agent：主循环反思（检查方向/止损）
    AGENT_REFLECTION_SYSTEM = "agent.reflection.system"
    # RAG：文档描述精炼
    RAG_DESCRIPTION_SYSTEM = "rag.description.system"
    RAG_DESCRIPTION_USER = "rag.description.user"
    # RAG：垂直领域图抽取
    RAG_GRAPH_SYSTEM = "rag.graph.system"
    RAG_GRAPH_USER = "rag.graph.user"
    # RAG：表格/代码语义描述（用于低语义特殊块的向量化）
    RAG_TABLE_DESCRIPTION_SYSTEM = "rag.table.description.system"
    RAG_TABLE_DESCRIPTION_USER = "rag.table.description.user"
    RAG_CODE_DESCRIPTION_SYSTEM = "rag.code.description.system"
    RAG_CODE_DESCRIPTION_USER = "rag.code.description.user"
    # RAG：知识库片段冲突处理（事实冲突/观点冲突/上下文不足）
    RAG_CHUNK_CONFLICT_SYSTEM = "rag.chunk_conflict.system"
    # Memory：长期画像/事实抽取、会话压缩
    MEMORY_EXTRACTION_SYSTEM = "memory.extraction.system"
    MEMORY_COMPRESSION_SYSTEM = "memory.compression.system"
