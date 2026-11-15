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
    # RAG：文档描述精炼
    RAG_DESCRIPTION_SYSTEM = "rag.description.system"
    RAG_DESCRIPTION_USER = "rag.description.user"
    # RAG：查询改写与意图分类
    RAG_QUERY_REWRITE_SYSTEM = "rag.query_rewrite.system"
    RAG_INTENT_SYSTEM = "rag.intent.system"
    # RAG：垂直领域图抽取
    RAG_GRAPH_SYSTEM = "rag.graph.system"
    RAG_GRAPH_USER = "rag.graph.user"
    # RAG：表格/代码语义描述（用于低语义特殊块的向量化）
    RAG_TABLE_DESCRIPTION_SYSTEM = "rag.table.description.system"
    RAG_TABLE_DESCRIPTION_USER = "rag.table.description.user"
    RAG_CODE_DESCRIPTION_SYSTEM = "rag.code.description.system"
    RAG_CODE_DESCRIPTION_USER = "rag.code.description.user"
    # Memory：长期画像/事实抽取、会话压缩、事实冲突判定
    MEMORY_EXTRACTION_SYSTEM = "memory.extraction.system"
    MEMORY_COMPRESSION_SYSTEM = "memory.compression.system"
    MEMORY_FACT_CONFLICT_SYSTEM = "memory.fact_conflict.system"
