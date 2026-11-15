"""Context 组装的数据模型。

v1 只实现“可用的预算/裁剪计划”，不包含完整 MemoryView/ContextSnapshot；
后续扩展字段时保持 ``ContextPlan`` 是 prompt 组装的唯一输出，避免 Agent
循环直接拼接 system/history。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent.llm import LLMMessage

__all__ = ["ContextSection", "ContextPlan"]


class _Model(BaseModel):
    """内部模型基类：拒绝未知字段，避免拼写错误静默生效。"""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class ContextSection(_Model):
    """一段可排序、可裁剪的上下文来源。"""

    # 来源：base / memory / rag / tool / summary
    source: str
    # 展示名；用于日志和 Prompt 边界标签
    name: str
    content: str
    # 数值越大越优先保留
    priority: int = 100
    # 预估 token；为 0 时由 assembler 计算
    token_estimate: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextPlan(_Model):
    """一次上下文组装的最终结果。"""

    messages: list[LLMMessage] = Field(default_factory=list)
    sections: list[ContextSection] = Field(default_factory=list)
    token_estimate: int = 0
    dropped_sections: list[str] = Field(default_factory=list)
    truncated: bool = False
