"""人类可读的多行日志摘要。

关键流程（Agent 一次问答、一次检索、一次图抽取、一次入库）用标题 + 缩进
字段输出，避免把所有指标挤在一条超长 key=value 里。需要机器解析的完整数据
仍在事件/DB/trace 中，日志只保留排障最常用的字段。
"""

from __future__ import annotations

from collections.abc import Iterable

__all__ = ["summary_block"]


def summary_block(title: str, lines: Iterable[str]) -> str:
    """生成 ``[标题]`` + 缩进字段的多行日志正文；跳过空行。"""
    body = [f"  {line}" for line in lines if line]
    return "\n".join([f"[{title}]", *body])
