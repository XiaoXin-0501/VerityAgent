"""Provider Chat 响应文本提取辅助。"""

from __future__ import annotations

from typing import Any

from .types import ModelTextPart

__all__ = ["response_text"]


def response_text(response: Any) -> str:
    """从 ChatModelResponse 提取普通文本，忽略 thinking/tool parts。"""
    message = getattr(response, "message", None)
    parts = getattr(message, "content", None) or []
    return "".join(
        part.text
        for part in parts
        if isinstance(part, ModelTextPart)
    ).strip()
