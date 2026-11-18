"""Provider Chat 响应文本提取辅助。"""

from __future__ import annotations

import base64
from typing import Any

from exception.provider import ProviderError, ProviderErrorCode

from .types import ModelImagePart, ModelTextPart

__all__ = ["response_text", "image_part_url"]


def response_text(response: Any) -> str:
    """从 ChatModelResponse 提取普通文本，忽略 thinking/tool parts。"""
    message = getattr(response, "message", None)
    parts = getattr(message, "content", None) or []
    return "".join(
        part.text
        for part in parts
        if isinstance(part, ModelTextPart)
    ).strip()


def image_part_url(part: ModelImagePart) -> str:
    """把图片 part 转成远程 URL 或 base64 data URI。"""
    if part.url:
        return part.url
    if part.data:
        mime = part.mime_type or "image/png"
        encoded = base64.b64encode(part.data).decode("ascii")
        return f"data:{mime};base64,{encoded}"
    raise ProviderError(
        ProviderErrorCode.INVALID_REQUEST,
        "图片 part 缺少 url/data",
    )
