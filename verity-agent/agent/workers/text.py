"""Worker 共用的消息文本提取。"""

from __future__ import annotations

import json
from typing import Any, Mapping

from store.pg.models.memory import UserMessage
from utils.tokens import estimate_tokens, truncate_to_tokens

__all__ = ["extract_content_text", "messages_text", "parse_json_object"]


def extract_content_text(value: Any) -> str:
    """递归提取 content 中的文本片段；忽略二进制/结构元数据。"""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        if isinstance(value.get("text"), str):
            return value["text"].strip()
        parts: list[str] = []
        for item in value.values():
            text = extract_content_text(item)
            if text:
                parts.append(text)
        return "\n".join(parts)
    if isinstance(value, (list, tuple)):
        return "\n".join(
            text for item in value if (text := extract_content_text(item))
        )
    return ""


def messages_text(items: list[UserMessage], max_tokens: int) -> str:
    """把队列消息 content 抽成文本并按 token 截断。"""
    lines: list[str] = []
    for item in items:
        text = extract_content_text(item.content)
        if text:
            lines.append(f"- {text}")
    value = "\n".join(lines)
    if max_tokens > 0 and estimate_tokens(value) > max_tokens:
        value = truncate_to_tokens(value, max_tokens)
    return value


def parse_json_object(text: str) -> dict[str, Any]:
    """解析模型 JSON；容忍 ```json 围栏与前后说明文本。"""
    value = text.strip()
    if value.startswith("```"):
        value = value.strip("`")
        if value.startswith("json"):
            value = value[4:]
    start = value.find("{")
    end = value.rfind("}")
    if start >= 0 and end > start:
        value = value[start : end + 1]
    data = json.loads(value)
    if not isinstance(data, dict):
        raise ValueError("模型输出不是 JSON object")
    return data
