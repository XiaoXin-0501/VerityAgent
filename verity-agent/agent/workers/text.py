"""Worker 共用的消息文本提取。"""

from __future__ import annotations

from store.pg.models.memory import UserMessage
from utils.serialization import parse_json_object
from utils.text import extract_content_text
from utils.tokens import estimate_tokens, truncate_to_tokens

__all__ = ["extract_content_text", "messages_text", "parse_json_object"]


def messages_text(items: list[UserMessage], max_tokens: int) -> str:
    """把队列消息抽成带消息 id 的文本并按 token 截断。

    消息 id 供抽取模型为每条原子事实填写 ``msg_id``；截断时可能丢掉尾部
    消息，模型只能引用仍然出现在输入中的 id。
    """
    lines: list[str] = []
    for item in items:
        text = extract_content_text(item.content)
        if text:
            lines.append(f"- [msg:{item.message_id}] {text}")
    value = "\n".join(lines)
    if max_tokens > 0 and estimate_tokens(value) > max_tokens:
        value = truncate_to_tokens(value, max_tokens)
    return value
