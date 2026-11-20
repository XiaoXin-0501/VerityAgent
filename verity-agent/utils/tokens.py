"""token 估算与截断。

优先使用 ``tiktoken`` 做精确估算；未安装或编码器不可用时，退回保守的
字符启发式（CJK 1 字符 ≈ 1 token，其他 4 字符 ≈ 1 token）。

编码器通过环境变量 ``LLM__TOKENIZER__ENCODING`` 选择，默认 ``o200k_base``
（对中英文混合文本比 cl100k_base 更贴近现代模型）。该估算仍是近似值；
最终用量以 provider 返回的 ``usage`` 为准。
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any

from .cjk import is_cjk_char

__all__ = [
    "configure_tokenizer",
    "estimate_tokens",
    "fits_budget",
    "split_by_tokens",
    "tokenizer_name",
    "truncate_to_tokens",
]

logger = logging.getLogger(__name__)

# 由 Settings 在启动时注入；None 时读 LLM__TOKENIZER__ENCODING/默认值。
_ENCODING_NAME: str | None = None


def configure_tokenizer(encoding: str | None) -> None:
    """设置全局 tokenizer 编码；清空缓存，下一次估算重新加载。"""
    global _ENCODING_NAME
    name = (encoding or "").strip() or None
    if name != _ENCODING_NAME:
        _ENCODING_NAME = name
        _encoding.cache_clear()


@lru_cache(maxsize=1)
def _encoding() -> Any | None:
    """懒加载 tiktoken 编码器；失败时返回 None 并退回启发式。"""
    try:
        import tiktoken
    except ImportError:
        return None
    name = (
        _ENCODING_NAME
        or os.getenv("LLM__TOKENIZER__ENCODING")
        or "o200k_base"
    ).strip()
    try:
        return tiktoken.get_encoding(name)
    except Exception as exc:  # noqa: BLE001 - 首次下载失败时降级
        logger.warning("tiktoken 编码器不可用，退回字符估算: %s", exc)
        return None


def tokenizer_name() -> str:
    """返回当前实际使用的 tokenizer 标识，供日志/启动摘要。"""
    encoding = _encoding()
    return getattr(encoding, "name", "heuristic")


def estimate_tokens(text: str) -> int:
    """估算文本 token 数；空文本返回 0。"""
    if not text:
        return 0
    encoding = _encoding()
    if encoding is not None:
        try:
            return len(encoding.encode(text, disallowed_special=()))
        except Exception:  # noqa: BLE001 - 编码异常退回启发式
            pass
    cjk = sum(1 for char in text if is_cjk_char(char))
    other = len(text) - cjk
    return cjk + (other + 3) // 4


def fits_budget(text: str, max_tokens: int) -> bool:
    """判断文本估算 token 是否不超过预算。"""
    if max_tokens < 0:
        raise ValueError("max_tokens 不能为负数")
    return estimate_tokens(text) <= max_tokens


def truncate_to_tokens(text: str, max_tokens: int, *, suffix: str = "…") -> str:
    """按 token 预算从尾部截断；尽量不切断代理对。"""
    if max_tokens <= 0:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text
    encoding = _encoding()
    if encoding is None:
        return _truncate_heuristic(text, max_tokens, suffix)
    try:
        suffix_tokens = len(encoding.encode(suffix, disallowed_special=()))
        budget = max_tokens - suffix_tokens
        if budget <= 0:
            return suffix[:max_tokens]
        end = _prefix_chars(encoding, text, budget)
        if end <= 0:
            return suffix[:max_tokens]
        return text[:end].rstrip() + suffix
    except Exception:  # noqa: BLE001 - 编码异常退回启发式
        return _truncate_heuristic(text, max_tokens, suffix)


def split_by_tokens(text: str, max_tokens: int) -> list[str]:
    """把文本切成每段不超过 ``max_tokens`` 的片段，不丢字符。

    优先用 tiktoken 找“最大可容纳字符前缀”，保证切片后拼接仍是原文。
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens 必须 > 0")
    if not text:
        return []
    encoding = _encoding()
    if encoding is None:
        return _split_heuristic(text, max_tokens)
    pieces: list[str] = []
    rest = text
    try:
        while rest:
            end = _prefix_chars(encoding, rest, max_tokens)
            if end <= 0:
                # 极端 token 边界（半个代理对等）：至少消费一个字符，避免死循环。
                end = 1
            piece = rest[:end]
            pieces.append(piece)
            rest = rest[end:]
    except Exception:  # noqa: BLE001 - 编码异常退回启发式
        return _split_heuristic(text, max_tokens)
    return pieces


def _prefix_chars(encoding: Any, text: str, max_tokens: int) -> int:
    """二分查找最长字符前缀，使其 token 数不超过预算。"""
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        # 不在代理对中间截断（text[middle] 是低代理说明前一个字符是配对高代理）。
        if 0 < middle < len(text) and 0xDC00 <= ord(text[middle]) <= 0xDFFF:
            middle -= 1
            if middle <= low:
                break
        tokens = encoding.encode(text[:middle], disallowed_special=())
        if len(tokens) <= max_tokens:
            low = middle
        else:
            high = middle - 1
    return low


def _truncate_heuristic(text: str, max_tokens: int, suffix: str) -> str:
    """字符启发式截断（tiktoken 不可用时的兜底）。"""
    budget = max_tokens - _heuristic_tokens(suffix)
    if budget <= 0:
        return suffix[:max_tokens]
    used = 0
    end = 0
    pending_ascii = 0
    for index, char in enumerate(text):
        if is_cjk_char(char):
            if pending_ascii:
                pending_ascii = 0
            used += 1
        else:
            pending_ascii += 1
            if pending_ascii == 1:
                used += 1
            elif pending_ascii == 4:
                pending_ascii = 0
        if used > budget:
            break
        end = index + 1
    return text[:end].rstrip() + suffix


def _split_heuristic(text: str, max_tokens: int) -> list[str]:
    """字符启发式切片（tiktoken 不可用时的兜底）。"""
    pieces: list[str] = []
    rest = text
    while rest:
        piece = _truncate_heuristic(rest, max_tokens, suffix="")
        if not piece:
            break
        pieces.append(piece)
        rest = rest[len(piece):]
    return pieces


def _heuristic_tokens(text: str) -> int:
    """启发式 token 数（不依赖 tiktoken）。"""
    cjk = sum(1 for char in text if is_cjk_char(char))
    other = len(text) - cjk
    return cjk + (other + 3) // 4
