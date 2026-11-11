"""token 估算。

v1 不引入 tiktoken / 厂商 tokenizer 依赖，先用保守的字符启发式：

- 中文/日文/韩文等 CJK 字符按 1 字符 ≈ 1 token；
- 其余字符按 4 字符 ≈ 1 token；
- 空白与标点计入字符数。

该估算只用于预算预检与截断，最终用量以 provider 返回的 ``usage`` 为准。
需要精确计数时，由 Provider 的 ``count_tokens`` 覆盖。
"""

from __future__ import annotations

__all__ = ["estimate_tokens", "fits_budget", "truncate_to_tokens", "split_by_tokens"]

_CJK_RANGES = (
    (0x4E00, 0x9FFF),    # CJK Unified Ideographs
    (0x3400, 0x4DBF),    # CJK Extension A
    (0x3040, 0x30FF),    # Hiragana / Katakana
    (0xAC00, 0xD7AF),    # Hangul Syllables
    (0xF900, 0xFAFF),    # CJK Compatibility Ideographs
)


def _is_cjk(char: str) -> bool:
    code = ord(char)
    return any(start <= code <= end for start, end in _CJK_RANGES)


def estimate_tokens(text: str) -> int:
    """估算文本 token 数；空文本返回 0。"""
    if not text:
        return 0
    cjk = sum(1 for char in text if _is_cjk(char))
    other = len(text) - cjk
    return cjk + (other + 3) // 4


def fits_budget(text: str, max_tokens: int) -> bool:
    if max_tokens < 0:
        raise ValueError("max_tokens 不能为负数")
    return estimate_tokens(text) <= max_tokens


def truncate_to_tokens(text: str, max_tokens: int, *, suffix: str = "…") -> str:
    """按估算值从尾部截断；尽量不切断代理对。"""
    if max_tokens <= 0:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text

    budget = max_tokens - estimate_tokens(suffix)
    if budget <= 0:
        return suffix[:max_tokens]

    # 逐字符累计：CJK 记 1；非 CJK 每 4 字符记 1，但不足 4 字符的
    # 尾组也按 1 token 估算（与 estimate_tokens 的 ceil 口径一致）。
    used = 0
    end = 0
    pending_ascii = 0
    for index, char in enumerate(text):
        if _is_cjk(char):
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


def split_by_tokens(text: str, max_tokens: int) -> list[str]:
    """把文本切成每段不超过 ``max_tokens`` 的片段，不丢字符。

    用于切片器处理单个超长段落；优先调用 ``truncate_to_tokens``，保证
    与 token 估算使用同一套口径。
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens 必须 > 0")
    pieces: list[str] = []
    rest = text
    while rest:
        piece = truncate_to_tokens(rest, max_tokens, suffix="")
        if not piece:
            break
        pieces.append(piece)
        rest = rest[len(piece):]
    return pieces
