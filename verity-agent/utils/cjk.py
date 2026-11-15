"""CJK 字符范围判断。

`utils.text` 与 `utils.tokens` 都需要判断中文/日文/韩文字符；把范围与
判断逻辑放在这里，避免两套常量随版本漂移。全角标点是否算 CJK 由调用方决定
（文本拼接需要，token 估算不需要）。
"""

from __future__ import annotations

__all__ = ["CJK_RANGES", "FULLWIDTH_PUNCT_RANGES", "is_cjk_char"]

CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x3400, 0x4DBF),   # CJK Extension A
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0x3040, 0x30FF),   # Hiragana / Katakana
    (0xAC00, 0xD7AF),   # Hangul Syllables
    (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
)

FULLWIDTH_PUNCT_RANGES: tuple[tuple[int, int], ...] = (
    (0x3000, 0x303F),   # CJK Symbols and Punctuation
    (0xFF00, 0xFFEF),   # Halfwidth and Fullwidth Forms
)

_ALL_RANGES = CJK_RANGES + FULLWIDTH_PUNCT_RANGES


def is_cjk_char(char: str, *, include_fullwidth: bool = False) -> bool:
    """判断单个字符是否属于 CJK 文字（可选包含全角标点）。"""
    if len(char) != 1:
        return False
    code = ord(char)
    ranges = _ALL_RANGES if include_fullwidth else CJK_RANGES
    return any(start <= code <= end for start, end in ranges)
