"""JSONC（带注释 JSON）解析。

标准 JSON 不支持注释；配置目录示例使用 ``.jsonc``，这里提供一个只处理
``//`` 与 ``/* ... */`` 注释的轻量解析器，字符串内的 URL（``https://``）
不会被误删。
"""

from __future__ import annotations

import json

__all__ = ["loads"]


def loads(text: str):
    """解析带 // 与 /* */ 注释的 JSON；字符串内内容不受影响。"""
    out: list[str] = []
    index = 0
    length = len(text)
    in_string = False
    escaped = False
    while index < length:
        char = text[index]
        if in_string:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            out.append(char)
            index += 1
            continue
        if char == "/" and index + 1 < length:
            next_char = text[index + 1]
            if next_char == "/":
                index += 2
                while index < length and text[index] not in "\r\n":
                    index += 1
                continue
            if next_char == "*":
                index += 2
                while index + 1 < length and not (
                    text[index] == "*" and text[index + 1] == "/"
                ):
                    index += 1
                index += 2
                continue
        out.append(char)
        index += 1
    return json.loads("".join(out))
