"""文本清洗与展示工具。

这里的函数不依赖任何业务模块，解析器、切片器、上下文组装等都可以复用：
编码探测、Markdown 行内标记清理、句子切分、智能拼接、表格渲染。
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .cjk import is_cjk_char

__all__ = [
    "normalize_whitespace",
    "normalize_query",
    "strip_control_chars",
    "truncate",
    "ensure_prefix",
    "mask_secret",
    "decode_text",
    "detect_language",
    "smart_join",
    "split_sentences",
    "strip_markdown_inline",
    "table_to_markdown",
    "join_wrapped_lines",
    "filename_to_title",
    "is_meaningful_filename",
    "extract_content_text",
]

_WHITESPACE = re.compile(r"[ \t ]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def normalize_whitespace(text: str, *, keep_newlines: bool = True) -> str:
    """压缩水平空白；``keep_newlines`` 时保留段落，否则压成单行。"""
    value = _CONTROL.sub("", text.replace("\r\n", "\n").replace("\r", "\n"))
    value = _WHITESPACE.sub(" ", value)
    if keep_newlines:
        value = _MULTI_NEWLINE.sub("\n\n", value)
    else:
        value = value.replace("\n", " ")
    return value.strip()


def normalize_query(query: str) -> str:
    """查询文本归一：NFKC 全半角归一，水平空白压成一个空格。

    只做检索无关的通用清洗；实体大小写/原始写法保留，归一后不改变语义。
    """
    value = unicodedata.normalize("NFKC", query or "")
    return normalize_whitespace(value, keep_newlines=False)


def strip_control_chars(text: str) -> str:
    """移除 ASCII 控制字符，保留换行/制表以外的可见文本。"""
    return _CONTROL.sub("", text)


def extract_content_text(value: Any) -> str:
    """递归提取消息 content 中的文本片段；忽略二进制/结构元数据。

    消息 content 是版本化 JSON（v2），文本可能位于 ``text`` 字段或嵌套的
    parts 数组；本函数只做通用文本收集，不解释业务结构。
    """
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


def truncate(text: str, max_chars: int, *, suffix: str = "…") -> str:
    """按字符数截断并追加后缀；``max_chars`` 为负时抛 ``ValueError``。"""
    if max_chars < 0:
        raise ValueError("max_chars 不能为负数")
    if len(text) <= max_chars:
        return text
    if max_chars <= len(suffix):
        return suffix[:max_chars]
    return text[: max_chars - len(suffix)].rstrip() + suffix


def ensure_prefix(value: str, prefix: str) -> str:
    """确保字符串以 ``prefix`` 开头。"""
    return value if value.startswith(prefix) else f"{prefix}{value}"


def mask_secret(value: str | None, *, visible: int = 4) -> str:
    """日志/错误里脱敏密钥；短值整体打码。"""
    if not value:
        return ""
    if len(value) <= visible * 2:
        return "*" * len(value)
    return f"{value[:visible]}{'*' * (len(value) - visible * 2)}{value[-visible:]}"


# ——————————————————— 编码与语言 ———————————————————

# 按“中文环境优先”的顺序尝试；latin-1 永不失败，放最后兜底。
_DEFAULT_ENCODINGS: tuple[str, ...] = (
    "utf-8-sig",
    "utf-8",
    "gb18030",
    "big5",
    "latin-1",
)

# 判断语言时使用的 CJK 区间由 utils.cjk 统一维护；全角标点也视为 CJK，
# 用于中文拼接时不插空格。

# 收尾标点：拼接时标点前不留空格。
_CLOSING_PUNCT = set("。，、；：！？）】」』》〉”’,.;:!?)]}\"'")

# 开头标点：拼接时标点后不留空格。
_OPENING_PUNCT = set("（【「『《〈“‘([{\"'")


def _is_cjk(char: str) -> bool:
    """判断字符是否属于 CJK 文字或全角标点。"""
    return is_cjk_char(char, include_fullwidth=True)


def decode_text(
    data: bytes,
    *,
    encodings: tuple[str, ...] = _DEFAULT_ENCODINGS,
) -> str:
    """按候选编码依次解码字节；latin-1 永不失败，作为最后兜底。

    parser 不应假设上传文件一定是 UTF-8；中文 TXT/Markdown 常见 GBK/GB18030。
    因此 latin-1 的“兜底”优先保证不抛异常，可能产生 mojibake；需要严格
    诊断时应由调用方显式指定 ``encodings``。
    """
    for encoding in encodings:
        try:
            return data.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("utf-8", errors="replace")


def detect_language(text: str) -> str | None:
    """根据字符占比粗略判断主语言；只返回 ``zh`` / ``en`` / ``None``。

    该结果只用于 ``DocumentProfile.languages`` 和检索过滤，不替代专业语言检测。
    """
    if not text:
        return None
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return None
    cjk = sum(1 for char in letters if _is_cjk(char))
    if cjk / len(letters) >= 0.3:
        return "zh"
    latin = sum(1 for char in letters if char.isascii())
    if latin / len(letters) >= 0.6:
        return "en"
    return None


# ——————————————————— 拼接与切句 ———————————————————

def _needs_space(left: str, right: str) -> bool:
    """判断两段文本拼接时是否需要一个空格。

    - 中文/全角标点之间：不加空格；
    - 英文单词之间：加空格；
    - 收尾/开头标点附近：不加空格。
    """
    if not left or not right:
        return False
    if right[0] in _CLOSING_PUNCT or left[-1] in _OPENING_PUNCT:
        return False
    if _is_cjk(left[-1]) or _is_cjk(right[0]):
        return False
    return True


def smart_join(parts: list[str]) -> str:
    """拼接文本片段；英文加空格，中文不加，避免中文句子被空格切碎。"""
    result = ""
    for part in parts:
        value = part.strip()
        if not value:
            continue
        if not result:
            result = value
        elif _needs_space(result[-1], value[0]):
            result = f"{result} {value}"
        else:
            result = f"{result}{value}"
    return result


# 中文/英文句末标点后切分；英文句点要求后跟空白，避免把 3.14 切成两段。
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;])\s*|(?<=\.)\s+|\n+")


def split_sentences(text: str) -> list[str]:
    """把文本切成句子/行单元；保留句末标点，过滤空片段。

    中文句末标点后常跟右引号/右括号，``re.split`` 会把引号切到下一段；
    这里把“以收尾标点开头”的片段并回上一句，避免出现孤立的 ``”``。
    """
    raw = [
        item.strip()
        for item in _SENTENCE_SPLIT.split(text)
        if item and item.strip()
    ]
    units: list[str] = []
    for item in raw:
        if units and item[0] in _CLOSING_PUNCT:
            units[-1] = smart_join([units[-1], item])
        else:
            units.append(item)
    return units


def join_wrapped_lines(lines: list[str]) -> str:
    """合并 PDF/DOCX 的视觉换行。

    - 英文行尾连字符 + 下一行小写字母：去掉连字符直接拼接；
    - 其余情况按 ``smart_join`` 规则拼接（中文不插空格）。
    """
    result = ""
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if not result:
            result = line
            continue
        if result.endswith("-") and line[0].islower() and line[0].isascii():
            result = f"{result[:-1]}{line}"
        else:
            result = smart_join([result, line])
    return result


# ——————————————————— Markdown 行内清理 ———————————————————

# 顺序很重要：图片先于链接，行内代码先于强调，避免把代码内容当标记删掉。
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_IMAGE_REF = re.compile(r"!\[[^\]]*\]\[[^\]]*\]")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_MD_LINK_REF = re.compile(r"\[([^\]]+)\]\[[^\]]*\]")
_MD_AUTOLINK = re.compile(r"<((?:https?|mailto):[^>]+)>")
_MD_HTML_TAG = re.compile(r"</?[A-Za-z][^>]*>")
_MD_INLINE_CODE = re.compile(r"`+([^`]+)`+")
_MD_STRONG = re.compile(r"(\*\*|__)(.+?)\1")
_MD_EMPHASIS = re.compile(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])")
_MD_STRIKE = re.compile(r"~~(.+?)~~")
_MD_ESCAPE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|>])")


def strip_markdown_inline(text: str) -> str:
    """去掉行内 Markdown 标记，保留可读正文（图片整体移除）。"""
    value = _MD_IMAGE.sub("", text)
    value = _MD_IMAGE_REF.sub("", value)
    value = _MD_LINK.sub(r"\1", value)
    value = _MD_LINK_REF.sub(r"\1", value)
    value = _MD_AUTOLINK.sub(r"\1", value)
    value = _MD_HTML_TAG.sub("", value)
    value = _MD_INLINE_CODE.sub(r"\1", value)
    value = _MD_STRONG.sub(r"\2", value)
    value = _MD_STRIKE.sub(r"\1", value)
    value = _MD_EMPHASIS.sub(r"\1", value)
    value = _MD_ESCAPE.sub(r"\1", value)
    # 标记移除后可能留下连续空格（如“文本 ![图](x) 继续”）。
    value = re.sub(r"[ \t]+", " ", value)
    return value.strip()


# ——————————————————— 表格渲染 ———————————————————

def _cell_text(value: object) -> str:
    """把单元格值规范成单行文本，并转义 Markdown 表格分隔符。"""
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text.replace("|", r"\|")


def table_to_markdown(
    headers: list[str],
    rows: list[list[str]],
    *,
    max_columns: int | None = None,
) -> str:
    """把表头+行数据渲染成 Markdown 表格，并补齐/裁剪到统一列数。

    ``max_columns`` 可用于把宽表截断到前 N 列；默认按最宽一行对齐。
    """
    width = max([len(headers), *(len(row) for row in rows)], default=0)
    if max_columns is not None:
        width = min(width, max(0, max_columns))
    if width == 0:
        return ""

    def normalize(row: list[str]) -> list[str]:
        """把一行补齐/裁剪到统一列数。"""
        cells = [_cell_text(item) for item in row[:width]]
        return cells + [""] * (width - len(cells))

    header_cells = normalize(headers)
    has_header = any(header_cells)
    lines: list[str] = []
    if has_header:
        lines.append(" | ".join(header_cells))
        lines.append(" | ".join("---" for _ in range(width)))
    lines.extend(" | ".join(normalize(row)) for row in rows)
    return "\n".join(lines).strip()


# ——————————————————— 文件名可读性 ———————————————————

# UUID 形如 550e8400-e29b-41d4-a716-446655440000；文件名清洗后会变成 32 位十六进制。
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_HEX_RE = re.compile(r"^[0-9a-f]{16,}$", re.IGNORECASE)
_GENERIC_STEM_RE = re.compile(
    r"^(?:"
    r"document|doc|file|scan|scanned|untitled|unnamed|temp|tmp|new|copy|"
    r"output|download|附件|文档|文件|扫描件|未命名|新建文档|副本|临时文件|"
    r"输出|下载"
    r")[-_ ]?\d*$",
    re.IGNORECASE,
)
_NOISE_SUFFIX_RE = re.compile(
    r"\s+(?:v\d+(?:\.\d+)*|final|copy|副本|最终版)$",
    re.IGNORECASE,
)


def filename_to_title(filename: str) -> str:
    """把文件名转成可读标题：去扩展名、分隔符、版本/副本后缀。"""
    stem = Path(filename).stem
    value = re.sub(r"[_\-.]+", " ", stem)
    value = re.sub(r"\s+", " ", value).strip()
    value = _NOISE_SUFFIX_RE.sub("", value).strip()
    return value


def is_meaningful_filename(filename: str) -> bool:
    """判断文件名是否适合作为描述/标题，过滤 UUID、哈希、乱码和通用名。

    只有“看起来像人写的”文件名才允许进入 heading 或文档描述：
    - 至少 2 个 CJK 字符，或至少 4 个拉丁字母且元音比例正常；
    - 拒绝 UUID、纯十六进制、替换字符/控制字符、通用名（document/scan/未命名等）。
    """
    title = filename_to_title(filename)
    if not title or len(title) < 2:
        return False
    compact = re.sub(r"\s+", "", title)
    if _UUID_RE.match(compact) or _HEX_RE.match(compact):
        return False
    if "�" in title or any(ord(char) < 32 for char in title):
        return False
    if _GENERIC_STEM_RE.match(compact):
        return False

    letters = [char for char in title if char.isalpha()]
    if not letters:
        return False
    cjk_count = sum(1 for char in letters if _is_cjk(char))
    if cjk_count >= 2:
        return True
    latin = [char for char in letters if char.isascii()]
    if len(latin) < 4:
        return False
    vowels = sum(1 for char in latin if char.lower() in "aeiou")
    return vowels / len(latin) >= 0.12
