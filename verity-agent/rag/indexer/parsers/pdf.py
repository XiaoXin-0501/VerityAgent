"""PDF parser（PyMuPDF 主，pdfplumber fallback）。

v1 能力：
- 读取文本层，按页面文本块（而不是整页一段）输出，保留页码和阅读顺序；
- 依据字号/编号识别标题，写入 ``heading_level`` / ``heading_text``，
  让后续 chunk 获得标题上下文；支持 ``1.诊断`` / ``1.1 病例信息`` 等编号标题；
- 双栏页面按“跨栏块锚点 + 左栏先于右栏”还原阅读顺序，避免左右栏逐行交错；
- 识别跨页重复的水印/页眉页脚短行并在解析阶段丢弃（每行都会记录告警）；
- 只对有足够横竖线的页面做表格检测（PyMuPDF ``find_tables``）；复杂矢量页
  会改用 pdfplumber 抽表，避免 PyMuPDF 在超大图元页面上崩溃；表格输出为
  ``TableBlock``；
- 图片页先尝试简单 OCR；整篇均为图片/扫描页时交给 ``image_pdf`` 版式解析
  流水线（MinerU/RapidOCR）；都不进入正文时只记录 ``has_images``；

PDF 解析是同步阻塞操作，通过 ``asyncio.to_thread`` 隔离，避免卡住事件循环。
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, cast
from uuid import UUID

from config.rag import ImagePdfConfig, OcrConfig
from exception import RagError, RagErrorCode
from rag.shared.enums import BlockType, DocumentType
from rag.shared.models import (
    ContentBlock,
    DocumentProfile,
    DocumentSource,
    Page,
    ParsedDocument,
)

from ._common import (
    HeadingStack,
    heading_level_from_text,
    is_heading_text,
    make_block,
    make_page,
    make_parsed_document,
    make_table,
)
from .image_pdf.pipeline import ImagePdfPipeline
from .ocr_provider import OcrProvider, create_ocr_provider
from utils.text import join_wrapped_lines

__all__ = ["PdfParser"]

logger = logging.getLogger(__name__)

# 纯页码行：12 / 12/145 / - 12 - / 第 12 页。
_PAGE_NUMBER = re.compile(
    r"^\s*(?:[-—–]\s*)?(?:第\s*)?\d{1,4}\s*(?:/\s*\d{1,4})?\s*(?:页)?\s*(?:[-—–])?\s*$"
)
# 粗体 flag（PyMuPDF span.flags bit 4）。
_BOLD_FLAG = 1 << 4

# 句末/分句标点：出现这些标点的块是正文，不参与字体/粗体标题推断。
# 问号保留：FAQ/面试题标题本身可能就是问句。
_SENTENCE_PUNCT = re.compile(r"[。！!；;]")

# 页眉/页脚模板：数字通常随页码变化，归一成 # 后再统计重复率。
_DIGITS_TEMPLATE = re.compile(r"\d+")
# 目录行：点线/省略号引导 + 页码。目录页对 RAG 没有价值，且会大量命中
# 文档标题/章节名，必须在解析阶段剔除。
_TOC_LINE = re.compile(r"(?:\.{4,}|…{3,}|·{3,}|。{4,})\s*\d{1,4}\s*$")


def _is_toc_line(text: str) -> bool:
    """判断一行是否是目录项（点线引导 + 页码，或 "< 标题 页码" 连续项）。"""
    value = text or ""
    if _TOC_LINE.search(value):
        return True
    return bool(re.search(r"(?:<[^<>]{1,60}\s+\d{1,3}\s*){2,}", value))


def _toc_pages(payloads: list[_PdfPage]) -> set[int]:
    """识别以目录为主的页面；至少 4 行目录项且占文本行 40% 以上。"""
    pages: set[int] = set()
    for payload in payloads:
        lines = [
            line.text
            for block in payload.blocks
            for line in block.lines
            if line.text.strip()
        ]
        if len(lines) < 4:
            continue
        toc_lines = sum(1 for line in lines if _is_toc_line(line))
        if toc_lines >= 4 and toc_lines / len(lines) >= 0.4:
            pages.add(payload.page_no)
    return pages


def _boilerplate_key(text: str) -> str:
    """把页眉/页脚中的数字替换为占位符，用于识别“管理指南93/161”这类模板。"""
    return _DIGITS_TEMPLATE.sub("#", text or "")

# PDF 常见的分级编号标题：
#   1.诊断 / 4.讨论      → 一级小节（挂在文档标题下）
#   1.1 病例信息         → 二级小节
#   1.3.1 血常规检查     → 三级小节
# 要求编号后紧跟中文/英文，避免把 “1. 列表项” 或表格数值误判成标题。
_NUMBERED_COMPOUND_HEADING = re.compile(
    # 编号后必须是空白、汉字或大写英文；避免把 "28.1mg/L" 化验值当标题。
    r"^\s*\d{1,2}(?:\.\d{1,2})+(?:\s+|[一-鿿]|[A-Z])"
)
_NUMBERED_SINGLE_HEADING = re.compile(
    r"^\s*\d{1,2}\.(?=\s*[一-鿿A-Za-z])\s*"
)
# 期刊常见 “1 | Introduction”“2.1 | Histopathology” 编号格式。
_NUMBERED_PIPE_HEADING = re.compile(
    r"^\s*\d{1,2}(?:\.\d{1,2})*\s*\|\s*[A-Za-z一-鿿]"
)
# 标准/技术规范常用 “1 范围”“7.1 样品采集” 的空格编号，而不是 “1.”。
_NUMBERED_SPACE_HEADING = re.compile(
    r"^\s*\d{1,2}(?:\.\d{1,2})*\s+([一-鿿A-Za-z].*)$"
)
# 空格编号后若是量词/单位开头，通常是编号列表而不是章节标题。
_NUMBERED_LIST_PREFIX = frozenset(
    "只个次种类例名头条张份组项天年月日周岁倍号点位页行"
)
# 摘要标签常与正文同在一个文本块：ABSTRACT Feline megaesophagus...
# 英文标签要求词边界，避免把 “ABSTRACTION...” 误判成摘要。
_INLINE_ABSTRACT_EN = re.compile(
    r"^\s*(?P<label>ABSTRACT|SUMMARY)\b\s*[:：]?\s*(?P<body>.+)$",
    re.IGNORECASE | re.DOTALL,
)
_INLINE_ABSTRACT_CN = re.compile(
    r"^\s*(?P<label>摘要|概要)\s*[:：]?\s*(?P<body>.+)$",
    re.DOTALL,
)

# 学术 PDF 常见的独立小节标题；这些标题经常比正文小、但加粗，
# 字号规则识别不到，用白名单兜底（只接受短行，避免正文误判）。
_SECTION_LABEL_HEADINGS = frozenset(
    {
        "abstract", "summary", "introduction", "background", "objectives",
        "objective", "methods", "materials and methods", "results",
        "case presentation", "case description", "discussion", "conclusion",
        "conclusions", "references", "acknowledgment", "acknowledgments",
        "acknowledgement", "acknowledgements", "funding", "ethics statement",
        "consent", "data availability statement", "conflict of interest",
        "author contributions", "supplementary materials", "abbreviations",
        "摘要", "引言", "前言", "方法", "材料与方法", "结果", "病例介绍",
        "病例报告", "讨论", "结论", "参考文献", "致谢", "基金项目",
        "伦理声明", "知情同意", "利益冲突", "作者贡献", "数据可用性声明",
    }
)

# 版面阅读顺序：宽度超过页面 60% 的块视为“跨栏块”，作为上下分区的锚点；
# 左右栏之间的最小空白用于判定双栏。
_WIDE_BLOCK_RATIO = 0.6
_COLUMN_SPLIT_MIN_GAP = 10.0

# 页面边角的出版社页脚/版权/下载水印；只在页边区域命中，不影响正文里
# 偶尔出现的 “copyright” 等词。
_PAGE_FURNITURE = re.compile(
    r"(this is an open access article|creative commons attribution license|"
    r"copyright\s*©|downloaded from|wiley online library|terms and conditions|"
    r"oa articles are governed|for personal use only|"
    r"provided the original work is properly cited|"
    r"^case reports in veterinary medicine,?\s*\d{4}|"
    r"^case reports in veterinary medicine$|"
    r"^https?://doi\.org/|"
    r"^\d+\s+of\s+\d+$)",
    re.IGNORECASE,
)

# PyMuPDF find_tables 在“线条极多”的复杂页面上可能崩溃（进程级 segfault），
# 超过该绘制对象数时改用 pdfplumber 抽表，避免整篇文档解析失败。
_PYMUPDF_TABLE_MAX_DRAWINGS = 150


@dataclass(slots=True)
class _PdfLine:
    """PDF 中一行文本及其排版信息。"""

    text: str
    bbox: tuple[float, float, float, float]
    size: float
    bold: bool
    # 主要字体名（按字符数加权）；用于区分书宋正文与黑体标题。
    font: str = ""
    # OCR 生成的文本没有可靠字号/位置，避免被页眉页脚规则误删。
    ocr: bool = False


@dataclass(slots=True)
class _PdfTextBlock:
    """PDF 文本块（通常对应一个段落或列表项）。"""

    bbox: tuple[float, float, float, float]
    lines: list[_PdfLine] = field(default_factory=list)


@dataclass(slots=True)
class _PdfTable:
    """PDF 表格；rows 为含表头的二维文本。"""

    bbox: tuple[float, float, float, float]
    rows: list[list[str]] = field(default_factory=list)


@dataclass(slots=True)
class _PdfPage:
    """一页 PDF 的解析中间结果。"""

    page_no: int
    width: float
    height: float
    rotation: int = 0
    blocks: list[_PdfTextBlock] = field(default_factory=list)
    tables: list[_PdfTable] = field(default_factory=list)
    image_count: int = 0
    size_chars: Counter = field(default_factory=Counter)
    # 本页正文主字号；排除表格文本后统计，供本页标题判断使用。
    body_size: float | None = None
    # 本页通过 OCR 获得文本，以及整体置信度。
    ocr_used: bool = False
    ocr_confidence: float | None = None
    # 本页过滤掉的作者/单位元数据、图片说明块数，供解析告警统计。
    dropped_metadata: int = 0
    dropped_captions: int = 0


def _bbox_tuple(value) -> tuple[float, float, float, float]:
    """把 PyMuPDF 的 Rect/序列统一成四元组。"""
    x0, y0, x1, y1 = value
    return (float(x0), float(y0), float(x1), float(y1))


def _overlap_ratio(
    inner: tuple[float, float, float, float],
    outer: tuple[float, float, float, float],
) -> float:
    """计算 ``inner`` 被 ``outer`` 覆盖的面积比例。"""
    x0 = max(inner[0], outer[0])
    y0 = max(inner[1], outer[1])
    x1 = min(inner[2], outer[2])
    y1 = min(inner[3], outer[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inner_area = max(0.0, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    if inner_area <= 0:
        return 0.0
    return ((x1 - x0) * (y1 - y0)) / inner_area


def _bbox_matches(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
    tolerance: float = 1.5,
) -> bool:
    """两个 bbox 是否基本重合（PDF 导出的浮点坐标可能略有差异）。"""
    return all(abs(left[index] - right[index]) <= tolerance for index in range(4))


def _same_row(left: "_PdfLine", right: "_PdfLine") -> bool:
    """两段文本是否在同一视觉行（用于编号/标题/正文同行的拆分）。"""
    top = max(left.bbox[1], right.bbox[1])
    bottom = min(left.bbox[3], right.bbox[3])
    if bottom >= top:
        return True
    return abs(left.bbox[1] - right.bbox[1]) <= max(left.size, right.size) * 0.8


def _normalize_line(text: str) -> str:
    """用于重复行统计的归一化：去掉所有空白。"""
    return re.sub(r"\s+", "", text)


def _section_label_level(text: str) -> int | None:
    """识别 ``References / Funding / Ethics Statement`` 等独立小节标题。

    这些标题经常是 8pt 加粗（比 9pt 正文还小），字号规则不命中。
    只接受白名单短行，正文中出现同样的词不会被当成标题。
    """
    value = re.sub(r"\s+", " ", (text or "").strip()).rstrip(":：").lower()
    if not value or len(value) > 60:
        return None
    compact = re.sub(r"\s+", "", value)
    if value in _SECTION_LABEL_HEADINGS or compact in _SECTION_LABEL_HEADINGS:
        # 全小写英文标签（如表格里的 background/abbreviations）不可靠。
        raw = (text or "").strip().rstrip(":：")
        letters = [char for char in raw if char.isalpha()]
        if letters and all(char.isascii() for char in letters) and raw.islower():
            return None
        # 文档标题为一级，独立小节标题统一挂在二级。
        return 2
    return None


def _split_inline_abstract(text: str) -> tuple[str, str] | None:
    """拆分 ``ABSTRACT 正文...`` 这类“标签 + 正文同块”的摘要。

    期刊 PDF 常把 ``ABSTRACT`` 和摘要正文放在同一个文本块里；如果整块按
    普通段落处理，摘要标题就丢了，正文也无法作为独立章节切片。
    只识别摘要类标签，避免把普通段落开头的词误当标题。
    """
    match = _INLINE_ABSTRACT_EN.match(text or "") or _INLINE_ABSTRACT_CN.match(
        text or ""
    )
    if not match:
        return None
    label = match.group("label").strip()
    body = match.group("body").strip()
    if not body:
        return None
    return label, body


# —— 字体族、作者元数据与图片说明识别 ———————————————————
# 中文 PDF 常以字体族而不是 bold flag 表达层级：
#   FZSS/SimSun/FZKT = 书宋/宋体/楷体（正文、摘要、图注）
#   FZLT/FZHT/Hei/Heavy/Black = 兰亭黑/黑体（标题、作者名）
# PyMuPDF 的 flags 对中文字体经常没有 bold 位，必须结合字体名判断。
_HEADING_FONT_MARKERS = (
    "fzlt", "fzht", "hei", "heavy", "bold", "black", "yahei", "黑",
)
_BODY_FONT_MARKERS = (
    "fzss", "simsun", "song", "fzkt", "kaiti", "times", "georgia", "cambria",
)


def _font_is_heading(font: str) -> bool:
    """字体名是否属于“黑体/标题”族。"""
    value = (font or "").lower()
    return any(marker in value for marker in _HEADING_FONT_MARKERS)


def _font_is_body(font: str) -> bool:
    """字体名是否属于“宋体/楷体”正文字体族。"""
    value = (font or "").lower()
    return any(marker in value for marker in _BODY_FONT_MARKERS)


def _looks_letter_spaced(text: str) -> bool:
    """识别被 PDF 拆成单字母的标题/页眉，如 "P R A C T I C E"。"""
    value = (text or "").strip()
    if not value:
        return False
    tokens = [token for token in value.split() if token]
    if len(tokens) >= 4:
        singles = sum(1 for token in tokens if len(token) == 1)
        if singles / len(tokens) >= 0.6:
            return True
    compact = re.sub(r"\s+", "", value)
    return len(value) >= 8 and len(compact) <= len(value) * 0.55


def _line_is_bold(spans: list[dict[str, Any]], font: str) -> bool:
    """综合 bold flag 与字体族判断一行是否加粗/黑体。"""
    if not spans:
        return False
    if _font_is_heading(font):
        return True
    total = sum(len(span.get("text", "")) for span in spans)
    bold_chars = sum(
        len(span.get("text", ""))
        for span in spans
        if int(span.get("flags", 0)) & _BOLD_FLAG
    )
    return total > 0 and bold_chars / total >= 0.8


def _split_spans_by_gap(
    spans: list[dict[str, Any]],
    line_bbox: tuple[float, float, float, float],
) -> list["_PdfLine"] | None:
    """把“编号 | 标题 | 正文”同一行按大水平间距拆成多段。

    仅当首段是纯编号、第二段是短标题、第三段是正文时生效；普通正文
    即使有字间距也不会被拆开，避免破坏正常段落。
    """
    items: list[Any] = []
    for span in spans:
        text = span.get("text", "")
        if not text:
            continue
        items.append(
            (
                text,
                _bbox_tuple(span.get("bbox", line_bbox)),
                float(span.get("size", 0.0) or 0.0),
                int(span.get("flags", 0) or 0),
                str(span.get("font", "")),
            )
        )
    if not items:
        return None
    max_size = max(item[2] for item in items) or 10.0
    threshold = max(6.0, max_size * 0.8)
    segments: list[list[Any]] = []
    current: list[Any] = []
    prev_x1: float | None = None
    for item in items:
        if (
            prev_x1 is not None
            and item[1][0] - prev_x1 > threshold
            and current
        ):
            segments.append(current)
            current = []
        current.append(item)
        prev_x1 = max(prev_x1 or item[1][2], item[1][2])
    if current:
        segments.append(current)
    if len(segments) < 3:
        return None
    first = "".join(item[0] for item in segments[0]).strip()
    second = "".join(item[0] for item in segments[1]).strip()
    third = "".join(item[0] for item in segments[2]).strip()
    if not re.fullmatch(r"\d{1,3}(?:\.\d{1,3})*[.、]?", first):
        return None
    if not re.search(r"[A-Za-z一-鿿]", second):
        return None
    if len(second) > 24 or _SENTENCE_PUNCT.search(second):
        return None
    if len(third) < 8:
        return None
    lines: list[_PdfLine] = []
    for segment in segments:
        text = "".join(item[0] for item in segment).strip()
        if not text:
            continue
        bbox = (
            min(item[1][0] for item in segment),
            min(item[1][1] for item in segment),
            max(item[1][2] for item in segment),
            max(item[1][3] for item in segment),
        )
        size = max(item[2] for item in segment)
        font_counter: Counter[str] = Counter()
        for item in segment:
            font_counter[item[4]] += len(item[0])
        font = font_counter.most_common(1)[0][0] if font_counter else ""
        bold = _line_is_bold(
            [
                {"text": item[0], "flags": item[3], "size": item[2]}
                for item in segment
            ],
            font,
        )
        lines.append(
            _PdfLine(text=text, bbox=bbox, size=size, bold=bold, font=font)
        )
    if len(lines) < 3:
        return None
    return lines


# 作者行：全部由 2-4 字中文姓名 + 上标数字/空白/逗号组成。
_AUTHOR_LIST_RE = re.compile(
    r"^[\s　\d,，、;；]*"
    r"(?:[一-鿿]{2,4})"
    r"(?:[\s　\d,，、;；]+[一-鿿]{2,4})*"
    r"[\s　\d,，、;；]*$"
)
_AFFILIATION_KEYWORDS = (
    "大学", "学院", "医院", "研究所", "研究院", "实验室", "动物医学",
    "兽医学", "邮编", "e-mail", "邮箱", "公司", "中心", "门诊", "诊所",
    "宠物", "连锁",
    # 英文期刊的作者单位/学位信息
    "department", "university", "college", "school of", "institute",
    "hospital", "clinic", "corresponding author", "author for correspondence",
    "dvm", "davdc", "msc", "phd", "dabvp", "dacvim",
)
_METADATA_LABEL_RE = re.compile(
    r"^\s*(作者|通信作者|通讯作者|翻译|译者|审校|校对|编辑|收稿日期|"
    r"基金项目|作者简介|单位|通讯地址|e-?mail|邮箱|"
    r"corresponding author|author for correspondence|affiliations?)\s*[：:]",
    re.IGNORECASE,
)
# 期刊页眉/版权行的引用格式（Journal of ... / Vol. / DOI）。
_MASTHEAD_RE = re.compile(
    r"^\s*(?:journal of|vol\.?\s*\d|volume\s*\d|doi\s*:|https?://doi\.)",
    re.IGNORECASE,
)
_SOURCE_NOTE_RE = re.compile(
    r"(选自|摘自|转载).{0,30}(杂志|期刊|第\s*\d+\s*卷)"
)
# 图/表编号：支持“图1”“图2-1”“图B.2”“Figure S1”这类正文与附录编号，
# 附录图表常用字母前缀（表B.2、图C.1），不能只认纯数字。
_CAPTION_NUMBER = (
    r"(?:[A-Za-z]\s*[.\-·]?\s*)?(?:\d+(?:[.\-]\d+)*|[一二三四五六七八九十]+)"
)
# 图片说明：图 2：... / Figure 2. / Fig. 2 / 图B.2；表说明不在此列。
_FIGURE_CAPTION_RE = re.compile(
    rf"^\s*(?:图|figure|fig\.?)\s*{_CAPTION_NUMBER}",
    re.IGNORECASE,
)
# 图注前的面板字母：A B C D E F图 1 MRI...
_PANEL_CAPTION_RE = re.compile(
    rf"^\s*(?:[A-Za-z]\s*){{2,}}(?:图|figure|fig\.?)\s*{_CAPTION_NUMBER}",
    re.IGNORECASE,
)
# “图1显示/可见/为...” 是正文引用，不是图注。
_FIGURE_REFERENCE_RE = re.compile(
    r"^\s*(?:图|figure|fig\.?)\s*[\d一二三四五六七八九十]+\s*"
    r"(?:显示|可见|所示|为|是|中|的|表明|提示)",
    re.IGNORECASE,
)
# 表说明（表1/表2-1/表B.2/Table S1）保留为文字，但不能更新标题栈。
# 附录表题若被当成标题，会把表前的“见表B.2。”切成孤立短 chunk。
_TABLE_CAPTION_RE = re.compile(
    rf"^\s*(?:附?表|续表|table)\s*{_CAPTION_NUMBER}",
    re.IGNORECASE,
)
# 图片面板字母（A B C D E F）单独成块时是图片标注，不是正文。
_PANEL_LABEL_RE = re.compile(r"^\s*(?:[A-Za-z]\s*){2,}$")
# 文末作者简介/通信作者信息；不是知识正文。
_AUTHOR_BIO_RE = re.compile(
    r"(作者简介|通信作者|通讯作者|现攻读|专业兴趣方向|"
    r"国家执业|执业兽医师|本科毕业于|"
    r"攻读.{0,10}(?:硕士|博士))"
)


def _looks_like_author_list(text: str) -> bool:
    """整块是否像作者署名行（中文姓名 + 上标数字，或英文姓名列表）。"""
    value = (text or "").strip()
    if not value or len(value) > 80:
        return False
    if _AUTHOR_LIST_RE.match(value):
        return True
    # 英文作者：去掉数字/分隔符后，所有 token 都是首字母大写的姓名/缩写。
    cleaned = re.sub(r"[\d\s,;，、]+", " ", value).strip()
    tokens = [token for token in cleaned.split() if token]
    if len(tokens) < 2:
        return False
    name_tokens = [
        token
        for token in tokens
        if re.fullmatch(r"[A-Z][A-Za-z.'\-]*", token)
    ]
    return len(name_tokens) == len(tokens) and len(name_tokens) >= 2


def _is_figure_caption(text: str, *, image_count: int) -> bool:
    """判断一段文本是否是图片说明（表说明不在此列）。"""
    value = (text or "").strip()
    if not value:
        return False
    if _PANEL_CAPTION_RE.match(value):
        return True
    match = _FIGURE_CAPTION_RE.match(value)
    if not match:
        return False
    if _FIGURE_REFERENCE_RE.match(value):
        return False
    # “图1”“图2”这种只有编号的标签块同样不是知识正文。
    remainder = value[match.end():].strip(" 　：:.-—_()（）")
    if not remainder:
        return True
    # 图注一般带冒号或明显长于“图1”这种标签。
    return "：" in value[:12] or ":" in value[:12] or len(value) >= 8


def _is_author_metadata(
    text: str,
    *,
    page_no: int,
    bbox: tuple[float, float, float, float],
    page_height: float,
    body_size: float,
    title_bbox: tuple[float, float, float, float] | None,
    fonts: set[str],
    max_size: float,
) -> bool:
    """判断页首的作者/单位/来源说明等元数据，避免被当成章节标题。"""
    value = (text or "").strip()
    if not value or len(value) > 140:
        return False
    # 这些标签在任何页面都属于元数据。
    if _METADATA_LABEL_RE.match(value):
        return True
    if _SOURCE_NOTE_RE.search(value):
        return True
    # “—— 《What Is Your Diagnosis？》” 这类副标题/栏目名。
    if value.startswith(("——", "—", "-")) and "《" in value and "》" in value:
        return True
    # 文末作者简介/通信作者信息。
    if len(value) <= 300 and _AUTHOR_BIO_RE.search(value):
        return True
    # 期刊页眉/版权引用行，不是知识正文。
    if len(value) <= 120 and _MASTHEAD_RE.match(value):
        return True
    if _CONTINUED_RE.match(value):
        return True
    if page_no != 1 or title_bbox is None:
        return False
    # 只处理标题下方、正文开始前的页首区域。
    if bbox == title_bbox:
        return False
    if bbox[1] < title_bbox[1] - 10 or bbox[1] > page_height * 0.45:
        return False
    # “前言/病史/讨论”等独立小节标题即使很短也不能当作者名丢掉。
    normalized = re.sub(r"\s+", "", value).rstrip("：:").lower()
    if normalized in _SECTION_LABEL_HEADINGS:
        return False
    if _looks_like_author_list(value):
        compact = re.sub(r"[\s　]", "", value)
        if re.fullmatch(r"[一-鿿]{2,4}", compact):
            # 单个姓名：只接受作者/黑体字体且不高于正文字号。
            return (
                bool(fonts)
                and all(_font_is_heading(font) or "fzkt" in font.lower() for font in fonts)
                and max_size <= body_size + 0.6
            )
        return True
    lower_value = value.lower()
    if len(value) <= 300 and any(
        key in lower_value for key in _AFFILIATION_KEYWORDS
    ):
        return True
    return False


def _caption_continues(
    lines: list["_PdfLine"],
    bbox: tuple[float, float, float, float],
    context: tuple[float, str, tuple[float, float, float, float]],
) -> bool:
    """图注被 PDF 拆成多个块时，判断当前块是否仍是同一图注。"""
    size, font, ctx_bbox = context
    if any(abs(line.size - size) > 0.8 for line in lines):
        return False
    if any((line.font or "") != font for line in lines):
        return False
    if bbox[1] - ctx_bbox[3] > 24 or ctx_bbox[1] - bbox[3] > 24:
        return False
    overlap = min(bbox[2], ctx_bbox[2]) - max(bbox[0], ctx_bbox[0])
    narrower = min(bbox[2] - bbox[0], ctx_bbox[2] - ctx_bbox[0])
    return narrower > 0 and overlap / narrower >= 0.5


def _find_document_title(
    payload: "_PdfPage",
    *,
    body_size: float,
    boilerplate: set[str],
) -> tuple[
    tuple[float, int, float], tuple[float, float, float, float], str
] | None:
    """在第一页找出文档标题块（最大字号、页首、非页眉/非元数据）。

    返回 ``(权重, bbox, 文本)``；权重用于跨页比较候选。
    """
    best: tuple[tuple[float, int, float], tuple[float, float, float, float], str] | None = None
    for block in payload.blocks:
        if any(
            _overlap_ratio(block.bbox, table.bbox) >= 0.5
            for table in payload.tables
        ):
            continue
        lines = [
            line
            for line in block.lines
            if line.text.strip()
            and _boilerplate_key(_normalize_line(line.text)) not in boilerplate
        ]
        if not lines:
            continue
        text = join_wrapped_lines([line.text for line in lines]).strip()
        if not text or len(text) > 220 or _SENTENCE_PUNCT.search(text):
            continue
        # 页眉/页脚区域与正文中段不作为标题候选。
        if block.bbox[1] < payload.height * 0.06:
            continue
        if block.bbox[3] > payload.height * 0.55:
            continue
        max_size = max(line.size for line in lines)
        if max_size < 14 or (body_size > 0 and max_size < body_size * 1.35):
            continue
        weight = (
            round(max_size, 1),
            1 if any(line.bold for line in lines) else 0,
            -block.bbox[1],
        )
        if best is None or weight > best[0]:
            best = (weight, block.bbox, text)
    if best is None:
        return None
    return best[0], best[1], best[2]


def _title_key(text: str) -> str:
    """标题比较用的归一化：去掉空白与书名号/标点。"""
    return re.sub(r"[\s《》〈〉“”\"'·:：、，,.\-—_]", "", text or "").lower()


# 段落开头“标签 + 正文”同块的中文小节标题（如“前　言：癫痫是……”）。
_INLINE_SECTION_CJK = (
    "影像学发现与判读", "影像学检查征象与判读", "影像学检查", "影像学表现",
    "病例情况", "病例介绍", "病例报告", "临床检查", "检查过程", "治疗方案",
    "治疗及转归", "治疗与转归", "诊断结果", "基本信息", "既往史", "现病史",
    "病史", "主诉", "前言", "引言", "讨论", "结论", "预后", "预防", "康复",
    "护理", "麻醉", "手术", "诊断", "治疗",
)
_INLINE_SECTION_EN = (
    "case presentation", "case report", "materials and methods", "results",
    "discussion", "conclusion", "introduction", "background", "methods",
)
_SECTION_KEYWORDS = frozenset(
    {*_INLINE_SECTION_CJK, *_INLINE_SECTION_EN, "摘要", "abstract", "summary",
     "检查", "影像学", "转归", "病例", "结果", "方法", "references"}
)


def _loose_cjk_pattern(label: str) -> str:
    """允许汉字之间夹空白，匹配“前　言”这种排版。"""
    return r"\s*".join(re.escape(char) for char in label)


_INLINE_SECTION_RE = re.compile(
    r"^\s*(?P<label>"
    + "|".join(
        sorted(
            [_loose_cjk_pattern(item) for item in _INLINE_SECTION_CJK],
            key=len,
            reverse=True,
        )
        + [re.escape(item) for item in _INLINE_SECTION_EN]
    )
    + r")\s*[：:]\s*(?P<body>.{10,})$",
    re.IGNORECASE | re.DOTALL,
)


def _split_inline_numbered_heading(
    lines: list["_PdfLine"],
) -> tuple[str, str, int] | None:
    """拆分同一行的“编号 / 标题 / 正文”，返回 (标题, 正文, 级别)。

    药典/标准常见 ``2.2 | 培养特性 | 在CM 改良培养基…``；如果不拆，
    整块会因为“2.2培养特性…”命中编号规则而变成 heading，正文也跟着
    进入标题路径，后续 chunk 既丢正文又过小。
    """
    if len(lines) < 3:
        return None
    first = (lines[0].text or "").strip()
    if not re.fullmatch(r"\d{1,3}(?:\.\d{1,3})*[.、]?", first):
        return None
    if not (_same_row(lines[0], lines[1]) and _same_row(lines[1], lines[2])):
        return None
    title = (lines[1].text or "").strip()
    if not title or len(title) > 24 or _SENTENCE_PUNCT.search(title):
        return None
    body = join_wrapped_lines([line.text for line in lines[2:]]).strip()
    if len(body) < 10:
        return None
    number = first.rstrip(".、")
    heading = f"{number} {title}"
    level = _numbered_heading_level(heading) or 2
    return heading, body, level


def _split_inline_section_label(text: str) -> tuple[str, str] | None:
    """拆分“前　言：正文……”同块的小节标题与正文。"""
    match = _INLINE_SECTION_RE.match(text or "")
    if not match:
        return None
    label = re.sub(r"\s+", "", match.group("label")).strip()
    body = match.group("body").strip()
    if not label or len(body) < 10:
        return None
    return label, body


def _has_section_keyword(text: str) -> bool:
    """编号标题后是否带常见小节词，用于区分列表项。"""
    value = (text or "").lower()
    return any(keyword in value for keyword in _SECTION_KEYWORDS)


# 表格表头/化验值的常见词；命中多个时更可能是表格碎片而不是标题。
_TABLE_HEADER_WORDS = frozenset(
    {
        "parameter", "result", "results", "reference", "range", "interval",
        "exam", "value", "values", "unit", "units", "finding", "findings",
        "dilution", "marker", "stain", "clone", "catalog", "manufacturer",
        "antibody", "antibodies", "test", "susceptibility", "sample",
        "group", "normal", "case", "antigen", "retrieval",
    }
)
_GENERIC_TABLE_WORDS = frozenset(
    {"case", "sample", "parameter", "result", "value", "range", "marker",
     "stain", "dilution", "exam"}
)
_TIME_PREFIX_RE = re.compile(
    r"^\s*\d+(?:\.\d+)?\s*"
    r"(?:days?|weeks?|months?|years?|hours?|minutes?|mg|ml|kg|g|%)\b",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"(?:https?://|www\.|\.com\b|\.org\b|\.edu\b)", re.I)
_CONTINUED_RE = re.compile(r"^\s*\(?\s*(?:continued|cont['’]?d)\b", re.I)


def _numeric_token_ratio(text: str) -> float:
    """数字 token 占比；化验值/表格行通常很高。"""
    tokens = re.findall(r"[A-Za-z]+|\d+(?:[.,]\d+)?", text or "")
    if not tokens:
        return 0.0
    return sum(1 for token in tokens if token[0].isdigit()) / len(tokens)


def _looks_like_table_fragment(text: str) -> bool:
    """识别未抽成表格的表头/化验值行，避免变成 heading。"""
    value = (text or "").strip()
    if not value:
        return False
    tokens = re.findall(r"[A-Za-z]+|\d+(?:[.,]\d+)?", value)
    if len(tokens) >= 2:
        numeric = sum(1 for token in tokens if token[0].isdigit())
        if numeric / len(tokens) >= 0.5:
            return True
    words = [word.lower() for word in re.findall(r"[A-Za-z]+", value)]
    if not words:
        return False
    if len(words) == 1 and words[0] in _GENERIC_TABLE_WORDS:
        return True
    hits = sum(1 for word in set(words) if word in _TABLE_HEADER_WORDS)
    if hits >= 2 and len(words) <= 8:
        return True
    return False


def _numbered_heading_level(text: str) -> int | None:
    """识别 ``1.诊断`` / ``1.1 病例信息`` / ``1.3.1 血常规`` 这类编号标题。

    编号层级映射：``1.`` → 2，``1.1`` → 3，``1.3.1`` → 4；文档大标题保持 1。
    只接受编号后直接跟中文/英文的短行，避免把编号列表和表格数值当标题。
    """
    value = (text or "").strip()
    if not value or len(value) > 80:
        return None
    if _NUMBERED_PIPE_HEADING.match(value):
        # 期刊格式 “1 | Introduction” 放在文档标题下一级（level 3），
        # “2.1 | Histopathology” 再下一级（level 4）。
        number = value.split("|", 1)[0].strip().rstrip(".")
        # 文档标题为一级；"1 | Introduction" -> 2，"2.1 | ..." -> 3。
        return min(6, number.count(".") + 2)
    match = _NUMBERED_COMPOUND_HEADING.match(value)
    if match:
        number = value.split()[0].rstrip(".")
        return min(6, number.count(".") + 2)
    if _NUMBERED_SINGLE_HEADING.match(value):
        return 2
    # 标准文本 “1 范围”“7.1 样品采集”：编号后是空格。要求标题短、无逗号，
    # 且不以量词/单位开头，避免把“1 只犬出现呕吐”这类编号列表当标题。
    space_match = _NUMBERED_SPACE_HEADING.match(value)
    if space_match:
        title = space_match.group(1).strip()
        if (
            0 < len(title) <= 30
            and title[0] not in _NUMBERED_LIST_PREFIX
            and not any(mark in title for mark in ("，", ",", "。", "；", ";"))
        ):
            number = value.split()[0].rstrip(".")
            return min(6, number.count(".") + 2)
    # 药典/标准常见 “2气囊病变记分标准…” 这种编号后直接跟汉字的标题。
    cjk_match = re.match(r"^\s*(\d{1,2})([一-鿿].*)$", value)
    if cjk_match:
        title = cjk_match.group(2)
        if (
            0 < len(title) <= 30
            and title[0] not in _NUMBERED_LIST_PREFIX
            and not any(mark in title for mark in ("，", ",", "。", "；", ";"))
        ):
            return 2
    return None


def _is_page_furniture(line: "_PdfLine", page_height: float) -> bool:
    """判断页边出版社页脚/版权/下载水印，避免混入正文 chunk。"""
    if line.ocr:
        return False
    text = (line.text or "").strip()
    if not text:
        return False
    in_margin = (
        line.bbox[1] < page_height * 0.06
        or line.bbox[3] > page_height * 0.88
    )
    return in_margin and bool(_PAGE_FURNITURE.search(text))


def _bbox_width(bbox: tuple[float, float, float, float]) -> float:
    """返回 bbox 宽度。"""
    return max(0.0, bbox[2] - bbox[0])


def _detect_column_split(
    entries: list[tuple[tuple[float, float, float, float], str, object]],
    page_width: float,
) -> float | None:
    """检测双栏页面的竖向分栏位置；单栏返回 ``None``。

    只使用“不跨页面中线、且不是跨栏块”的文本/表格框，避免标题、摘要等
    居中或通栏内容干扰；左右两侧都需要足够多的块，且中间存在明确空白。
    """
    mid = page_width / 2
    left: list[tuple[float, float, float, float]] = []
    right: list[tuple[float, float, float, float]] = []
    for bbox, _kind, _item in entries:
        if _bbox_width(bbox) >= page_width * _WIDE_BLOCK_RATIO:
            continue
        if bbox[2] <= mid:
            left.append(bbox)
        elif bbox[0] >= mid:
            right.append(bbox)
    # 期刊首页常只有 2 个窄块/栏（如 Introduction 标题 + 首段），
    # 3 块阈值会把这种页误判为单栏。
    if len(left) < 2 or len(right) < 2:
        return None
    gap_left = max(bbox[2] for bbox in left)
    gap_right = min(bbox[0] for bbox in right)
    if gap_right - gap_left < _COLUMN_SPLIT_MIN_GAP:
        return None
    return (gap_left + gap_right) / 2


def _order_interval(
    entries: list[tuple[tuple[float, float, float, float], str, object]],
    split_x: float,
) -> list[tuple[tuple[float, float, float, float], str, object]]:
    """对同一纵向区间内的块排序：左栏整体在前，右栏整体在后。"""
    if not entries:
        return []
    if any(bbox[0] < split_x < bbox[2] for bbox, _kind, _item in entries):
        # 含跨栏/居中块时无法安全分栏，退化为从上到下、从左到右。
        return sorted(
            entries,
            key=lambda entry: (round(entry[0][1], 1), round(entry[0][0], 1)),
        )
    left = [entry for entry in entries if entry[0][2] <= split_x]
    right = [entry for entry in entries if entry[0][0] >= split_x]
    if left and right and len(left) + len(right) == len(entries):
        return sorted(
            left,
            key=lambda entry: (round(entry[0][1], 1), round(entry[0][0], 1)),
        ) + sorted(
            right,
            key=lambda entry: (round(entry[0][1], 1), round(entry[0][0], 1)),
        )
    return sorted(
        entries,
        key=lambda entry: (round(entry[0][1], 1), round(entry[0][0], 1)),
    )


def _page_entries_in_reading_order(
    entries: list[tuple[tuple[float, float, float, float], str, object]],
    page_width: float,
) -> list[tuple[tuple[float, float, float, float], str, object]]:
    """按双栏版面还原阅读顺序。

    采用“跨栏块作为纵向锚点 + 栏内自上而下 + 左栏先于右栏”的简单 XY-cut：

    1. 跨栏块（标题、通栏摘要、通栏表格行）按 y 顺序插入；
    2. 锚点之间的窄块，若页面判定为双栏，则先输出左栏再输出右栏；
    3. 单栏页面直接按 y/x 排序。
    """
    if len(entries) <= 1:
        return entries
    split_x = _detect_column_split(entries, page_width)
    if split_x is None:
        return sorted(
            entries,
            key=lambda entry: (round(entry[0][1], 1), round(entry[0][0], 1)),
        )

    wide: list[tuple[tuple[float, float, float, float], str, object]] = []
    narrow: list[tuple[tuple[float, float, float, float], str, object]] = []
    for entry in entries:
        bbox = entry[0]
        # 跨栏块本身也作为锚点：通栏标题/摘要，以及略越过中线的段落行。
        if (
            _bbox_width(bbox) >= page_width * _WIDE_BLOCK_RATIO
            or bbox[0] < split_x < bbox[2]
        ):
            wide.append(entry)
        else:
            narrow.append(entry)

    ordered: list[tuple[tuple[float, float, float, float], str, object]] = []
    pending = list(narrow)
    for anchor in sorted(
        wide,
        key=lambda entry: (round(entry[0][1], 1), round(entry[0][0], 1)),
    ):
        anchor_top = anchor[0][1]
        above: list[tuple[tuple[float, float, float, float], str, object]] = []
        rest: list[tuple[tuple[float, float, float, float], str, object]] = []
        for entry in pending:
            if entry[0][3] <= anchor_top + 1:
                above.append(entry)
            else:
                rest.append(entry)
        pending = rest
        if above:
            ordered.extend(_order_interval(above, split_x))
        ordered.append(anchor)
    if pending:
        ordered.extend(_order_interval(pending, split_x))
    return ordered


class PdfParser:
    """v1 PDF 解析器。"""

    parser_id = "pdf_pymupdf"
    parser_version = "3"

    def __init__(
        self,
        *,
        ocr_config: OcrConfig | None = None,
        ocr_provider: OcrProvider | None = None,
        image_pdf_config: ImagePdfConfig | None = None,
    ) -> None:
        """保存 OCR/图片 PDF 配置；provider 未注入时按配置自动选择。"""
        self._ocr_config = ocr_config or OcrConfig()
        self._ocr_provider = (
            ocr_provider
            if ocr_provider is not None
            else create_ocr_provider(self._ocr_config)
        )
        self._image_pdf_config = image_pdf_config or ImagePdfConfig()
        self._image_pdf_pipeline = (
            ImagePdfPipeline(self._image_pdf_config)
            if self._image_pdf_config.enabled
            else None
        )

    async def parse(
        self,
        source: DocumentSource,
        profile: DocumentProfile,
        *,
        doc_id: UUID,
        kb_id: UUID,
        data: bytes,
    ) -> ParsedDocument:
        """解析 PDF 字节为 ``ParsedDocument``。"""
        try:
            import pymupdf as fitz  # type: ignore[import-not-found]
        except ImportError:
            try:
                import fitz  # type: ignore[import-not-found]
            except ImportError:
                return await self._parse_pdfplumber(
                    source, profile, doc_id=doc_id, kb_id=kb_id, data=data
                )

        # PyMuPDF 是同步 CPU 操作：放到线程池，避免阻塞 API 事件循环。
        return await asyncio.to_thread(
            self._parse_pymupdf,
            fitz,
            source,
            profile,
            doc_id,
            kb_id,
            data,
        )

    # ——————————————————— PyMuPDF 主路径 ———————————————————

    def _parse_pymupdf(
        self,
        fitz,
        source: DocumentSource,
        profile: DocumentProfile,
        doc_id: UUID,
        kb_id: UUID,
        data: bytes,
    ) -> ParsedDocument:
        """同步执行 PyMuPDF 解析（在线程池中调用）。"""
        try:
            document = fitz.open(stream=data, filetype="pdf")
        except Exception as exc:
            raise RagError(
                RagErrorCode.INVALID_REQUEST,
                "PDF 无法打开",
                details={"filename": source.filename, "cause": str(exc)},
            ) from exc

        ocr_pages = 0
        ocr_limited = False
        ocr_warnings: list[str] = []
        # OCR 是可选增强：引擎存在但依赖损坏（如 torch DLL 缺失）时，只应
        # 跳过扫描页并告警，不能让整篇文本 PDF 解析失败。一次失败后本文件
        # 不再重试，避免每页都重复加载坏依赖导致解析超时。
        ocr_failed = False
        try:
            payloads: list[_PdfPage] = []
            for page_no, page in enumerate(document, start=1):
                payload = self._extract_page(page, page_no, data=data)
                if self._should_ocr(payload):
                    if (
                        self._ocr_provider is not None
                        and not ocr_failed
                        and ocr_pages < self._ocr_config.max_pages
                    ):
                        try:
                            self._apply_ocr(page, payload)
                        except Exception as exc:  # noqa: BLE001 - OCR 可降级
                            ocr_failed = True
                            message = (
                                f"OCR 不可用，已跳过扫描页: {str(exc)[:200]}"
                            )
                            ocr_warnings.append(message)
                            logger.warning(
                                "PDF OCR 失败，按无 OCR 继续: %s: %s",
                                source.filename,
                                exc,
                            )
                        else:
                            ocr_pages += 1
                    elif ocr_pages >= self._ocr_config.max_pages:
                        ocr_limited = True
                payloads.append(payload)
        except RagError:
            raise
        except Exception as exc:
            raise RagError(
                RagErrorCode.INVALID_REQUEST,
                "PDF 解析失败",
                details={"filename": source.filename, "cause": str(exc)},
            ) from exc
        finally:
            document.close()

        # 整篇都是图片/扫描页时，交给版式解析流水线（MinerU/RapidOCR）。
        image_page_numbers = [
            payload.page_no
            for payload in payloads
            if self._is_image_like_page(payload)
        ]
        image_pipeline_warnings: list[str] = []
        if (
            image_page_numbers
            and len(image_page_numbers) == len(payloads)
            and self._image_pdf_pipeline is not None
        ):
            try:
                return self._image_pdf_pipeline.parse(
                    data,
                    page_numbers=image_page_numbers,
                    total_pages=len(payloads),
                    source=source,
                    doc_id=doc_id,
                    kb_id=kb_id,
                    profile=profile,
                )
            except RagError as exc:
                if self._ocr_provider is None:
                    raise
                image_pipeline_warnings.append(
                    f"图片 PDF 版式解析失败，回退简单 OCR: {exc.message}"
                )

        body_size = self._body_font_size(payloads)
        boilerplate = self._detect_boilerplate(payloads)
        # 目录页不是正文：剔除后在检索期不会用章节名/页码污染召回。
        toc_pages = _toc_pages(payloads)

        blocks: list[ContentBlock] = []
        pages: list[Page] = []
        dropped_lines = 0
        has_tables = False
        has_images = any(item.image_count > 0 for item in payloads)
        # 标题栈跨页延续：下一页首段仍能拿到完整的标题路径。
        headings = HeadingStack()
        # 封面页可能是纯图片；取前 3 页中最早出现的标题候选。
        document_title: tuple[
            int, tuple[float, float, float, float], str
        ] | None = None
        for candidate_page in payloads[:3]:
            info = _find_document_title(
                candidate_page,
                body_size=candidate_page.body_size or body_size,
                boilerplate=boilerplate,
            )
            if info is not None:
                _weight, bbox, text = info
                document_title = (candidate_page.page_no, bbox, text)
                break

        for payload in payloads:
            page_blocks, page_tables, page_dropped = self._build_page_blocks(
                payload,
                body_size=payload.body_size or body_size,
                boilerplate=boilerplate,
                doc_id=doc_id,
                start_order=len(blocks),
                headings=headings,
                skip_toc=payload.page_no in toc_pages,
                document_title=document_title,
            )
            blocks.extend(page_blocks)
            dropped_lines += page_dropped
            has_tables = has_tables or page_tables
            page_warnings = []
            if payload.ocr_used:
                page_warnings.append("本页文本来自 OCR")
            elif not page_blocks and payload.image_count > 0:
                page_warnings.append("页面仅含图片，OCR 未启用或不可用")
            pages.append(
                make_page(
                    page_no=payload.page_no,
                    blocks=page_blocks,
                    width=payload.width,
                    height=payload.height,
                    rotation=payload.rotation,
                    warnings=page_warnings,
                )
            )

        page_count = len(payloads)
        native_text_pages = sum(
            1
            for item in payloads
            if (item.blocks or item.tables) and not item.ocr_used
        )
        ocr_page_count = sum(1 for item in payloads if item.ocr_used)
        pages_with_text = native_text_pages + ocr_page_count
        coverage = pages_with_text / page_count if page_count else 0.0
        ocr_ratio = ocr_page_count / page_count if page_count else 0.0

        warnings: list[str] = [*image_pipeline_warnings, *ocr_warnings]
        if dropped_lines:
            warnings.append(f"已过滤跨页重复的水印/页眉页脚 {dropped_lines} 行")
        metadata_dropped = sum(item.dropped_metadata for item in payloads)
        if metadata_dropped:
            warnings.append(f"已过滤作者/单位等元数据 {metadata_dropped} 块")
        captions_dropped = sum(item.dropped_captions for item in payloads)
        if captions_dropped:
            warnings.append(f"已过滤图片说明 {captions_dropped} 块")
        if ocr_page_count:
            warnings.append(f"已对 {ocr_page_count} 页扫描页执行 OCR")
        if ocr_limited:
            warnings.append(
                f"OCR 页数达到上限 {self._ocr_config.max_pages}，其余页面未识别"
            )
        low_confidence_pages = [
            item.page_no
            for item in payloads
            if item.ocr_used
            and item.ocr_confidence is not None
            and item.ocr_confidence < self._ocr_config.min_confidence
        ]
        if low_confidence_pages:
            warnings.append(
                "部分 OCR 页面置信度较低: "
                + ", ".join(str(item) for item in low_confidence_pages[:10])
            )
        if not pages_with_text:
            if has_images and self._ocr_config.enabled and self._ocr_provider is None:
                raise RagError(
                    RagErrorCode.INVALID_CONFIG,
                    "扫描版 PDF 需要 OCR，但未检测到可用的 OCR 提供方"
                    "（PaddleOCR 或 Tesseract）",
                    details={
                        "filename": source.filename,
                        "provider": self._ocr_config.provider,
                    },
                )
            warnings.append("PDF 未检测到可提取文本层（OCR/版式解析不可用或失败）")

        return make_parsed_document(
            source=source,
            doc_id=doc_id,
            kb_id=kb_id,
            parser_id=self.parser_id,
            parser_version=self.parser_version,
            profile=profile.model_copy(
                update={
                    "doc_type": DocumentType.PDF,
                    "page_count": page_count,
                    "has_text_layer": native_text_pages > 0,
                    "has_images": has_images,
                    "has_tables": has_tables,
                    "is_scanned": native_text_pages == 0 and ocr_page_count > 0,
                    "text_coverage": coverage,
                    "ocr_ratio": ocr_ratio,
                    "quality_score": coverage,
                }
            ),
            blocks=blocks,
            pages=pages,
            warnings=warnings,
        )

    def _extract_page(self, page, page_no: int, *, data: bytes) -> _PdfPage:
        """抽取一页的文本块、表格和基础版面信息。"""
        payload = _PdfPage(
            page_no=page_no,
            width=float(page.rect.width),
            height=float(page.rect.height),
            rotation=int(getattr(page, "rotation", 0) or 0),
        )
        try:
            payload.image_count = len(page.get_images(full=False))
        except Exception:
            payload.image_count = 0

        text_dict = page.get_text("dict")
        for raw_block in text_dict.get("blocks", []):
            if raw_block.get("type") != 0:
                # type=1 是图片块；图片页由 _apply_ocr / image_pdf 流水线处理。
                continue
            block = _PdfTextBlock(bbox=_bbox_tuple(raw_block.get("bbox", (0, 0, 0, 0))))
            for raw_line in raw_block.get("lines", []):
                spans = raw_line.get("spans", [])
                # 药典/标准常把“2.2 | 培养特性 | 正文…”放在同一行：
                # 先按大水平间距拆成编号/标题/正文段，避免整块被当成标题。
                segments = _split_spans_by_gap(
                    spans,
                    _bbox_tuple(raw_line.get("bbox", (0, 0, 0, 0))),
                )
                if segments is not None:
                    for segment in segments:
                        block.lines.append(segment)
                        payload.size_chars[round(segment.size, 1)] += len(
                            segment.text
                        )
                    continue
                text = "".join(span.get("text", "") for span in spans)
                if not text.strip():
                    continue
                size = max((float(span.get("size", 0.0)) for span in spans), default=0.0)
                font_counter: Counter[str] = Counter()
                for span in spans:
                    font_counter[str(span.get("font", ""))] += len(
                        span.get("text", "")
                    )
                font = font_counter.most_common(1)[0][0] if font_counter else ""
                bold = _line_is_bold(spans, font)
                line = _PdfLine(
                    text=text,
                    bbox=_bbox_tuple(raw_line.get("bbox", (0, 0, 0, 0))),
                    size=size,
                    bold=bold,
                    font=font,
                )
                block.lines.append(line)
                # 用字符数加权统计字号，后续据此判断文档正文字号。
                for span in spans:
                    payload.size_chars[round(size, 1)] += len(span.get("text", ""))
            if block.lines:
                payload.blocks.append(block)

        payload.tables = self._extract_tables(page, page_no=page_no, data=data)
        # 页面局部正文字号：同一文档可能混用 8.5pt 表格/10pt 正文/10pt 小标题；
        # 仅用全局字号会把某些正文整段误判成标题。样本太少时留空，使用全局值。
        local: Counter = Counter()
        for block in payload.blocks:
            if any(
                _overlap_ratio(block.bbox, table.bbox) >= 0.5
                for table in payload.tables
            ):
                continue
            for line in block.lines:
                if not line.ocr:
                    local[round(line.size, 1)] += len(line.text)
        if sum(local.values()) >= 200:
            payload.body_size = float(local.most_common(1)[0][0])
        return payload

    @staticmethod
    def _native_text_chars(payload: _PdfPage) -> int:
        """页面原生文本层的字符数（不含表格）。"""
        return sum(
            len(line.text)
            for block in payload.blocks
            for line in block.lines
            if not line.ocr
        )

    @classmethod
    def _is_image_like_page(cls, payload: _PdfPage) -> bool:
        """含图片且原生文本极少时视为图片/扫描页。"""
        return payload.image_count > 0 and cls._native_text_chars(payload) < 30

    @classmethod
    def _should_ocr(cls, payload: _PdfPage) -> bool:
        """只有“几乎无文本层且含图片”的页面才需要 OCR。"""
        return cls._is_image_like_page(payload)

    def _apply_ocr(self, page, payload: _PdfPage) -> None:
        """对扫描页执行 OCR，并把结果转成内部文本块。"""
        if self._ocr_provider is None:
            return
        result = self._ocr_provider.recognize(
            page,
            page_no=payload.page_no,
            dpi=self._ocr_config.dpi,
        )
        if not result.text:
            return
        lines: list[_PdfLine] = []
        for index, line in enumerate(result.lines):
            text = (line.text or "").strip()
            if not text:
                continue
            bbox = line.bbox or (
                0.0,
                float(index * 12),
                payload.width,
                float(index * 12 + 12),
            )
            lines.append(
                _PdfLine(
                    text=text,
                    bbox=_bbox_tuple(bbox),
                    size=10.0,
                    bold=False,
                    ocr=True,
                )
            )
        if not lines:
            return
        payload.blocks = [
            _PdfTextBlock(
                bbox=(0.0, 0.0, payload.width, payload.height),
                lines=lines,
            )
        ]
        payload.ocr_used = True
        payload.ocr_confidence = result.confidence
        # OCR 文本按正文处理；不参与字号统计，避免影响标题比例。
        for line in lines:
            payload.size_chars[10.0] += len(line.text)

    def _extract_tables(
        self,
        page,
        *,
        page_no: int,
        data: bytes,
    ) -> list[_PdfTable]:
        """抽取页面表格；复杂页面用 pdfplumber 兜底，避免 find_tables 崩溃。"""
        try:
            drawings = page.get_drawings()
        except Exception:
            drawings = []
        # find_tables 在复杂矢量页上可能直接 segfault（无法用 try 捕获），
        # 超过阈值时改用 pdfplumber；pdfplumber 失败则退回纯文本，不阻断解析。
        if len(drawings) > _PYMUPDF_TABLE_MAX_DRAWINGS:
            return self._extract_tables_pdfplumber(data, page_no)
        # 有竖线的网格表用 PyMuPDF find_tables；无竖线的三线表改用
        # pdfplumber——后者按文本列切分，比 find_tables 更能保住小数/区间值。
        structure = self._table_structure_kind(page, drawings=drawings)
        if structure is None:
            return []
        if structure == "borderless":
            return self._extract_tables_pdfplumber(
                data, page_no, borderless=True
            )
        try:
            finder = page.find_tables()
        except Exception:
            return self._extract_tables_pdfplumber(data, page_no)

        tables: list[_PdfTable] = []
        for raw_table in getattr(finder, "tables", []):
            try:
                raw_rows = raw_table.extract()
            except Exception:
                continue
            rows = self._normalize_table_rows(raw_rows)
            if not rows or not self._is_plausible_table(rows, raw_table.bbox):
                continue
            tables.append(
                _PdfTable(
                    bbox=_bbox_tuple(raw_table.bbox),
                    rows=rows,
                )
            )
        return tables

    def _extract_tables_pdfplumber(
        self,
        data: bytes,
        page_no: int,
        *,
        borderless: bool = False,
    ) -> list[_PdfTable]:
        """用 pdfplumber 抽取一页表格；依赖缺失或失败时返回空列表。"""
        try:
            import pdfplumber  # type: ignore[import-not-found]

            with pdfplumber.open(__import__("io").BytesIO(data)) as document:
                if page_no < 1 or page_no > len(document.pages):
                    return []
                page = document.pages[page_no - 1]
                # 竖排/旋转 90° 的表格（大型指南的横向表格）在 pdfplumber 中
                # 会按字符逆序返回；先按字符朝向判断，再逐单元格整体反转修复。
                vertical = self._pdfplumber_page_is_vertical(page)
                # 无竖线三线表：横线只覆盖数值列时，默认的“横竖都按线”
                # 会把标签列切掉（如 CBC 表只剩日期列）。这里用文本位置
                # 推断列、横线确定行——横线跨不满列时 pdfplumber 返回空，
                # 此时保持原文本块，避免丢掉项目名/正常范围等列。
                settings = (
                    {
                        "vertical_strategy": "text",
                        "horizontal_strategy": "lines",
                    }
                    if borderless
                    else None
                )
                raw_tables = (
                    page.find_tables(table_settings=settings)
                    if settings is not None
                    else page.find_tables()
                )
                tables: list[_PdfTable] = []
                for raw_table in raw_tables:
                    box = _bbox_tuple(raw_table.bbox)
                    # 无竖线表若几乎占满整页高度，通常是跨页合并单元格/
                    # 版式网格被 text+lines 误切，抽取结果会变成碎片；
                    # 宁可保留原文段落，也不要生成超长错位表格。
                    if (
                        borderless
                        and box is not None
                        and box[3] - box[1] > page.height * 0.8
                    ):
                        continue
                    raw_rows = raw_table.extract()
                    # 密度要在“删除空列”之前判断：合并单元格表经过列裁剪后
                    # 会显得很密，但原始网格仍是大片空位，抽取结果容易错列。
                    if borderless and self._raw_table_density(raw_rows) < 0.75:
                        continue
                    rows = self._normalize_table_rows(
                        raw_rows,
                        reverse_text=vertical,
                    )
                    if not rows or not self._is_plausible_table(
                        rows, raw_table.bbox, borderless=borderless
                    ):
                        continue
                    tables.append(
                        _PdfTable(
                            bbox=_bbox_tuple(raw_table.bbox),
                            rows=rows,
                        )
                    )
                return tables
        except Exception:
            return []

    @staticmethod
    def _pdfplumber_page_is_vertical(page) -> bool:
        """判断 pdfplumber 页面是否以竖排/旋转文本为主。

        ``upright=False`` 的字符是旋转文本；只有占比过半才反转表格单元格，
        避免普通页面里少量旋转水印/页码被误处理。
        """
        chars = getattr(page, "chars", None) or []
        if len(chars) < 10:
            return False
        rotated = sum(1 for item in chars if not item.get("upright", True))
        return rotated / len(chars) >= 0.5

    @staticmethod
    def _normalize_table_rows(
        raw_rows,
        *,
        reverse_text: bool = False,
    ) -> list[list[str]]:
        """把表格二维数据清理成非空行；单行/单列视作非表格。

        ``reverse_text=True`` 用于 pdfplumber 的竖排表格：它按字符逆序返回，
        整串反转后再压平换行即可恢复阅读顺序。不能按行反转，否则词序仍错。
        """
        def clean(cell) -> str:
            """单元格转字符串；竖排时整串反转并压成单行。"""
            value = "" if cell is None else str(cell).strip()
            if value and reverse_text:
                return " ".join(value[::-1].split())
            return value

        rows = [
            [clean(cell) for cell in row]
            for row in (raw_rows or [])
            if row
        ]
        rows = [row for row in rows if any(cell for cell in row)]
        if len(rows) < 2 or max((len(row) for row in rows), default=0) < 2:
            return []
        return rows

    @staticmethod
    def _table_structure_kind(page, *, drawings=None) -> str | None:
        """判断页面表格结构，返回 ``"grid"`` / ``"borderless"`` / ``None``。

        命中条件（二选一）：
        - ``grid``：3 条以上水平线，且某条水平线被 3 条以上竖线穿过；
        - ``borderless``：无竖线的三线表——至少 3 条水平线共享同一段足够宽
          的水平范围且 y 不同。这种表用 pdfplumber 抽取更稳，PyMuPDF 会把
          “5.05”“5.1-7.6” 这类小数/区间按列切成两半。

        单个矩形框只有 2 横 2 竖；多个独立矩形框的每条横线也只被 2 条竖线
        穿过，因此不会误触发；无竖线分支还要求横线共享水平范围，避免把
        页眉分隔线当成表格。
        """
        horizontals: list[tuple[float, float, float]] = []
        verticals: list[tuple[float, float, float]] = []
        if drawings is None:
            try:
                drawings = page.get_drawings()
            except Exception:
                return None

        for drawing in drawings:
            for item in drawing.get("items", []):
                if not item:
                    continue
                if item[0] == "l":
                    p1, p2 = item[1], item[2]
                    if abs(p1.y - p2.y) <= 1.0 and abs(p1.x - p2.x) >= 40:
                        horizontals.append(
                            (min(p1.x, p2.x), max(p1.x, p2.x), (p1.y + p2.y) / 2)
                        )
                    elif abs(p1.x - p2.x) <= 1.0 and abs(p1.y - p2.y) >= 20:
                        verticals.append(
                            (p1.x, min(p1.y, p2.y), max(p1.y, p2.y))
                        )
                elif item[0] == "re":
                    rect = item[1]
                    if rect.height <= 2 and rect.width >= 40:
                        horizontals.append((rect.x0, rect.x1, rect.y0))
                    elif rect.width <= 2 and rect.height >= 20:
                        verticals.append((rect.x0, rect.y0, rect.y1))

        if len(horizontals) < 3:
            return None
        # 常规网格表：横线被多条竖线穿过。
        if len(verticals) >= 3:
            for x0, x1, y in horizontals:
                crossings = sum(
                    1
                    for x, y0, y1 in verticals
                    if x0 - 1 <= x <= x1 + 1 and y0 - 1 <= y <= y1 + 1
                )
                if crossings >= 3:
                    return "grid"
        # 无竖线三线表：找 ≥3 条共享水平范围且 y 不同的横线。
        for x0, x1, _y in horizontals:
            if x1 - x0 < 80:
                continue
            ys = {round(_y, 1)}
            for ox0, ox1, oy in horizontals:
                if (ox0, ox1, oy) == (x0, x1, _y):
                    continue
                overlap = min(x1, ox1) - max(x0, ox0)
                if overlap > 0 and overlap >= 0.6 * min(x1 - x0, ox1 - ox0):
                    ys.add(round(oy, 1))
            if len(ys) >= 3:
                return "borderless"
        return None

    @staticmethod
    def _raw_table_density(raw_rows) -> float:
        """原始网格的非空单元格密度；用于识别合并单元格/错位表。"""
        rows = [list(row) for row in (raw_rows or []) if row is not None]
        if not rows:
            return 0.0
        width = max(len(row) for row in rows)
        if width == 0:
            return 0.0
        filled = sum(
            1
            for row in rows
            for cell in row
            if cell and str(cell).strip()
        )
        return filled / (len(rows) * width)

    @staticmethod
    def _is_plausible_table(
        rows: list[list[str]],
        bbox,
        *,
        borderless: bool = False,
    ) -> bool:
        """过滤误检的单行/单列/正文碎片“表格”。

        - 网格表：至少 2 行 2 列且宽度 ≥80；
        - 无竖线三线表（``borderless=True``）：还要求 ≥3 行、宽度 ≥100、
          非空单元格密度 ≥80%，避免把页眉横幅、合并单元格错位或正文碎片
          当成表；合并单元格较多时宁可保留原文段落。
        """
        if len(rows) < (3 if borderless else 2):
            return False
        width = max((len(row) for row in rows), default=0)
        if width < 2:
            return False
        box = _bbox_tuple(bbox)
        min_width = 100 if borderless else 80
        if box is not None and box[2] - box[0] < min_width:
            return False
        if borderless:
            filled = sum(
                1
                for row in rows
                for cell in row
                if cell and str(cell).strip()
            )
            if filled / (len(rows) * width) < 0.8:
                return False
        return True

    @staticmethod
    def _body_font_size(payloads: list[_PdfPage]) -> float:
        """按字符数加权选择文档正文字号。

        必须排除与表格区域重叠的文本块：大型指南/年报的表格常用 8pt，
        而正文用 10pt；如果把表格字符计入统计，正文会被误判成“更大字号”
        并整段变成标题。排除表格后仍无样本时才退回全量统计。
        """
        counter: Counter = Counter()
        for payload in payloads:
            for block in payload.blocks:
                if any(
                    _overlap_ratio(block.bbox, table.bbox) >= 0.5
                    for table in payload.tables
                ):
                    continue
                for line in block.lines:
                    if line.ocr:
                        continue
                    counter[round(line.size, 1)] += len(line.text)
        if counter:
            return float(counter.most_common(1)[0][0])
        fallback: Counter = Counter()
        for payload in payloads:
            fallback.update(payload.size_chars)
        if not fallback:
            return 10.0
        return float(fallback.most_common(1)[0][0])

    @staticmethod
    def _detect_boilerplate(payloads: list[_PdfPage]) -> set[str]:
        """识别跨页重复且位置稳定的短行（水印/页眉页脚）。

        只有同时满足以下条件才丢弃：
        - 归一化后长度 <= 60；
        - 数字替换为 ``#`` 后出现在至少 30% 的页面（且不少于 3 页）；
        - 各页的纵坐标差异 <= 12pt。

        数字模板用于识别“管理指南93/161”这类带页码的页眉；否则每页文本
        都不同，页眉会被误判为标题并切断跨页章节。
        """
        if not payloads:
            return set()
        min_pages = max(3, math.ceil(len(payloads) * 0.3))
        occurrences: dict[str, list[tuple[int, float]]] = {}
        for payload in payloads:
            if payload.ocr_used:
                # OCR 行没有可靠位置，参与跨页重复统计容易误删正文。
                continue
            for block in payload.blocks:
                for line in block.lines:
                    normalized = _normalize_line(line.text)
                    if not normalized or len(normalized) > 60:
                        continue
                    # 页码会变化，按“数字模板”统计，才能识别页眉/页脚。
                    key = _boilerplate_key(normalized)
                    occurrences.setdefault(key, []).append(
                        (payload.page_no, line.bbox[1])
                    )

        result: set[str] = set()
        for key, items in occurrences.items():
            pages = {page_no for page_no, _ in items}
            if len(pages) < min_pages:
                continue
            ys = [y for _, y in items]
            if max(ys) - min(ys) <= 12:
                result.add(key)
        return result

    def _build_page_blocks(
        self,
        payload: _PdfPage,
        *,
        body_size: float,
        boilerplate: set[str],
        doc_id: UUID,
        start_order: int,
        headings: HeadingStack,
        skip_toc: bool = False,
        document_title: tuple[
            int, tuple[float, float, float, float], str
        ] | None = None,
    ) -> tuple[list[ContentBlock], bool, int]:
        """把一页的文本块/表格按阅读顺序转成 ``ContentBlock``。

        ``headings`` 是跨页共享的标题栈；本页标题会更新栈，后续页继续沿用。
        返回本页块、是否含表格、过滤行数。
        """
        entries: list[tuple[tuple[float, float, float, float], str, object]] = []
        for block in payload.blocks:
            # 与表格区域重叠的文本块会由表格结构表达，跳过以避免重复。
            if any(
                _overlap_ratio(block.bbox, table.bbox) >= 0.5
                for table in payload.tables
            ):
                continue
            entries.append((block.bbox, "text", block))
        for table in payload.tables:
            entries.append((table.bbox, "table", table))
        entries = _page_entries_in_reading_order(entries, payload.width)

        # 文档标题由调用方在第一页检测后传入；只有第一页需要识别标题块，
        # 后续页只需要知道“已有文档标题”以便把章节钳制到二级。
        title_info = (
            (document_title[1], document_title[2])
            if document_title is not None
            and payload.page_no == document_title[0]
            else None
        )
        title_bbox = title_info[0] if title_info is not None else None
        if title_info is not None:
            # 提前入栈，保证标题块之前/之后的正文都能拿到文档标题路径。
            headings.apply(1, title_info[1])

        blocks: list[ContentBlock] = []
        order = start_order
        dropped = 0
        has_table = False
        # 图片说明常被 PDF 拆成多个文本块；记录上一个图注块用于吸收续行。
        caption_context: tuple[
            float, str, tuple[float, float, float, float]
        ] | None = None

        for _bbox, kind, item in entries:
            if skip_toc and kind == "text":
                # 目录页整页剔除；页码/章节名不是可回答知识。
                continue
            if kind == "table":
                caption_context = None
                table = cast(Any, item)
                structured = make_table(table.rows[0], table.rows[1:])
                current_heading, current_path = headings.current()
                blocks.append(
                    make_block(
                        doc_id=doc_id,
                        page_no=payload.page_no,
                        order=order,
                        block_type=BlockType.TABLE,
                        text=structured.markdown or "",
                        parser_id=self.parser_id,
                        table=structured,
                        bbox=list(_bbox),
                        heading_text=current_heading,
                        heading_path=current_path,
                    )
                )
                has_table = True
                order += 1
                continue

            block = cast(Any, item)
            kept_lines = []
            for line in block.lines:
                normalized = _normalize_line(line.text)
                if normalized and _boilerplate_key(normalized) in boilerplate:
                    dropped += 1
                    continue
                if line.ocr:
                    # OCR 行的坐标是合成的，不做页边页码判断。
                    kept_lines.append(line)
                    continue
                if self._is_margin_page_number(line, payload.height):
                    dropped += 1
                    continue
                if _is_page_furniture(line, payload.height):
                    dropped += 1
                    continue
                kept_lines.append(line)
            if not kept_lines:
                continue

            is_title = title_bbox is not None and _bbox_matches(_bbox, title_bbox)
            if not is_title:
                # 图注整块过滤：块首就是“图1/Figure 1”的说明不是知识正文。
                if _is_figure_caption(
                    kept_lines[0].text,
                    image_count=payload.image_count,
                ):
                    payload.dropped_captions += 1
                    caption_context = (
                        max(line.size for line in kept_lines),
                        kept_lines[0].font or "",
                        _bbox,
                    )
                    continue
                if caption_context is not None and _caption_continues(
                    kept_lines, _bbox, caption_context
                ):
                    payload.dropped_captions += 1
                    caption_context = (
                        caption_context[0],
                        caption_context[1],
                        _bbox,
                    )
                    continue
                caption_context = None
                # 混合块里的图注行 / 面板字母行单独剔除，保留正文行。
                filtered_lines = []
                for line in kept_lines:
                    if (
                        _is_figure_caption(
                            line.text, image_count=payload.image_count
                        )
                        or _PANEL_CAPTION_RE.match(line.text)
                        or (
                            payload.image_count > 0
                            and len(line.text.strip()) <= 12
                            and _PANEL_LABEL_RE.match(line.text)
                        )
                    ):
                        payload.dropped_captions += 1
                        continue
                    filtered_lines.append(line)
                if not filtered_lines:
                    continue
                kept_lines = filtered_lines

            text = join_wrapped_lines([line.text for line in kept_lines])
            if not text:
                continue
            # 目录行/单字母拆分页眉不是知识正文。
            if _is_toc_line(text) or (
                len(text) <= 100 and _looks_letter_spaced(text)
            ):
                dropped += 1
                continue
            # 页眉区域（如“病例报告篇”“小动物病例报告分享”）不是章节标题。
            if (
                not is_title
                and _bbox[3] <= payload.height * 0.12
                and len(text) <= 30
            ):
                dropped += 1
                continue
            # 封面/章节扉页重复印刷的文档标题不重复入栈。
            if (
                not is_title
                and document_title is not None
                and _title_key(text) == _title_key(document_title[2])
            ):
                dropped += 1
                continue
            if (
                payload.image_count > 0
                and len(text) <= 12
                and _PANEL_LABEL_RE.match(text)
            ):
                payload.dropped_captions += 1
                continue
            is_ocr = any(line.ocr for line in kept_lines)
            fonts = {line.font for line in kept_lines if line.font}
            all_heading_font = bool(fonts) and all(
                _font_is_heading(font) for font in fonts
            )
            max_size = max(line.size for line in kept_lines)
            min_size = min(line.size for line in kept_lines)
            block_meta = {"ocr": True} if is_ocr else {}

            if is_title:
                path_text, path = headings.apply(1, text)
                blocks.append(
                    make_block(
                        doc_id=doc_id,
                        page_no=payload.page_no,
                        order=order,
                        block_type=BlockType.HEADING,
                        text=text,
                        parser_id=self.parser_id,
                        heading_level=1,
                        heading_text=path_text,
                        heading_path=path,
                        bbox=list(_bbox),
                        confidence=payload.ocr_confidence if is_ocr else None,
                        metadata=block_meta,
                    )
                )
                order += 1
                continue

            # 页首作者、单位、来源说明、副标题等元数据：不是章节标题，
            # 也不应进入切片（否则会出现“杨嘉丽”这种 3 token chunk）。
            if _is_author_metadata(
                text,
                page_no=payload.page_no,
                bbox=_bbox,
                page_height=payload.height,
                body_size=body_size,
                title_bbox=title_bbox,
                fonts=fonts,
                max_size=max_size,
            ):
                payload.dropped_metadata += 1
                continue

            # “2.2 培养特性 | 正文……”同块：拆成标题 + 正文，避免正文
            # 被写进 heading_path、后续 chunk 只剩很短的一句。
            inline_numbered = _split_inline_numbered_heading(kept_lines)
            if inline_numbered is not None:
                heading_text, body_text, numbered_level = inline_numbered
                path_text, path = headings.apply(numbered_level, heading_text)
                blocks.append(
                    make_block(
                        doc_id=doc_id,
                        page_no=payload.page_no,
                        order=order,
                        block_type=BlockType.HEADING,
                        text=heading_text,
                        parser_id=self.parser_id,
                        heading_level=numbered_level,
                        heading_text=path_text,
                        heading_path=path,
                        bbox=list(_bbox),
                        confidence=payload.ocr_confidence if is_ocr else None,
                        metadata=block_meta,
                    )
                )
                order += 1
                blocks.append(
                    make_block(
                        doc_id=doc_id,
                        page_no=payload.page_no,
                        order=order,
                        block_type=BlockType.PARAGRAPH,
                        text=body_text,
                        parser_id=self.parser_id,
                        heading_text=path_text,
                        heading_path=path,
                        bbox=list(_bbox),
                        confidence=payload.ocr_confidence if is_ocr else None,
                        metadata=block_meta,
                    )
                )
                order += 1
                continue

            # “前　言：正文……”同块：拆成二级标题 + 正文段落。
            inline_section = _split_inline_section_label(text)
            if inline_section is not None:
                label, body = inline_section
                path_text, path = headings.apply(2, label)
                blocks.append(
                    make_block(
                        doc_id=doc_id,
                        page_no=payload.page_no,
                        order=order,
                        block_type=BlockType.HEADING,
                        text=label,
                        parser_id=self.parser_id,
                        heading_level=2,
                        heading_text=path_text,
                        heading_path=path,
                        bbox=list(_bbox),
                        confidence=payload.ocr_confidence if is_ocr else None,
                        metadata=block_meta,
                    )
                )
                order += 1
                blocks.append(
                    make_block(
                        doc_id=doc_id,
                        page_no=payload.page_no,
                        order=order,
                        block_type=BlockType.PARAGRAPH,
                        text=body,
                        parser_id=self.parser_id,
                        heading_text=path_text,
                        heading_path=path,
                        bbox=list(_bbox),
                        confidence=payload.ocr_confidence if is_ocr else None,
                        metadata=block_meta,
                    )
                )
                order += 1
                continue

            # “ABSTRACT 正文...” 同块：拆出摘要标题，正文作为独立段落，
            # 后续切片才能形成“文档标题 > ABSTRACT > 摘要正文”的结构。
            inline_abstract = _split_inline_abstract(text)
            if inline_abstract is not None:
                label, body = inline_abstract
                path_text, path = headings.apply(2, label)
                blocks.append(
                    make_block(
                        doc_id=doc_id,
                        page_no=payload.page_no,
                        order=order,
                        block_type=BlockType.HEADING,
                        text=label,
                        parser_id=self.parser_id,
                        heading_level=2,
                        heading_text=path_text,
                        heading_path=path,
                        bbox=list(_bbox),
                        confidence=payload.ocr_confidence if is_ocr else None,
                        metadata=block_meta,
                    )
                )
                order += 1
                blocks.append(
                    make_block(
                        doc_id=doc_id,
                        page_no=payload.page_no,
                        order=order,
                        block_type=BlockType.PARAGRAPH,
                        text=body,
                        parser_id=self.parser_id,
                        heading_text=path_text,
                        heading_path=path,
                        bbox=list(_bbox),
                        confidence=payload.ocr_confidence if is_ocr else None,
                        metadata=block_meta,
                    )
                )
                order += 1
                continue
            level = self._heading_level(
                text,
                max_size=max_size,
                # 多行块只有“所有行都明显大于正文”才按字号当标题；一行大、
                # 其余是正文的混合块通常是正文里的强调/上标，不能整块变标题。
                min_size=min_size,
                line_count=len(kept_lines),
                body_size=body_size,
                all_bold=all(line.bold for line in kept_lines),
                all_heading_font=all_heading_font,
                bbox=_bbox,
                page_width=payload.width,
            )
            if level is not None:
                # 文档标题占一级；正文中的章节至少从二级开始，避免多个 H1。
                if document_title is not None:
                    level = max(level, 2)
                path_text, path = headings.apply(level, text)
                blocks.append(
                    make_block(
                        doc_id=doc_id,
                        page_no=payload.page_no,
                        order=order,
                        block_type=BlockType.HEADING,
                        text=text,
                        parser_id=self.parser_id,
                        heading_level=level,
                        heading_text=path_text,
                        heading_path=path,
                        bbox=list(_bbox),
                        confidence=payload.ocr_confidence if is_ocr else None,
                        metadata=block_meta,
                    )
                )
            else:
                current_heading, current_path = headings.current()
                blocks.append(
                    make_block(
                        doc_id=doc_id,
                        page_no=payload.page_no,
                        order=order,
                        block_type=BlockType.PARAGRAPH,
                        text=text,
                        parser_id=self.parser_id,
                        heading_text=current_heading,
                        heading_path=current_path,
                        bbox=list(_bbox),
                        confidence=payload.ocr_confidence if is_ocr else None,
                        metadata=block_meta,
                    )
                )
            order += 1

        return blocks, has_table, dropped

    @staticmethod
    def _heading_level(
        text: str,
        *,
        max_size: float,
        min_size: float,
        line_count: int,
        body_size: float,
        all_bold: bool,
        all_heading_font: bool = False,
        bbox: tuple[float, float, float, float] | None = None,
        page_width: float | None = None,
    ) -> int | None:
        """综合字号比例、字体族、编号习惯和粗体判断标题级别。

        中文 PDF 的标题经常与正文同字号，只靠字号识别不到；这里额外使用
        字体族（黑体/兰亭黑 vs 书宋/宋体）与编号+小节关键词兜底。

        关键保守规则：

        - 含句末/分句标点（。！；）的块一定是正文，不能因为字号/粗体变标题；
        - 多行块用 ``min_size`` 判断字号标题，只有所有行都大于正文才成立，
          避免正文中夹一个上标/强调词就让整段变标题；
        - 字体/粗体推断额外限制长度，正文长段落即使是粗体也不当标题；
        - “1. 列表项”与“1. 病史”都带编号，只有标题字体/加粗/更大字号或
          小节关键词时才按标题处理。
        """
        value = (text or "").strip()
        if not value or len(value) > 120:
            return None
        # 单字母拆分标题/页眉、目录页标题、列表/续行片段都不是章节标题。
        if _looks_letter_spaced(value):
            return None
        if _URL_RE.search(value) or _CONTINUED_RE.match(value):
            return None
        if _TIME_PREFIX_RE.match(value):
            return None
        # "It is important" 这类句子片段不是标题。
        if (
            not re.search(r"[一-鿿]", value)
            and re.match(r"^(?:It|This|That|There|These|Those)\b", value, re.I)
            and re.search(
                r"\b(?:is|are|was|were|has|have|had|can|could|should|would|"
                r"may|might|must|will|do|does|did)\b",
                value,
                re.I,
            )
        ):
            return None
        # 表格表头/化验值：含中文的编号标题不参与数字比例判断；
        # 期刊 "1 | Introduction" 这类 pipe 标题单独保留。
        if not _NUMBERED_PIPE_HEADING.match(value) and not re.search(
            r"[一-鿿]", value
        ):
            if _numeric_token_ratio(value) >= 0.5:
                return None
            if (
                _looks_like_table_fragment(value)
                and _numbered_heading_level(value) is None
            ):
                return None
        compact_value = re.sub(r"\s+", "", value)
        if re.match(r"^\d{0,3}(目录|contents|tableofcontents)$", compact_value, re.I):
            return None
        if value.endswith(("-", "–", "—")):
            return None
        if value.startswith(("<", ">", "•", "·")):
            return None
        # 表说明（表1/Table 1）保留为正文/表格文本，但不能更新标题栈。
        if _TABLE_CAPTION_RE.match(value):
            return None
        # 括号不配对通常是正文句子被 PDF 换行拆开（如“…结果如下（图”
        # 与下一行“中左侧）”），不能因为行首编号或黑体就当章节标题。
        if (
            value.count("（") != value.count("）")
            or value.count("(") != value.count(")")
        ):
            return None
        # 表头被重复拼成“穴位主治穴位主治”时不是章节标题。
        if len(value) <= 20 and re.fullmatch(r"(.{2,8})\1", value):
            return None
        # 正文句号/感叹号/分号是强信号；真正的标题偶尔带问号，但很少带这些。
        if _SENTENCE_PUNCT.search(value):
            return None
        # 含逗号且不是编号标题的短块更像表格片段/正文（如化验值）。
        if ("，" in value or "," in value) and len(value) > 10:
            if _numbered_heading_level(value) is None:
                return None
        # 英文正文片段常以句号/冒号结尾；“1. Introduction.”之类保留。
        if value.endswith(".") and _numbered_heading_level(value) is None:
            return None
        # 独立的小节标签（“前　言：”或“前言”）归一化后匹配白名单。
        bare = re.sub(r"\s+", "", value).rstrip("：:").lower()
        if len(value) <= 20 and bare in _SECTION_LABEL_HEADINGS:
            letters = [char for char in value if char.isalpha()]
            if (
                letters
                and all(char.isascii() for char in letters)
                and value.islower()
            ):
                return None
            return 2
        # 行尾逗号/顿号是正文被 PDF 分块的典型形态，不是标题。
        if value.endswith(("，", ",", "、", "：", ":")):
            return None
        # 长行即使没有句末标点也更像正文；标题通常短于 80 字符。
        if len(value) > 80 and ("，" in value or "," in value):
            return None
        # 学术小节标题（References/Funding/…）经常比正文小，优先用白名单兜底。
        label_level = _section_label_level(value)
        if label_level is not None:
            return label_level
        size_for_font = max_size if line_count <= 1 else min_size
        ratio = size_for_font / body_size if body_size > 0 else 1.0
        font_level: int | None = None
        if ratio >= 2.2:
            font_level = 1
        elif ratio >= 1.6:
            font_level = 2
        elif ratio >= 1.35:
            font_level = 3
        elif ratio >= 1.15:
            font_level = 4

        inferred = _numbered_heading_level(value) or heading_level_from_text(value)
        # 英文标题通常首字母大写；小写开头的短句更可能是正文/图注。
        if inferred is None and value[0].isascii() and value[0].islower():
            return None
        # “1. 病史”与“1. 患犬出现呕吐”都符合编号形态；带空格的单级编号
        # 需要标题字体/加粗/更大字号/小节关键词之一才接受。
        space_after_dot = bool(re.match(r"^\s*\d{1,2}\.\s+", value))
        if (
            inferred == 2
            and space_after_dot
            and not (
                all_heading_font
                or all_bold
                or ratio >= 1.05
                or _has_section_keyword(value)
            )
        ):
            inferred = None
        if inferred is not None and font_level is not None:
            return min(inferred, font_level)
        if inferred is not None:
            return inferred
        # 居中、黑体、略大于正文的短行通常是药典里的药品/章节大标题，
        # 提升到二级，避免被当成更深的小节。
        centered = False
        if bbox is not None and page_width:
            center = (bbox[0] + bbox[2]) / 2
            centered = abs(center - page_width / 2) <= page_width * 0.12
        if centered and all_heading_font and ratio >= 1.2 and len(value) <= 40:
            return 2
        if font_level is not None and len(value) <= 80:
            return font_level
        # 与正文同字号的黑体/兰亭黑短行：中文报告里最常见的“隐性标题”。
        # 略小于正文的 9.2–9.5pt 黑体行多是段内强调/小标签（如“红尿（…）”），
        # 不能当章节标题；要求不小于正文的 98%。
        if all_heading_font and ratio >= 0.98 and len(value) <= 60:
            return 2
        if all_bold and ratio >= 1.05 and len(value) <= 60:
            return 3
        return None

    @staticmethod
    def _is_margin_page_number(line: _PdfLine, page_height: float) -> bool:
        """只把页边区域内形如页码的短行当页码，避免误删正文数字。"""
        in_margin = line.bbox[1] < page_height * 0.08 or line.bbox[3] > page_height * 0.92
        return in_margin and bool(_PAGE_NUMBER.match(line.text))

    # ——————————————————— pdfplumber fallback ———————————————————

    async def _parse_pdfplumber(
        self,
        source: DocumentSource,
        profile: DocumentProfile,
        *,
        doc_id: UUID,
        kb_id: UUID,
        data: bytes,
    ) -> ParsedDocument:
        """PyMuPDF 不可用时的退路：pdfplumber 抽文本和带线表格。"""
        try:
            import pdfplumber  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RagError(
                RagErrorCode.INVALID_CONFIG,
                "未安装 PDF parser 依赖（PyMuPDF 或 pdfplumber）",
                details={"parser_id": self.parser_id},
            ) from exc

        blocks: list[ContentBlock] = []
        pages: list[Page] = []
        page_count = 0
        pages_with_text = 0
        has_tables = False
        headings = HeadingStack()
        try:
            with pdfplumber.open(__import__("io").BytesIO(data)) as document:
                page_count = len(document.pages)
                for page_index, page in enumerate(document.pages, start=1):
                    page_blocks: list[ContentBlock] = []
                    text = (page.extract_text() or "").strip()
                    heading_text, heading_path = headings.current()
                    if text:
                        paragraphs = [
                            item.strip()
                            for item in re.split(r"\n\s*\n", text)
                            if item.strip()
                        ] or [text]
                        for paragraph in paragraphs:
                            level = (
                                heading_level_from_text(paragraph)
                                if is_heading_text(paragraph)
                                else None
                            )
                            if level is not None:
                                path_text, path = headings.apply(level, paragraph)
                                heading_text, heading_path = path_text, path
                            else:
                                heading_text, heading_path = headings.current()
                            page_blocks.append(
                                make_block(
                                    doc_id=doc_id,
                                    page_no=page_index,
                                    order=len(blocks) + len(page_blocks),
                                    block_type=(
                                        BlockType.HEADING
                                        if level is not None
                                        else BlockType.PARAGRAPH
                                    ),
                                    text=paragraph,
                                    parser_id=self.parser_id,
                                    heading_level=level,
                                    heading_text=heading_text,
                                    heading_path=heading_path,
                                )
                            )
                    # pdfplumber 的表格检测只在页面上有线条时执行。
                    if getattr(page, "lines", None) or getattr(page, "rects", None):
                        try:
                            raw_tables = page.extract_tables() or []
                        except Exception:
                            raw_tables = []
                        for raw_rows in raw_tables:
                            rows = [
                                [
                                    "" if cell is None else str(cell).strip()
                                    for cell in row
                                ]
                                for row in raw_rows
                                if row
                            ]
                            rows = [row for row in rows if any(cell for cell in row)]
                            if len(rows) < 2:
                                continue
                            structured = make_table(rows[0], rows[1:])
                            page_blocks.append(
                                make_block(
                                    doc_id=doc_id,
                                    page_no=page_index,
                                    order=len(blocks) + len(page_blocks),
                                    block_type=BlockType.TABLE,
                                    text=structured.markdown or "",
                                    parser_id=self.parser_id,
                                    table=structured,
                                    heading_text=heading_text,
                                    heading_path=heading_path,
                                )
                            )
                            has_tables = True
                    if page_blocks:
                        pages_with_text += 1
                    blocks.extend(page_blocks)
                    pages.append(
                        make_page(
                            page_no=page_index,
                            blocks=page_blocks,
                            width=float(getattr(page, "width", 0) or 0) or None,
                            height=float(getattr(page, "height", 0) or 0) or None,
                        )
                    )
        except Exception as exc:
            raise RagError(
                RagErrorCode.INVALID_REQUEST,
                "PDF 解析失败",
                details={"filename": source.filename, "cause": str(exc)},
            ) from exc

        coverage = pages_with_text / page_count if page_count else 0.0
        return make_parsed_document(
            source=source,
            doc_id=doc_id,
            kb_id=kb_id,
            parser_id=self.parser_id,
            parser_version=self.parser_version,
            profile=profile.model_copy(
                update={
                    "doc_type": DocumentType.PDF,
                    "page_count": page_count,
                    "has_text_layer": pages_with_text > 0,
                    "has_images": False,
                    "has_tables": has_tables,
                    "is_scanned": pages_with_text == 0,
                    "text_coverage": coverage,
                }
            ),
            blocks=blocks,
            pages=pages,
            warnings=[] if pages_with_text else ["PDF 未检测到可提取文本层（OCR/版式解析不可用或失败）"],
        )
