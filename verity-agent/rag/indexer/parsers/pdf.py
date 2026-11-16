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

# 纯页码行：12 / 12/145 / - 12 - / 第 12 页。
_PAGE_NUMBER = re.compile(
    r"^\s*(?:[-—–]\s*)?(?:第\s*)?\d{1,4}\s*(?:/\s*\d{1,4})?\s*(?:页)?\s*(?:[-—–])?\s*$"
)
# 粗体 flag（PyMuPDF span.flags bit 4）。
_BOLD_FLAG = 1 << 4

# PDF 常见的分级编号标题：
#   1.诊断 / 4.讨论      → 一级小节（挂在文档标题下）
#   1.1 病例信息         → 二级小节
#   1.3.1 血常规检查     → 三级小节
# 要求编号后紧跟中文/英文，避免把 “1. 列表项” 或表格数值误判成标题。
_NUMBERED_COMPOUND_HEADING = re.compile(
    r"^\s*\d{1,2}(?:\.\d{1,2})+\s*[一-鿿A-Za-z]"
)
_NUMBERED_SINGLE_HEADING = re.compile(
    r"^\s*\d{1,2}\.(?=[一-鿿A-Za-z])\s*"
)
# 期刊常见 “1 | Introduction”“2.1 | Histopathology” 编号格式。
_NUMBERED_PIPE_HEADING = re.compile(
    r"^\s*\d{1,2}(?:\.\d{1,2})*\s*\|\s*[A-Za-z一-鿿]"
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
    # 本页是否通过 OCR 获得文本，以及整体置信度。
    ocr_used: bool = False
    ocr_confidence: float | None = None


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
    if value in _SECTION_LABEL_HEADINGS:
        return 3
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
        return min(6, number.count(".") + 3)
    match = _NUMBERED_COMPOUND_HEADING.match(value)
    if match:
        number = value.split()[0].rstrip(".")
        return min(6, number.count(".") + 2)
    if _NUMBERED_SINGLE_HEADING.match(value):
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
        try:
            payloads: list[_PdfPage] = []
            for page_no, page in enumerate(document, start=1):
                payload = self._extract_page(page, page_no, data=data)
                if self._should_ocr(payload):
                    if (
                        self._ocr_provider is not None
                        and ocr_pages < self._ocr_config.max_pages
                    ):
                        self._apply_ocr(page, payload)
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

        blocks: list[ContentBlock] = []
        pages: list[Page] = []
        dropped_lines = 0
        has_tables = False
        has_images = any(item.image_count > 0 for item in payloads)
        # 标题栈跨页延续：下一页首段仍能拿到完整的标题路径。
        headings = HeadingStack()

        for payload in payloads:
            page_blocks, page_tables, page_dropped = self._build_page_blocks(
                payload,
                body_size=body_size,
                boilerplate=boilerplate,
                doc_id=doc_id,
                start_order=len(blocks),
                headings=headings,
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

        warnings: list[str] = list(image_pipeline_warnings)
        if dropped_lines:
            warnings.append(f"已过滤跨页重复的水印/页眉页脚 {dropped_lines} 行")
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
                text = "".join(span.get("text", "") for span in spans)
                if not text.strip():
                    continue
                size = max((float(span.get("size", 0.0)) for span in spans), default=0.0)
                bold = bool(spans) and all(
                    int(span.get("flags", 0)) & _BOLD_FLAG for span in spans
                )
                line = _PdfLine(
                    text=text,
                    bbox=_bbox_tuple(raw_line.get("bbox", (0, 0, 0, 0))),
                    size=size,
                    bold=bold,
                )
                block.lines.append(line)
                # 用字符数加权统计字号，后续据此判断文档正文字号。
                for span in spans:
                    payload.size_chars[round(size, 1)] += len(span.get("text", ""))
            if block.lines:
                payload.blocks.append(block)

        payload.tables = self._extract_tables(page, page_no=page_no, data=data)
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
        if not self._has_table_structure(page, drawings=drawings):
            return []
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
            if not rows:
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
    ) -> list[_PdfTable]:
        """用 pdfplumber 抽取一页表格；依赖缺失或失败时返回空列表。"""
        try:
            import pdfplumber  # type: ignore[import-not-found]

            with pdfplumber.open(__import__("io").BytesIO(data)) as document:
                if page_no < 1 or page_no > len(document.pages):
                    return []
                page = document.pages[page_no - 1]
                tables: list[_PdfTable] = []
                for raw_table in page.find_tables():
                    rows = self._normalize_table_rows(raw_table.extract())
                    if not rows:
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
    def _normalize_table_rows(raw_rows) -> list[list[str]]:
        """把表格二维数据清理成非空行；单行/单列视作非表格。"""
        rows = [
            ["" if cell is None else str(cell).strip() for cell in row]
            for row in (raw_rows or [])
            if row
        ]
        rows = [row for row in rows if any(cell for cell in row)]
        if len(rows) < 2 or max((len(row) for row in rows), default=0) < 2:
            return []
        return rows

    @staticmethod
    def _has_table_structure(page, *, drawings=None) -> bool:
        """判断页面是否真的存在表格网格，避免把代码框/引用框当表格。

        表格至少要满足：
        - 3 条以上水平线（上边框、行分隔、下边框）；
        - 某条水平线被 3 条以上竖线穿过（说明有列分隔线）。

        单个矩形框只有 2 横 2 竖；多个独立矩形框的每条横线也只被 2 条竖线
        穿过，因此都不会触发昂贵的 ``find_tables``。
        """
        horizontals: list[tuple[float, float, float]] = []
        verticals: list[tuple[float, float, float]] = []
        if drawings is None:
            try:
                drawings = page.get_drawings()
            except Exception:
                return False

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

        if len(horizontals) < 3 or len(verticals) < 3:
            return False
        for x0, x1, y in horizontals:
            crossings = sum(
                1
                for x, y0, y1 in verticals
                if x0 - 1 <= x <= x1 + 1 and y0 - 1 <= y <= y1 + 1
            )
            if crossings >= 3:
                return True
        return False

    @staticmethod
    def _body_font_size(payloads: list[_PdfPage]) -> float:
        """按字符数加权选择文档正文字号。"""
        counter: Counter = Counter()
        for payload in payloads:
            counter.update(payload.size_chars)
        if not counter:
            return 10.0
        return float(counter.most_common(1)[0][0])

    @staticmethod
    def _detect_boilerplate(payloads: list[_PdfPage]) -> set[str]:
        """识别跨页重复且位置稳定的短行（水印/页眉页脚）。

        只有同时满足以下条件才丢弃：
        - 归一化后长度 <= 60；
        - 出现在至少 30% 的页面（且不少于 3 页）；
        - 各页的纵坐标差异 <= 12pt。
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
                    occurrences.setdefault(normalized, []).append(
                        (payload.page_no, line.bbox[1])
                    )

        result: set[str] = set()
        for text, items in occurrences.items():
            pages = {page_no for page_no, _ in items}
            if len(pages) < min_pages:
                continue
            ys = [y for _, y in items]
            if max(ys) - min(ys) <= 12:
                result.add(text)
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

        blocks: list[ContentBlock] = []
        order = start_order
        dropped = 0
        has_table = False

        for _bbox, kind, item in entries:
            if kind == "table":
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
                if normalized and normalized in boilerplate:
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

            text = join_wrapped_lines([line.text for line in kept_lines])
            if not text:
                continue
            is_ocr = any(line.ocr for line in kept_lines)
            # “ABSTRACT 正文...” 同块：拆出摘要标题，正文作为独立段落，
            # 后续切片才能形成“文档标题 > ABSTRACT > 摘要正文”的结构。
            inline_abstract = _split_inline_abstract(text)
            if inline_abstract is not None:
                label, body = inline_abstract
                path_text, path = headings.apply(3, label)
                blocks.append(
                    make_block(
                        doc_id=doc_id,
                        page_no=payload.page_no,
                        order=order,
                        block_type=BlockType.HEADING,
                        text=label,
                        parser_id=self.parser_id,
                        heading_level=3,
                        heading_text=path_text,
                        heading_path=path,
                        bbox=list(_bbox),
                        confidence=payload.ocr_confidence if is_ocr else None,
                        metadata={"ocr": True} if is_ocr else {},
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
                        metadata={"ocr": True} if is_ocr else {},
                    )
                )
                order += 1
                continue
            level = self._heading_level(
                text,
                max_size=max(line.size for line in kept_lines),
                body_size=body_size,
                all_bold=all(line.bold for line in kept_lines),
            )
            if level is not None:
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
                        metadata={"ocr": True} if is_ocr else {},
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
                        metadata={"ocr": True} if is_ocr else {},
                    )
                )
            order += 1

        return blocks, has_table, dropped

    @staticmethod
    def _heading_level(
        text: str,
        *,
        max_size: float,
        body_size: float,
        all_bold: bool,
    ) -> int | None:
        """综合字号比例、编号习惯和粗体判断标题级别。

        编号级别与字号级别同时存在时取较粗的级别（数值更小），避免同一
        文档里标题层级跳变；只有编号、没有字号差异时仍按编号识别标题。
        """
        if len(text) > 160:
            return None
        # 学术小节标题（References/Funding/…）经常比正文小，优先用白名单兜底。
        label_level = _section_label_level(text)
        if label_level is not None:
            return label_level
        ratio = max_size / body_size if body_size > 0 else 1.0
        font_level: int | None = None
        if ratio >= 2.2:
            font_level = 1
        elif ratio >= 1.6:
            font_level = 2
        elif ratio >= 1.35:
            font_level = 3
        elif ratio >= 1.15:
            font_level = 4

        inferred = _numbered_heading_level(text) or heading_level_from_text(text)
        if inferred is not None and font_level is not None:
            return min(inferred, font_level)
        if inferred is not None:
            return inferred
        if font_level is not None:
            return font_level
        if all_bold and ratio >= 1.05 and len(text) <= 80:
            return 5
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
