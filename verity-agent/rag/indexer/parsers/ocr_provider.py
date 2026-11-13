"""扫描件/图片 PDF 的 OCR 提供方。

支持：
- PaddleOCR（可选依赖 ``ocr`` extra）；
- Tesseract（通过 PyMuPDF 内置 OCR 接口，需要系统安装 tesseract 与中文语言包）；
- ``auto``：按可用性选择，都没有时返回 ``None``，由 PDF parser 在真正需要
  OCR 时给出明确错误；纯文本 PDF 不受影响。

所有 OCR 失败都收敛为 ``RagError``，不把第三方异常直接抛给上层。
"""

from __future__ import annotations

import importlib.util
import io
import shutil
from dataclasses import dataclass, field
from typing import Any, Protocol

from config.rag import OcrConfig
from exception import RagError, RagErrorCode

__all__ = [
    "OcrLine",
    "OcrResult",
    "OcrProvider",
    "PaddleOcrProvider",
    "TesseractOcrProvider",
    "create_ocr_provider",
]


@dataclass(slots=True)
class OcrLine:
    """一行 OCR 文本及其置信度/位置。"""

    text: str
    confidence: float | None = None
    bbox: tuple[float, float, float, float] | None = None


@dataclass(slots=True)
class OcrResult:
    """一页的 OCR 结果。"""

    lines: list[OcrLine] = field(default_factory=list)

    @property
    def text(self) -> str:
        """按行拼接的纯文本。"""
        return "\n".join(line.text for line in self.lines if line.text).strip()

    @property
    def confidence(self) -> float | None:
        """所有行的最低置信度；没有置信度信息时返回 ``None``。"""
        values = [
            line.confidence
            for line in self.lines
            if line.confidence is not None
        ]
        return min(values) if values else None


class OcrProvider(Protocol):
    """OCR 提供方协议；``page`` 为 PyMuPDF Page。"""

    def recognize(
        self,
        page: Any,
        *,
        page_no: int,
        dpi: int,
    ) -> OcrResult:
        """识别一页；失败时抛 ``RagError``。"""
        ...


class PaddleOcrProvider:
    """PaddleOCR 提供方（可选依赖，体积较大）。"""

    def __init__(self, language: str = "ch") -> None:
        """保存 PaddleOCR 语言参数，引擎懒加载。"""
        self.language = language
        self._engine = None

    def _get_engine(self):
        """懒加载 PaddleOCR，避免文本 PDF 也承担导入成本。"""
        if self._engine is not None:
            return self._engine
        try:
            from paddleocr import PaddleOCR  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RagError(
                RagErrorCode.INVALID_CONFIG,
                "未安装 PaddleOCR，无法执行扫描件 OCR",
                details={"provider": "paddle"},
            ) from exc
        try:
            self._engine = PaddleOCR(
                use_angle_cls=True,
                lang=self.language,
                show_log=False,
            )
        except TypeError:
            # PaddleOCR 3.x 参数有变化，去掉旧参数重试。
            self._engine = PaddleOCR(lang=self.language)
        return self._engine

    def recognize(
        self,
        page: Any,
        *,
        page_no: int,
        dpi: int,
    ) -> OcrResult:
        """渲染页面为图片后交给 PaddleOCR。"""
        try:
            import numpy as np  # type: ignore[import-not-found]
            from PIL import Image  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RagError(
                RagErrorCode.INVALID_CONFIG,
                "PaddleOCR 需要 numpy 与 pillow，请安装 ocr 依赖组",
                details={"provider": "paddle"},
            ) from exc

        pixmap = page.get_pixmap(dpi=dpi, alpha=False)
        image = np.array(
            Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
        )
        result = self._get_engine().ocr(image, cls=True)
        lines: list[OcrLine] = []
        for page_result in result or []:
            for item in page_result or []:
                if len(item) < 2 or not isinstance(item[1], (list, tuple)):
                    continue
                text = str(item[1][0]).strip()
                if not text:
                    continue
                try:
                    confidence = float(item[1][1])
                except (TypeError, ValueError):
                    confidence = None
                lines.append(OcrLine(text=text, confidence=confidence))
        return OcrResult(lines=lines)


class TesseractOcrProvider:
    """Tesseract 提供方：使用 PyMuPDF 内置 OCR 接口。"""

    def __init__(self, language: str = "chi_sim+eng") -> None:
        """保存 Tesseract 语言包配置。"""
        self.language = language

    def recognize(
        self,
        page: Any,
        *,
        page_no: int,
        dpi: int,
    ) -> OcrResult:
        """调用 PyMuPDF 的 Tesseract OCR；失败时给安装提示。"""
        try:
            textpage = page.get_textpage_ocr(
                language=self.language,
                dpi=dpi,
            )
            text = page.get_text("text", textpage=textpage)
        except Exception as exc:
            raise RagError(
                RagErrorCode.INVALID_CONFIG,
                "Tesseract OCR 不可用，请安装 tesseract 及 chi_sim 语言包",
                details={
                    "provider": "tesseract",
                    "language": self.language,
                    "cause": str(exc),
                },
            ) from exc
        lines = [
            OcrLine(text=line.strip())
            for line in str(text or "").splitlines()
            if line.strip()
        ]
        return OcrResult(lines=lines)


def _paddle_available() -> bool:
    """判断 PaddleOCR 是否可导入，不实际加载模型。"""
    try:
        return importlib.util.find_spec("paddleocr") is not None
    except (ImportError, ValueError):
        return False


def _tesseract_available() -> bool:
    """判断系统是否安装了 tesseract 可执行文件。"""
    return shutil.which("tesseract") is not None


def create_ocr_provider(config: OcrConfig) -> OcrProvider | None:
    """按配置选择 OCR 提供方；不可用时返回 ``None``。"""
    if not config.enabled:
        return None
    provider = (config.provider or "auto").strip().lower()
    if provider == "none":
        return None
    if provider == "paddle":
        return PaddleOcrProvider(config.paddle_lang)
    if provider == "tesseract":
        return TesseractOcrProvider(config.language)
    if provider != "auto":
        raise RagError(
            RagErrorCode.INVALID_CONFIG,
            f"未知的 OCR provider: {config.provider}",
            details={"provider": config.provider},
        )
    if _paddle_available():
        return PaddleOcrProvider(config.paddle_lang)
    if _tesseract_available():
        return TesseractOcrProvider(config.language)
    return None
