"""扫描件/图片 PDF 的 OCR 提供方。

支持：
- RapidOCR（可选依赖 ``ocr-light`` extra）：ONNXRuntime，轻量且不依赖
  Paddle/Torch，默认 ``auto`` 优先使用；
- PaddleOCR（可选依赖 ``ocr`` extra）；
- Tesseract（通过 PyMuPDF 内置 OCR 接口，需要系统安装 tesseract 与中文语言包）；
- ``auto``：按 RapidOCR → PaddleOCR → Tesseract 的可用性选择，都没有时返回
  ``None``，由 PDF parser 在真正需要 OCR 时给出明确错误；纯文本 PDF 不受影响。

所有 OCR 失败都收敛为 ``RagError``，不把第三方异常直接抛给上层。
"""

from __future__ import annotations

import importlib.util
import io
import shutil
import threading
from dataclasses import dataclass, field
from typing import Any, Protocol

from config.rag import OcrConfig
from exception import RagError, RagErrorCode

__all__ = [
    "OcrLine",
    "OcrResult",
    "OcrProvider",
    "PaddleOcrProvider",
    "RapidOcrProvider",
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


class RapidOcrProvider:
    """RapidOCR 提供方（ONNXRuntime，无 Paddle/Torch 依赖）。"""

    def __init__(self) -> None:
        """OCR 引擎懒加载，避免纯文本 PDF 承担模型加载成本。"""
        self._engine = None
        # PDF 解析在线程池中并发执行；ONNXRuntime session 不保证多线程并发
        # 调用安全，用锁把同一 provider 的识别串行化（RLock 允许识别时懒加载）。
        self._lock = threading.RLock()

    def _get_engine(self):
        """懒加载 RapidOCR；依赖损坏时转成可解释的 ``RagError``。"""
        with self._lock:
            if self._engine is not None:
                return self._engine
            try:
                from rapidocr import RapidOCR  # type: ignore[import-not-found]
            except ImportError as exc:
                raise RagError(
                    RagErrorCode.INVALID_CONFIG,
                    "未安装 RapidOCR，无法执行扫描件 OCR",
                    details={"provider": "rapid"},
                ) from exc
            try:
                self._engine = RapidOCR()
            except Exception as exc:  # noqa: BLE001 - 第三方引擎初始化失败统一收敛
                raise RagError(
                    RagErrorCode.INVALID_CONFIG,
                    "RapidOCR 初始化失败",
                    details={"provider": "rapid", "cause": str(exc)},
                ) from exc
            return self._engine

    def recognize(
        self,
        page: Any,
        *,
        page_no: int,
        dpi: int,
    ) -> OcrResult:
        """渲染页面为图片后交给 RapidOCR。"""
        try:
            import numpy as np  # type: ignore[import-not-found]
            from PIL import Image  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RagError(
                RagErrorCode.INVALID_CONFIG,
                "RapidOCR 需要 numpy 与 pillow，请安装 ocr-light 依赖组",
                details={"provider": "rapid"},
            ) from exc

        pixmap = page.get_pixmap(dpi=dpi, alpha=False)
        image = np.array(
            Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
        )
        try:
            with self._lock:
                output = self._get_engine()(image)
        except RagError:
            raise
        except Exception as exc:  # noqa: BLE001 - 第三方识别失败统一收敛
            raise RagError(
                RagErrorCode.INTERNAL,
                "RapidOCR 识别失败",
                details={"provider": "rapid", "page": page_no, "cause": str(exc)},
            ) from exc

        # RapidOCR 3.x 返回 RapidOCROutput(txts/scores 一一对应)；没有文本时
        # txts 可能为 None。旧版本返回 [[box, (text, score)], ...]，两种都兼容。
        if hasattr(output, "txts"):
            texts = getattr(output, "txts") or []
            scores = getattr(output, "scores") or []
            lines: list[OcrLine] = []
            for index, text in enumerate(texts):
                value = str(text or "").strip()
                if not value:
                    continue
                score: float | None = None
                if index < len(scores):
                    try:
                        score = float(scores[index])
                    except (TypeError, ValueError):
                        score = None
                lines.append(OcrLine(text=value, confidence=score))
            return OcrResult(lines=lines)

        lines = []
        for item in output or []:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            body = item[1]
            if not isinstance(body, (list, tuple)) or not body:
                continue
            value = str(body[0] or "").strip()
            if not value:
                continue
            try:
                score = float(body[1]) if len(body) > 1 else None
            except (TypeError, ValueError):
                score = None
            lines.append(OcrLine(text=value, confidence=score))
        return OcrResult(lines=lines)


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


def _rapid_available() -> bool:
    """判断 RapidOCR 是否可导入，不实际加载模型。"""
    try:
        return importlib.util.find_spec("rapidocr") is not None
    except (ImportError, ValueError):
        return False


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
    if provider == "rapid":
        return RapidOcrProvider()
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
    # auto 优先轻量 ONNXRuntime 方案：PaddleOCR 在部分 Windows 环境会因
    # Torch/Paddle DLL 冲突无法导入，RapidOCR 不依赖这两者。
    if _rapid_available():
        return RapidOcrProvider()
    if _paddle_available():
        return PaddleOcrProvider(config.paddle_lang)
    if _tesseract_available():
        return TesseractOcrProvider(config.language)
    return None
