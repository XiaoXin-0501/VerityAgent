"""图片 OCR parser（PaddleOCR 可选）。

PaddleOCR 体积较大，作为可选依赖；未安装时明确报 RagError，而不是静默返回
空文档。
"""

from __future__ import annotations

from uuid import UUID

from exception import RagError, RagErrorCode
from rag.shared.enums import BlockType, DocumentType
from rag.shared.models import DocumentProfile, DocumentSource, ParsedDocument

from ._common import make_block, make_parsed_document

__all__ = ["OcrParser"]


class OcrParser:
    parser_id = "image_paddleocr"
    parser_version = "1"

    async def parse(
        self,
        source: DocumentSource,
        profile: DocumentProfile,
        *,
        doc_id: UUID,
        kb_id: UUID,
        data: bytes,
    ) -> ParsedDocument:
        try:
            from paddleocr import PaddleOCR  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RagError(
                RagErrorCode.INVALID_CONFIG,
                "未安装 PaddleOCR，无法执行图片 OCR",
                details={"parser_id": self.parser_id},
            ) from exc
        try:
            import numpy as np  # type: ignore[import-not-found]
            from PIL import Image  # type: ignore[import-not-found]
            import io

            image = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
            engine = PaddleOCR(use_angle_cls=True, lang="ch")
            result = engine.ocr(image, cls=True)
        except Exception as exc:
            raise RagError(
                RagErrorCode.INVALID_REQUEST,
                "图片 OCR 失败",
                details={"filename": source.filename, "cause": str(exc)},
            ) from exc
        texts: list[str] = []
        confidences: list[float] = []
        for page in result or []:
            for line in page or []:
                if len(line) >= 2 and isinstance(line[1], (list, tuple)):
                    texts.append(str(line[1][0]))
                    try:
                        confidences.append(float(line[1][1]))
                    except (TypeError, ValueError):
                        pass
        text = "\n".join(texts).strip()
        blocks = (
            [
                make_block(
                    doc_id=doc_id,
                    page_no=1,
                    order=0,
                    block_type=BlockType.PARAGRAPH,
                    text=text,
                    parser_id=self.parser_id,
                    confidence=min(confidences) if confidences else None,
                )
            ]
            if text
            else []
        )
        return make_parsed_document(
            source=source,
            doc_id=doc_id,
            kb_id=kb_id,
            parser_id=self.parser_id,
            parser_version=self.parser_version,
            profile=profile.model_copy(
                update={
                    "doc_type": DocumentType.IMAGE,
                    "has_text_layer": False,
                    "is_scanned": True,
                    "page_count": 1,
                    "text_coverage": 1.0 if text else 0.0,
                    "ocr_ratio": 1.0 if text else 0.0,
                }
            ),
            blocks=blocks,
            warnings=[] if text else ["OCR 未识别到文本"],
        )
