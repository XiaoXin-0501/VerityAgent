"""图片 PDF 链路的 LLM 扩展点（默认不启用）。

这里只定义协议，不实现任何模型调用；后续可以：
- 用 ``VlmLayoutEngine`` 替换 MinerU/RapidOCR；
- 用 ``LayoutPostProcessor`` 修复 OCR、规范化标题；
- 用 ``DocumentEnhancer`` 生成文档描述/摘要。

主链路不会导入本模块的模型客户端，因此默认路径不产生 LLM 调用。
"""

from __future__ import annotations

from typing import Protocol

from config.rag import ImagePdfConfig
from rag.shared.models import ParsedDocument

from .models import LayoutDocument

__all__ = [
    "LayoutPostProcessor",
    "DocumentEnhancer",
    "VlmLayoutEngine",
]


class LayoutPostProcessor(Protocol):
    """版式结果后处理器（可接 LLM）。"""

    name: str

    def process(
        self,
        document: LayoutDocument,
        config: ImagePdfConfig,
    ) -> LayoutDocument:
        """返回修正后的 LayoutDocument。"""
        ...


class DocumentEnhancer(Protocol):
    """文档级富化器（可接 LLM）。"""

    name: str

    def enhance(
        self,
        parsed: ParsedDocument,
        config: ImagePdfConfig,
    ) -> ParsedDocument:
        """返回增强后的 ParsedDocument。"""
        ...


class VlmLayoutEngine(Protocol):
    """视觉大模型版式引擎的预留协议。

    实现者应与 ``image_pdf.engines.base.LayoutEngine`` 保持相同签名，
    由配置 ``engine=vlm`` 选择；默认链路不会实例化。
    """

    name: str

    def available(self) -> bool: ...
