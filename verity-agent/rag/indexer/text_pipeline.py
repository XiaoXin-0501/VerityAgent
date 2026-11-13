"""v1 文本入库流水线：解析 → 纯文本切片 → 质量闸门 → 去重。

这里只做“文件字节 → 文本 chunk”这段，不写 embedding / Weaviate / Neo4j；
调用方拿到 `TextIngestionResult` 后决定如何持久化。这样既能先验证解析与
切片质量，又不会提前把存储细节耦合进来。
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from config.rag import IndexerConfig
from exception import RagError, RagErrorCode
from rag.shared.enums import SUPPORTED_DOC_TYPES, DocumentType
from rag.shared.registry import ChunkerRegistry, ParserRegistry
from rag.shared.models import (
    ChunkDraft,
    DocumentSource,
    NoiseReport,
    ParsedDocument,
)

from .chunkers import (
    SectionAwareChunker,
    build_chunker_registry,
)
from .description import DescriptionRefiner, DocumentDescriptionExtractor
from .parsers import build_parser_registry, detect_profile
from .parsers.base import DocumentParser
from .quality import QualityGate, deduplicate_chunks

__all__ = ["TextIngestionResult", "TextIngestionPipeline"]


@dataclass(slots=True)
class TextIngestionResult:
    """一次解析/切片的纯文本结果。"""

    parsed: ParsedDocument
    chunks: list[ChunkDraft]
    noise: NoiseReport
    warnings: list[str]


class TextIngestionPipeline:
    """v1 支持的 PDF/DOCX/Markdown/TXT/XLSX 文本入库流水线。"""

    def __init__(
        self,
        config: IndexerConfig | None = None,
        *,
        parser_registry: ParserRegistry | None = None,
        chunker_registry: ChunkerRegistry | None = None,
        description_refiner: DescriptionRefiner | None = None,
    ) -> None:
        self.config = config or IndexerConfig()
        self.parsers = parser_registry or build_parser_registry(self.config)
        self.chunkers = chunker_registry or build_chunker_registry(self.config)
        self.section_chunker = SectionAwareChunker(self.config, self.chunkers)
        self.descriptions = DocumentDescriptionExtractor(
            self.config,
            description_refiner,
        )

    async def process(
        self,
        source: DocumentSource,
        data: bytes,
        *,
        doc_id: UUID,
        kb_id: UUID,
        parser_hint: str | None = None,
    ) -> TextIngestionResult:
        """解析文档字节、切片、降噪并返回纯文本结果。"""
        profile = detect_profile(source)
        self._check_supported(source, profile.doc_type)

        parser: DocumentParser
        _spec, parser = self.parsers.resolve(
            source,
            profile,
            parser_hint=parser_hint,
        )
        try:
            parsed = await parser.parse(
                source,
                profile,
                doc_id=doc_id,
                kb_id=kb_id,
                data=data,
            )
        except RagError:
            raise
        except Exception as exc:
            # 解析器的第三方库异常统一收敛到 RAG 异常模块，避免泄漏原始栈。
            raise RagError(
                RagErrorCode.INTERNAL,
                "文档解析失败",
                details={
                    "filename": source.filename,
                    "parser_id": _spec.parser_id,
                    "cause": str(exc),
                },
            ) from exc
        if not parsed.blocks:
            raise RagError(
                RagErrorCode.INVALID_REQUEST,
                "文档没有可索引文本（图片/扫描件暂不处理）",
                details={"doc_type": parsed.profile.doc_type.value},
            )
        if parsed.profile.text_coverage < self.config.quality.min_text_coverage:
            raise RagError(
                RagErrorCode.INVALID_REQUEST,
                "文档文本覆盖率过低（图片/扫描件暂不处理）",
                details={
                    "doc_type": parsed.profile.doc_type.value,
                    "text_coverage": parsed.profile.text_coverage,
                    "min_text_coverage": self.config.quality.min_text_coverage,
                },
            )

        # 文档描述：规则优先；非可靠候选交给可选 LLM refiner。
        description = await self.descriptions.extract(parsed, source)
        if description.text or description.candidate:
            parsed = parsed.model_copy(
                update={"description": description.to_candidate(doc_id)}
            )

        # 按标题层级由粗到细切片：能放进一级/二级标题的内容不提前下钻。
        chunks = await self.section_chunker.chunk(
            parsed.blocks,
            kb_id=kb_id,
            doc_type=parsed.profile.doc_type,
            document_description=description.text,
        )

        noise = NoiseReport()
        if self.config.quality_gate_enabled:
            chunks, quality_noise = QualityGate(
                self.config.quality,
                max_chunk_tokens=self.config.max_chunk_tokens,
            ).filter_chunks(
                chunks
            )
            noise = _merge_noise(noise, quality_noise)

        if self.config.dedup.chunk_hash:
            chunks, dedup_noise = deduplicate_chunks(chunks, self.config.dedup)
            noise = _merge_noise(noise, dedup_noise)

        warnings = list(parsed.warnings)
        warnings.extend(description.warnings)
        warnings.extend(item.value for item in noise.warnings)
        return TextIngestionResult(
            parsed=parsed,
            chunks=chunks,
            noise=noise,
            warnings=warnings,
        )

    def _check_supported(self, source: DocumentSource, doc_type: DocumentType) -> None:
        """校验文档类型是否在 v1 支持列表和配置开关内。"""
        if doc_type not in SUPPORTED_DOC_TYPES:
            raise RagError(
                RagErrorCode.INVALID_REQUEST,
                f"v1 暂不支持该文档格式: {source.filename}",
                details={
                    "doc_type": doc_type.value,
                    "supported": sorted(item.value for item in SUPPORTED_DOC_TYPES),
                },
            )
        if self.config.enabled_doc_types:
            enabled = {item.lower() for item in self.config.enabled_doc_types}
            if doc_type.value not in enabled:
                raise RagError(
                    RagErrorCode.INVALID_REQUEST,
                    f"文档类型未启用: {doc_type.value}",
                    details={
                        "doc_type": doc_type.value,
                        "enabled": sorted(enabled),
                    },
                )


def _merge_noise(left: NoiseReport, right: NoiseReport) -> NoiseReport:
    """合并两个 ``NoiseReport`` 的计数与告警，告警去重。"""
    warnings = list(dict.fromkeys([*left.warnings, *right.warnings]))
    return NoiseReport(
        dropped_blocks=left.dropped_blocks + right.dropped_blocks,
        dropped_chunks=left.dropped_chunks + right.dropped_chunks,
        duplicate_chunks=left.duplicate_chunks + right.duplicate_chunks,
        low_quality_chunks=left.low_quality_chunks + right.low_quality_chunks,
        boilerplate_blocks=left.boilerplate_blocks + right.boilerplate_blocks,
        warnings=warnings,
        details={**left.details, **right.details},
    )
