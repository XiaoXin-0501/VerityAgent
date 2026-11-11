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
from rag.shared.enums import (
    SUPPORTED_DOC_TYPES,
    BlockType,
    DocumentType,
)
from rag.shared.registry import ChunkerRegistry, ParserRegistry
from rag.shared.models import (
    ChunkDraft,
    DocumentSource,
    NoiseReport,
    ParsedDocument,
    ProcessedBlock,
)

from .chunkers import build_chunker_registry
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
    ) -> None:
        self.config = config or IndexerConfig()
        self.parsers = parser_registry or build_parser_registry()
        self.chunkers = chunker_registry or build_chunker_registry(self.config)

    async def process(
        self,
        source: DocumentSource,
        data: bytes,
        *,
        doc_id: UUID,
        kb_id: UUID,
        parser_hint: str | None = None,
    ) -> TextIngestionResult:
        profile = detect_profile(source)
        self._check_supported(source, profile.doc_type)

        parser: DocumentParser
        _spec, parser = self.parsers.resolve(
            source,
            profile,
            parser_hint=parser_hint,
        )
        parsed = await parser.parse(
            source,
            profile,
            doc_id=doc_id,
            kb_id=kb_id,
            data=data,
        )
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

        chunks: list[ChunkDraft] = []
        for block in parsed.blocks:
            # v1 不处理图片/图注；parser 不应产出这些块，这里再做一层保护。
            if block.block_type in (BlockType.FIGURE, BlockType.CAPTION):
                continue
            resolved = self.chunkers.resolve(
                parsed.profile.doc_type,
                block.block_type,
            )
            if resolved is None:
                # 标题等块只作为上下文，不单独成 chunk。
                continue
            _chunking_spec, chunker = resolved
            processed = ProcessedBlock(block=block, text=block.text)
            chunks.extend(
                await chunker.chunk(
                    block,
                    processed,
                    kb_id=kb_id,
                    doc_type=parsed.profile.doc_type,
                    chunking_version=self.config.chunking_version,
                    start_index=len(chunks),
                )
            )

        noise = NoiseReport()
        if self.config.quality_gate_enabled:
            chunks, quality_noise = QualityGate(self.config.quality).filter_chunks(
                chunks
            )
            noise = _merge_noise(noise, quality_noise)

        if self.config.dedup.chunk_hash:
            chunks, dedup_noise = deduplicate_chunks(chunks, self.config.dedup)
            noise = _merge_noise(noise, dedup_noise)

        warnings = list(parsed.warnings)
        warnings.extend(item.value for item in noise.warnings)
        return TextIngestionResult(
            parsed=parsed,
            chunks=chunks,
            noise=noise,
            warnings=warnings,
        )

    def _check_supported(self, source: DocumentSource, doc_type: DocumentType) -> None:
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
