"""入库质量闸门与去重（框架层）。

只处理已经生成的 ChunkDraft；parser/processor/chunker 仍然负责结构降噪。
"""

from __future__ import annotations

from config.rag import DedupPolicyConfig, QualityPolicyConfig
from utils.hashing import sha256_text
from rag.shared.enums import IndexingWarningCode, QualityFlag
from rag.shared.models import ChunkDraft, NoiseReport

__all__ = [
    "QUALITY_PENALTIES",
    "QualityGate",
    "deduplicate_chunks",
    "quality_score",
]

# 质量标记 → 扣分（内置表）。
# 未声明的 flag 按 0 分处理，向前兼容新增枚举。
QUALITY_PENALTIES: dict[QualityFlag, float] = {
    QualityFlag.LOW_OCR: 0.25,
    QualityFlag.LOW_TABLE: 0.2,
    QualityFlag.LOW_TEXT_COVERAGE: 0.3,
    QualityFlag.DUPLICATE: 0.5,
    QualityFlag.TRUNCATED: 0.1,
    QualityFlag.BOILERPLATE: 0.5,
    QualityFlag.CODE_PARSE_FALLBACK: 0.1,
    QualityFlag.EMBEDDING_TRUNCATED: 0.05,
}


def quality_score(flags: set[QualityFlag]) -> float:
    """把质量标记折算成 0~1 分：``clamp(1 - Σ扣分, 0, 1)``。

    多个标记叠加扣分，不取最大值；结果保留 4 位小数，保证稳定。
    """
    total = sum(QUALITY_PENALTIES.get(flag, 0.0) for flag in flags)
    return round(max(0.0, min(1.0, 1.0 - total)), 4)


class QualityGate:
    """切片后的质量闸门：短/超长/低质量标记，按策略决定是否丢弃。"""

    def __init__(
        self,
        policy: QualityPolicyConfig,
        *,
        max_chunk_tokens: int,
    ) -> None:
        """``max_chunk_tokens`` 来自 ``IndexerConfig``，是唯一硬上限来源。"""
        self.policy = policy
        self.max_chunk_tokens = max_chunk_tokens

    def filter_chunks(
        self,
        chunks: list[ChunkDraft],
    ) -> tuple[list[ChunkDraft], NoiseReport]:
        """过滤 chunk；默认保留并标记短块/低质量块，可配置为丢弃。"""
        kept: list[ChunkDraft] = []
        report = NoiseReport()
        for chunk in chunks:
            if chunk.token_count < self.policy.min_chunk_tokens:
                # 短问答、列表项在合并后仍可能偏短，默认保留并标记，避免丢内容。
                report.low_quality_chunks += 1
                report.warnings.append(IndexingWarningCode.CHUNK_TOO_SHORT)
                if self.policy.drop_low_quality_chunks:
                    report.dropped_chunks += 1
                    continue
            elif chunk.token_count > self.max_chunk_tokens:
                # 切片器已经按 max_tokens 切分；命中这里说明单行/单表格超长。
                report.warnings.append(IndexingWarningCode.CHUNK_TRUNCATED)
                if self.policy.drop_low_quality_chunks:
                    report.dropped_chunks += 1
                    continue
            if chunk.quality_flags and self.policy.drop_low_quality_chunks:
                report.low_quality_chunks += 1
                report.dropped_chunks += 1
                continue
            kept.append(chunk)
        return kept, report


def deduplicate_chunks(
    chunks: list[ChunkDraft],
    policy: DedupPolicyConfig,
) -> tuple[list[ChunkDraft], NoiseReport]:
    """按正文 SHA256 做 chunk 级精确去重；保留首次出现的 chunk。"""
    if not policy.chunk_hash:
        return chunks, NoiseReport()
    seen: set[str] = set()
    kept: list[ChunkDraft] = []
    report = NoiseReport()
    for chunk in chunks:
        digest = sha256_text(chunk.text)
        if digest in seen:
            report.duplicate_chunks += 1
            report.warnings.append(IndexingWarningCode.DUPLICATE_DROPPED)
            continue
        seen.add(digest)
        kept.append(chunk)
    return kept, report
