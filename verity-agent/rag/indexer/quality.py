"""入库质量闸门与去重（框架层）。

只处理已经生成的 ChunkDraft；parser/processor/chunker 仍然负责结构降噪。
"""

from __future__ import annotations

from config.rag import DedupPolicyConfig, QualityPolicyConfig
from utils.hashing import sha256_text
from rag.shared.enums import IndexingWarningCode
from rag.shared.models import ChunkDraft, NoiseReport

__all__ = ["QualityGate", "deduplicate_chunks"]


class QualityGate:
    def __init__(self, policy: QualityPolicyConfig) -> None:
        self.policy = policy

    def filter_chunks(
        self,
        chunks: list[ChunkDraft],
    ) -> tuple[list[ChunkDraft], NoiseReport]:
        kept: list[ChunkDraft] = []
        report = NoiseReport()
        for chunk in chunks:
            dropped = False
            if chunk.token_count < self.policy.min_chunk_tokens:
                report.dropped_chunks += 1
                report.warnings.append(IndexingWarningCode.CHUNK_TOO_SHORT)
                dropped = True
            elif chunk.token_count > self.policy.max_chunk_tokens:
                report.warnings.append(IndexingWarningCode.CHUNK_TRUNCATED)
                if self.policy.drop_low_quality_chunks:
                    report.dropped_chunks += 1
                    dropped = True
            if dropped:
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
