"""RAG 确定性 id 与哈希。"""

from __future__ import annotations

from uuid import UUID, uuid5, NAMESPACE_URL

from utils.hashing import sha256_text

__all__ = ["chunk_hash", "deterministic_chunk_id", "deterministic_block_id"]


def chunk_hash(text: str) -> str:
    """规范化正文的 SHA256。"""
    return sha256_text(text)


def deterministic_block_id(
    *,
    doc_id: UUID,
    page_no: int,
    order: int,
    block_type: str,
    namespace: str = "verity:block",
) -> UUID:
    """按文档规范生成 content block 的确定性 id。"""
    key = chr(0).join(
        [namespace, str(doc_id), str(page_no), str(order), block_type]
    )
    return uuid5(NAMESPACE_URL, key)


def deterministic_chunk_id(
    *,
    kb_id: UUID,
    doc_id: UUID,
    chunk_index: int,
    content_hash: str,
    chunking_version: str,
    namespace: str = "verity:chunk",
) -> str:
    """按文档规范生成可重建的 chunk id。"""
    key = "\x00".join(
        [
            namespace,
            str(kb_id),
            str(doc_id),
            str(chunk_index),
            content_hash,
            chunking_version,
        ]
    )
    return str(uuid5(NAMESPACE_URL, key))
