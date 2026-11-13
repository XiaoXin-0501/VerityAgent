"""解析产物 manifest 存储接口与 MinIO 实现。

约定（D-58 / MinIO 设计）：

- ``parse/{kb_id}/{doc_id}/manifest.json``：解析统计、profile、artifacts；
- ``parse/{kb_id}/{doc_id}/blocks.jsonl``：每行一个 ``ContentBlock``。

manifest 是重切片/重灌/对账的入口；blocks.jsonl 保证解析结果可复现。
"""

from __future__ import annotations

from typing import Protocol

from exception import MinioObjectNotFoundError
from rag.shared.models import ArtifactRef, ContentBlock, ParseManifest
from store.minio import MinioStorage, kb_parse_prefix
from utils.serialization import json_dumps, json_loads

__all__ = [
    "ManifestStore",
    "MinioManifestStore",
    "InMemoryManifestStore",
    "manifest_key",
    "blocks_key",
]


def manifest_key(kb_id, doc_id) -> str:
    """``parse/{kb_id}/{doc_id}/manifest.json``。"""
    return f"{kb_parse_prefix(kb_id, doc_id)}manifest.json"


def blocks_key(kb_id, doc_id) -> str:
    """``parse/{kb_id}/{doc_id}/blocks.jsonl``。"""
    return f"{kb_parse_prefix(kb_id, doc_id)}blocks.jsonl"


def _serialize_blocks(blocks: list[ContentBlock]) -> bytes:
    """每行一个 ContentBlock；UTF-8、紧凑 JSON。"""
    lines = [json_dumps(block) for block in blocks]
    return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")


def _parse_blocks(data: bytes) -> list[ContentBlock]:
    """解析 blocks.jsonl；空行忽略。"""
    blocks: list[ContentBlock] = []
    for line in data.decode("utf-8").splitlines():
        text = line.strip()
        if text:
            blocks.append(ContentBlock.model_validate(json_loads(text)))
    return blocks


class ManifestStore(Protocol):
    """manifest + blocks 的读写协议。"""

    async def read(self, kb_id, doc_id) -> ParseManifest | None:
        """读取 manifest；不存在返回 None。"""
        ...

    async def read_blocks(self, kb_id, doc_id) -> list[ContentBlock]:
        """读取 blocks.jsonl；不存在返回空列表。"""
        ...

    async def write(
        self,
        manifest: ParseManifest,
        blocks: list[ContentBlock] | None = None,
    ) -> str:
        """写入 manifest（可选 blocks），返回 manifest 的 MinIO key。"""
        ...

    async def delete(self, kb_id, doc_id) -> None:
        """删除该文档全部解析产物，幂等。"""
        ...


class MinioManifestStore:
    """MinIO 解析产物存储。"""

    def __init__(self, storage: MinioStorage | None = None) -> None:
        """保存对象存储门面。"""
        self.storage = storage or MinioStorage()

    @property
    def bucket(self) -> str:
        """解析产物与源文件同属知识库 bucket。"""
        return self.storage.kb_bucket

    async def read(self, kb_id, doc_id) -> ParseManifest | None:
        """读取 manifest.json；对象不存在视为未解析。"""
        try:
            data = await self.storage.get_bytes(
                self.bucket,
                manifest_key(kb_id, doc_id),
            )
        except MinioObjectNotFoundError:
            return None
        return ParseManifest.model_validate(json_loads(data))

    async def read_blocks(self, kb_id, doc_id) -> list[ContentBlock]:
        """读取 blocks.jsonl；不存在返回空列表。"""
        try:
            data = await self.storage.get_bytes(
                self.bucket,
                blocks_key(kb_id, doc_id),
            )
        except MinioObjectNotFoundError:
            return []
        return _parse_blocks(data)

    async def write(
        self,
        manifest: ParseManifest,
        blocks: list[ContentBlock] | None = None,
    ) -> str:
        """写 blocks.jsonl 与 manifest.json，并登记 blocks artifact。"""
        artifacts = list(manifest.artifacts)
        if blocks is not None:
            payload = _serialize_blocks(blocks)
            key = blocks_key(manifest.kb_id, manifest.doc_id)
            await self.storage.put_bytes(
                self.bucket,
                key,
                payload,
                content_type="application/x-ndjson",
            )
            artifacts = [
                item for item in artifacts if item.kind != "blocks"
            ]
            artifacts.append(
                ArtifactRef(
                    kind="blocks",
                    object_key=key,
                    size_bytes=len(payload),
                    content_type="application/x-ndjson",
                )
            )

        stored = manifest.model_copy(update={"artifacts": artifacts})
        key = manifest_key(stored.kb_id, stored.doc_id)
        await self.storage.put_bytes(
            self.bucket,
            key,
            json_dumps(stored).encode("utf-8"),
            content_type="application/json",
        )
        return key

    async def delete(self, kb_id, doc_id) -> None:
        """删除 parse/{kb_id}/{doc_id}/ 前缀；幂等。"""
        await self.storage.remove_prefix(
            self.bucket,
            kb_parse_prefix(kb_id, doc_id),
        )


class InMemoryManifestStore:
    """测试/本地运行用的内存实现，语义与 MinIO 版一致。"""

    def __init__(self) -> None:
        """初始化内存表。"""
        self.manifests: dict[tuple[str, str], ParseManifest] = {}
        self.blocks: dict[tuple[str, str], list[ContentBlock]] = {}

    @staticmethod
    def _key(kb_id, doc_id) -> tuple[str, str]:
        """内存表的统一 key，统一转成字符串避免 UUID 与 str 混用。"""
        return str(kb_id), str(doc_id)

    async def read(self, kb_id, doc_id) -> ParseManifest | None:
        """读取 manifest；不存在返回 None。"""
        return self.manifests.get(self._key(kb_id, doc_id))

    async def read_blocks(self, kb_id, doc_id) -> list[ContentBlock]:
        """读取块列表；不存在返回空列表。"""
        return list(self.blocks.get(self._key(kb_id, doc_id), []))

    async def write(
        self,
        manifest: ParseManifest,
        blocks: list[ContentBlock] | None = None,
    ) -> str:
        """写入内存并返回约定 key。"""
        key = self._key(manifest.kb_id, manifest.doc_id)
        artifacts = list(manifest.artifacts)
        if blocks is not None:
            artifacts = [item for item in artifacts if item.kind != "blocks"]
            artifacts.append(
                ArtifactRef(
                    kind="blocks",
                    object_key=blocks_key(manifest.kb_id, manifest.doc_id),
                    size_bytes=len(_serialize_blocks(blocks)),
                    content_type="application/x-ndjson",
                )
            )
            self.blocks[key] = list(blocks)
        self.manifests[key] = manifest.model_copy(
            update={"artifacts": artifacts}
        )
        return manifest_key(manifest.kb_id, manifest.doc_id)

    async def delete(self, kb_id, doc_id) -> None:
        """删除内存记录；幂等。"""
        key = self._key(kb_id, doc_id)
        self.manifests.pop(key, None)
        self.blocks.pop(key, None)
