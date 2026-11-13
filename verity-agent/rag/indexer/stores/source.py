"""文档源文件存储接口。

IndexingPipeline 只依赖 ``SourceStore``：从对象存储读原件字节，删除时按
doc_id 清理 source 前缀。MinIO 的 bucket/key 细节留在 ``store.minio``。
"""

from __future__ import annotations

from typing import Protocol

from rag.shared.models import DocumentSource
from store.minio import MinioStorage

__all__ = ["SourceStore", "MinioSourceStore", "InMemorySourceStore"]


class SourceStore(Protocol):
    """源文件读取/删除协议。"""

    async def read_bytes(self, source: DocumentSource) -> bytes:
        """按 ``source.object_key`` 读取原件。"""
        ...

    async def delete_document(
        self,
        kb_id,
        doc_id,
        *,
        extension: str | None = None,
    ) -> int:
        """删除文档原件；``extension`` 为空时按 source 前缀清理，幂等。"""
        ...

    async def exists(self, source: DocumentSource) -> bool:
        """原件是否存在。"""
        ...


class MinioSourceStore:
    """MinIO 源文件实现。"""

    def __init__(self, storage: MinioStorage | None = None) -> None:
        """保存对象存储门面。"""
        self.storage = storage or MinioStorage()

    async def read_bytes(self, source: DocumentSource) -> bytes:
        """读取原件；对象不存在会抛 MinioObjectNotFoundError。"""
        return await self.storage.get_bytes(
            self.storage.kb_bucket,
            source.object_key,
        )

    async def delete_document(
        self,
        kb_id,
        doc_id,
        *,
        extension: str | None = None,
    ) -> int:
        """删除原件与整个 parse 前缀（与 manifest 删除重叠，幂等）。"""
        return await self.storage.delete_kb_document(
            kb_id,
            doc_id,
            extension=(
                extension.lstrip(".") if extension else None
            ),
        )

    async def exists(self, source: DocumentSource) -> bool:
        """检查原件是否存在。"""
        return await self.storage.exists(
            self.storage.kb_bucket,
            source.object_key,
        )


class InMemorySourceStore:
    """测试/本地运行用的内存源文件存储。"""

    def __init__(self) -> None:
        """初始化 key → bytes 表。"""
        self.objects: dict[str, bytes] = {}

    async def put(self, object_key: str, data: bytes) -> None:
        """测试辅助：写入对象。"""
        self.objects[object_key] = bytes(data)

    async def read_bytes(self, source: DocumentSource) -> bytes:
        """读取对象；不存在抛 KeyError（测试能直接暴露数据准备错误）。"""
        try:
            return self.objects[source.object_key]
        except KeyError as exc:
            raise KeyError(f"源文件不存在: {source.object_key}") from exc

    async def delete_document(
        self,
        kb_id,
        doc_id,
        *,
        extension: str | None = None,
    ) -> int:
        """按 ``source/{kb_id}/{doc_id}`` 前缀删除，返回删除数。"""
        prefix = f"source/{kb_id}/{doc_id}"
        keys = [
            key
            for key in self.objects
            if key == prefix or key.startswith(f"{prefix}.")
        ]
        for key in keys:
            self.objects.pop(key, None)
        return len(keys)

    async def exists(self, source: DocumentSource) -> bool:
        """检查内存对象是否存在。"""
        return source.object_key in self.objects
