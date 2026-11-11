"""解析产物 manifest 存储接口（v1 由 MinIO 实现）。"""

from __future__ import annotations

from typing import Protocol

from rag.shared.models import ParseManifest

__all__ = ["ManifestStore"]


class ManifestStore(Protocol):
    async def read(self, kb_id, doc_id) -> ParseManifest | None: ...
    async def write(self, manifest: ParseManifest) -> None: ...
    async def delete(self, kb_id, doc_id) -> None: ...
