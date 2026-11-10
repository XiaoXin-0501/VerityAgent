"""MinIO 对象读写门面。

只负责对象层：bucket/key、上传下载、预签名、按前缀删除。业务归属、权限、
PG 状态机由上层负责；对象不存在在删除路径上视为成功。
"""

from __future__ import annotations

import asyncio
import io
import mimetypes
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from exception import MinioObjectNotFoundError

from .connection import MinioConnectionManager, get_minio
from .paths import (
    kb_artifact_key,
    kb_parse_prefix,
    kb_source_key,
    session_artifact_key,
    session_overflow_key,
    session_parse_prefix,
    session_source_key,
)

__all__ = ["ObjectInfo", "MinioStorage"]


@dataclass(frozen=True, slots=True)
class ObjectInfo:
    bucket: str
    key: str
    size: int
    etag: str | None = None
    content_type: str | None = None
    last_modified: datetime | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)


class MinioStorage:
    """MinIO 对象存储门面；所有网络调用都放到线程池。"""

    def __init__(self, manager: MinioConnectionManager | None = None) -> None:
        self._manager = manager

    @property
    def manager(self) -> MinioConnectionManager:
        return self._manager or get_minio()

    @property
    def config(self):
        return self.manager.config

    # ——————————————————— bucket 约定 ———————————————————

    @property
    def kb_bucket(self) -> str:
        return self.config.buckets.kb

    @property
    def session_bucket(self) -> str:
        return self.config.buckets.session

    # ——————————————————— 上传 ———————————————————

    async def put_bytes(
        self,
        bucket: str,
        key: str,
        data: bytes,
        *,
        content_type: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> ObjectInfo:
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("data 必须是 bytes")
        payload = bytes(data)
        content_type = content_type or mimetypes.guess_type(key)[0] or "application/octet-stream"

        def _put():
            result = self.manager.client.put_object(
                bucket,
                key,
                io.BytesIO(payload),
                length=len(payload),
                content_type=content_type,
                metadata=dict(metadata or {}),
                part_size=self.config.part_size,
            )
            return result

        try:
            await asyncio.to_thread(_put)
        except Exception as exc:
            raise self.manager.translate_error(exc) from exc
        return ObjectInfo(
            bucket=bucket,
            key=key,
            size=len(payload),
            content_type=content_type,
            metadata=dict(metadata or {}),
        )

    async def put_file(
        self,
        bucket: str,
        key: str,
        path: str | Path,
        *,
        content_type: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> ObjectInfo:
        file_path = Path(path)
        data = await asyncio.to_thread(file_path.read_bytes)
        return await self.put_bytes(
            bucket,
            key,
            data,
            content_type=content_type,
            metadata=metadata,
        )

    async def put_kb_source(
        self,
        kb_id: Any,
        doc_id: Any,
        extension: str,
        data: bytes,
        *,
        content_type: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> ObjectInfo:
        return await self.put_bytes(
            self.kb_bucket,
            kb_source_key(kb_id, doc_id, extension),
            data,
            content_type=content_type,
            metadata=metadata,
        )

    async def put_session_source(
        self,
        conversation_id: Any,
        file_id: Any,
        extension: str,
        data: bytes,
        *,
        content_type: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> ObjectInfo:
        return await self.put_bytes(
            self.session_bucket,
            session_source_key(conversation_id, file_id, extension),
            data,
            content_type=content_type,
            metadata=metadata,
        )

    async def put_overflow(
        self,
        conversation_id: Any,
        call_id: Any,
        data: bytes,
        *,
        extension: str = "json",
        content_type: str | None = None,
    ) -> ObjectInfo:
        return await self.put_bytes(
            self.session_bucket,
            session_overflow_key(conversation_id, call_id, extension),
            data,
            content_type=content_type or "application/json",
        )

    async def put_artifact(
        self,
        *,
        kb_id: Any | None = None,
        conversation_id: Any | None = None,
        artifact_id: Any,
        extension: str,
        data: bytes,
        content_type: str | None = None,
    ) -> ObjectInfo:
        if (kb_id is None) == (conversation_id is None):
            raise ValueError("kb_id 与 conversation_id 必须且只能提供一个")
        if kb_id is not None:
            bucket = self.kb_bucket
            key = kb_artifact_key(kb_id, artifact_id, extension)
        else:
            bucket = self.session_bucket
            key = session_artifact_key(conversation_id, artifact_id, extension)
        return await self.put_bytes(bucket, key, data, content_type=content_type)

    # ——————————————————— 读取 ———————————————————

    async def get_bytes(self, bucket: str, key: str) -> bytes:
        def _get():
            response = self.manager.client.get_object(bucket, key)
            try:
                return response.read()
            finally:
                response.close()
                response.release_conn()

        try:
            return await asyncio.to_thread(_get)
        except Exception as exc:
            raise self.manager.translate_error(exc) from exc

    async def stat(self, bucket: str, key: str) -> ObjectInfo:
        try:
            stat = await asyncio.to_thread(self.manager.client.stat_object, bucket, key)
        except Exception as exc:
            raise self.manager.translate_error(exc) from exc
        return ObjectInfo(
            bucket=bucket,
            key=key,
            size=int(getattr(stat, "size", 0)),
            etag=getattr(stat, "etag", None),
            content_type=getattr(stat, "content_type", None),
            last_modified=getattr(stat, "last_modified", None),
            metadata=getattr(stat, "metadata", {}) or {},
        )

    async def exists(self, bucket: str, key: str) -> bool:
        try:
            await self.stat(bucket, key)
            return True
        except MinioObjectNotFoundError:
            return False

    async def presigned_get_url(
        self,
        bucket: str,
        key: str,
        *,
        expiry_s: int | None = None,
        response_headers: Mapping[str, str] | None = None,
    ) -> str:
        from datetime import timedelta

        expires = timedelta(seconds=expiry_s or self.config.presign_expiry_s)
        try:
            return await asyncio.to_thread(
                self.manager.client.presigned_get_object,
                bucket,
                key,
                expires,
                dict(response_headers or {}),
            )
        except Exception as exc:
            raise self.manager.translate_error(exc) from exc

    # ——————————————————— 列表 / 删除 ———————————————————

    async def list_keys(
        self,
        bucket: str,
        prefix: str = "",
        *,
        recursive: bool = True,
    ) -> list[str]:
        def _list() -> list[str]:
            return [
                item.object_name
                for item in self.manager.client.list_objects(
                    bucket, prefix=prefix, recursive=recursive
                )
            ]

        try:
            return await asyncio.to_thread(_list)
        except Exception as exc:
            raise self.manager.translate_error(exc) from exc

    async def remove(self, bucket: str, key: str) -> None:
        try:
            await asyncio.to_thread(self.manager.client.remove_object, bucket, key)
        except Exception as exc:
            error = self.manager.translate_error(exc)
            # 对象不存在在删除路径上视为成功
            if isinstance(error, MinioObjectNotFoundError):
                return
            raise error from exc

    async def remove_prefix(self, bucket: str, prefix: str) -> int:
        keys = await self.list_keys(bucket, prefix, recursive=True)
        for key in keys:
            await self.remove(bucket, key)
        return len(keys)

    async def delete_kb_document(
        self,
        kb_id: Any,
        doc_id: Any,
        *,
        extension: str | None = None,
    ) -> int:
        """删文档：source 原件 + parse 前缀；按前缀幂等。"""
        removed = 0
        source_prefix = f"source/{kb_id}/{doc_id}."
        if extension:
            await self.remove(
                self.kb_bucket, kb_source_key(kb_id, doc_id, extension)
            )
            removed += 1
        else:
            removed += await self.remove_prefix(self.kb_bucket, source_prefix)
        removed += await self.remove_prefix(
            self.kb_bucket, kb_parse_prefix(kb_id, doc_id)
        )
        return removed

    async def delete_session_file(
        self,
        conversation_id: Any,
        file_id: Any,
        *,
        extension: str | None = None,
    ) -> int:
        removed = 0
        if extension:
            await self.remove(
                self.session_bucket, session_source_key(conversation_id, file_id, extension)
            )
            removed += 1
        else:
            removed += await self.remove_prefix(
                self.session_bucket, f"source/{conversation_id}/{file_id}."
            )
        removed += await self.remove_prefix(
            self.session_bucket, session_parse_prefix(conversation_id, file_id)
        )
        return removed
