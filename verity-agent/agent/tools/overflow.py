"""工具大结果溢出到 MinIO。"""

from __future__ import annotations

import logging
from uuid import UUID

from agent.message.models import ToolFile
from store.minio import MinioStorage
from utils.serialization import json_dumps

__all__ = ["MinioOverflowWriter"]

logger = logging.getLogger(__name__)


class MinioOverflowWriter:
    """把完整 ToolResult 写入会话溢出对象，返回可访问引用。"""

    def __init__(
        self,
        storage: MinioStorage | None = None,
        *,
        presign_expiry_s: int | None = None,
    ) -> None:
        """保存存储门面；不主动建立 MinIO 连接。"""
        self.storage = storage or MinioStorage()
        self.presign_expiry_s = presign_expiry_s

    async def write(
        self,
        *,
        conversation_id: UUID,
        call_id: str,
        tool_name: str,
        content: str,
        data: object,
    ) -> ToolFile:
        """写入 JSON 溢出对象，并返回带预签名 URL 的 ToolFile。"""
        payload = {
            "tool_name": tool_name,
            "call_id": call_id,
            "content": content,
            "data": data,
        }
        try:
            raw = json_dumps(payload).encode("utf-8")
        except TypeError as exc:
            # 执行器返回了不可序列化对象时，至少保留 LLM 可见文本，避免
            # 溢出失败导致整个工具结果丢失。
            logger.warning("工具溢出 data 不可序列化，仅保存 content: %s", exc)
            raw = json_dumps(
                {"tool_name": tool_name, "call_id": call_id, "content": content}
            ).encode("utf-8")
        info = await self.storage.put_overflow(
            conversation_id,
            call_id,
            raw,
            extension="json",
        )
        url = await self.storage.presigned_get_url(
            info.bucket,
            info.key,
            expiry_s=self.presign_expiry_s,
        )
        return ToolFile(filename=info.key, url=url)
