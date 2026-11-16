"""消息模块配置。"""

from __future__ import annotations

from pydantic import Field, model_validator

from .base import ConfigModel

__all__ = ["MessageConfig"]

_MB = 1024 * 1024


class MessageConfig(ConfigModel):
    # content 协议的当前写入版本；v1 仅保留读取兼容
    content_version: int = Field(default=2, ge=1)
    # 单条用户消息最多图片数
    max_images: int = Field(default=9, ge=0)
    # 单条用户消息最多文件数
    max_files: int = Field(default=9, ge=0)
    # 单个文件大小上限（字节）；知识库上传与会话附件共用
    max_file_bytes: int = Field(default=20 * _MB, gt=0)
    # 单条消息文本最大字符数
    max_text_chars: int = Field(default=200_000, gt=0)

    @model_validator(mode="after")
    def _writer_is_v2(self) -> "MessageConfig":
        """新消息只写 v2；v1 只保留读取兼容，禁止配置成回写旧版本。"""
        if self.content_version != 2:
            raise ValueError("message.content_version 只支持 2（v1 仅读取兼容）")
        return self
