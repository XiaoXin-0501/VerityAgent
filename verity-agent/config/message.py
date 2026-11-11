"""消息模块配置。"""

from __future__ import annotations

from pydantic import Field

from .base import ConfigModel

__all__ = ["MessageConfig"]

_MB = 1024 * 1024


class MessageConfig(ConfigModel):
    # content 协议的当前写入版本；读取时兼容 v1（D-22/v2）
    content_version: int = Field(default=2, ge=1)
    # 单条用户消息最多图片数
    max_images: int = Field(default=9, ge=0)
    # 单条用户消息最多文件数
    max_files: int = Field(default=9, ge=0)
    # 单张图片大小上限（字节）
    max_image_bytes: int = Field(default=20 * _MB, gt=0)
    # 单个文件大小上限（字节）
    max_file_bytes: int = Field(default=20 * _MB, gt=0)
    # 单条消息文本最大字符数
    max_text_chars: int = Field(default=200_000, gt=0)
    # 单条工具结果转文本回喂 LLM 的字符上限；超出走 overflow 文件
    max_tool_result_chars: int = Field(default=60_000, gt=0)
