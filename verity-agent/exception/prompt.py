"""Prompt 模块异常：加载、查找与渲染失败。"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Mapping

from .base import VerityError

__all__ = ["PromptErrorCode", "PromptError"]


class PromptErrorCode(StrEnum):
    # 配置不合法（目录缺失、同时关闭内置与覆盖源等）
    INVALID_CONFIG = "INVALID_CONFIG"
    # manifest/模板文件读取或校验失败
    LOAD_FAILED = "LOAD_FAILED"
    # 请求的 prompt id 不存在
    NOT_FOUND = "NOT_FOUND"
    # 变量缺失、未知变量或模板语法错误
    RENDER_FAILED = "RENDER_FAILED"


class PromptError(VerityError):
    """Prompt 管理模块的统一异常。

    Prompt 属于服务端配置资产，失败通常意味着部署/模板有问题，
    因此默认 ``user_visible=False``、HTTP 500；调用方只判断 ``error_code``。
    """

    default_message = "Prompt 处理失败"
    http_status = 500
    user_visible = False

    def __init__(
        self,
        code: PromptErrorCode,
        message: str | None = None,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        """保存错误码并生成稳定的 ``PROMPT_*`` 机器码。"""
        self.error_code = code
        self.code = f"PROMPT_{code.value}"
        super().__init__(message, details=details)
