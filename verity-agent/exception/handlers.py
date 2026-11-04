"""FastAPI 全局异常处理器。

把项目自定义异常（``VerityError`` 及其子类）统一转成::

    {"code": "...", "message": "...", "detail": {...}}

并带上异常自带的 HTTP 状态码，避免业务异常退化成裸 500。

使用方式::

    from exception import register_exception_handlers

    app = FastAPI()
    register_exception_handlers(app)
"""

from __future__ import annotations

from typing import cast

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .base import VerityError


async def verity_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    """将 ``VerityError`` 转成统一 JSON 响应。"""
    error = cast(VerityError, exc)
    return JSONResponse(status_code=error.http_status, content=error.to_dict())


def register_exception_handlers(app: FastAPI) -> None:
    """注册项目内所有全局异常处理器。"""
    app.add_exception_handler(VerityError, verity_error_handler)
