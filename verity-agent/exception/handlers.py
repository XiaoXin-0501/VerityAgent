"""FastAPI 全局异常处理器。

统一输出 ``ApiError`` 信封（《API 与流式契约》§0.2）：

- 领域异常：保留自带 code / HTTP 状态 / retryable / details；
- 请求校验：转 ``INVALID_REQUEST`` 422；
- 未捕获异常：``INTERNAL`` 500，只记日志，不把堆栈返回给客户端；
- 每个响应都带 ``X-Request-ID``，便于日志关联。
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .base import VerityError

logger = logging.getLogger(__name__)

__all__ = ["register_exception_handlers", "verity_error_handler"]


def _request_id(request: Request) -> str:
    """读取中间件写入的 request_id；缺失时回退到请求头。"""
    value = getattr(request.state, "request_id", None)
    if value:
        return str(value)
    return request.headers.get("X-Request-ID", "")


def _envelope(
    *,
    code: str,
    message: str,
    request: Request,
    status_code: int,
    details: dict[str, Any] | None = None,
    user_visible: bool = True,
    retryable: bool = False,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """按统一错误信封构造 JSON 响应，并回写 X-Request-ID。"""
    error: dict[str, Any] = {
        "code": code,
        "message": message,
        "user_visible": user_visible,
        "retryable": retryable,
        "details": details or {},
    }
    request_id = _request_id(request)
    if request_id:
        error["request_id"] = request_id
    response_headers = dict(headers or {})
    if request_id:
        response_headers.setdefault("X-Request-ID", request_id)
    return JSONResponse(
        status_code=status_code,
        content={"error": error},
        headers=response_headers,
    )


async def verity_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """把 ``VerityError`` 转成统一错误信封。"""
    if not isinstance(exc, VerityError):
        raise exc
    return _envelope(
        code=exc.code,
        message=exc.message,
        request=request,
        status_code=exc.http_status,
        details=exc.details,
        user_visible=exc.user_visible_value,
        retryable=exc.retryable_value,
        headers=(
            {"Retry-After": str(int(exc.retry_after_s))}
            if exc.retry_after_s
            else None
        ),
    )


async def _validation_error_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """把 FastAPI 请求校验错误转换为统一 422 信封。"""
    if not isinstance(exc, RequestValidationError):
        raise exc
    return _envelope(
        code="INVALID_REQUEST",
        message="请求参数校验失败",
        request=request,
        status_code=422,
        details={"errors": jsonable_encoder(exc.errors())},
    )


async def _http_error_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """把 Starlette HTTPException 转换为统一错误信封。"""
    if not isinstance(exc, StarletteHTTPException):
        raise exc
    code_by_status = {
        401: "UNAUTHENTICATED",
        403: "PERMISSION_DENIED",
        404: "NOT_FOUND",
        409: "CONFLICT",
        429: "RATE_LIMITED",
    }
    return _envelope(
        code=code_by_status.get(exc.status_code, "HTTP_ERROR"),
        message=str(exc.detail),
        request=request,
        status_code=exc.status_code,
        headers=getattr(exc, "headers", None),
        retryable=exc.status_code in {429, 502, 503, 504},
    )


async def _unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """记录未处理异常并返回脱敏的 500 信封。"""
    logger.exception("未处理异常: %s %s", request.method, request.url.path)
    return _envelope(
        code="INTERNAL",
        message="服务内部错误",
        request=request,
        status_code=500,
        user_visible=False,
    )


def register_exception_handlers(app: FastAPI) -> None:
    """注册项目内所有全局异常处理器。"""
    app.add_exception_handler(VerityError, verity_error_handler)
    app.add_exception_handler(RequestValidationError, _validation_error_handler)
    app.add_exception_handler(StarletteHTTPException, _http_error_handler)
    app.add_exception_handler(Exception, _unhandled_error_handler)
