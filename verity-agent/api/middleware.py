"""HTTP 请求上下文中间件。"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from config import settings
from utils.ids import new_request_id

__all__ = ["RequestContextMiddleware"]


class RequestContextMiddleware(BaseHTTPMiddleware):
    """为每个请求生成/透传请求 id，并在入口拦截超大请求体。

    错误处理器与日志都从 ``request.state.request_id`` 取值；客户端传入的
    id 会被保留，缺失时服务端生成。header 名来自
    ``APP__REQUEST_ID_HEADER``，不写死。
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """生成或透传请求 id，写入 request.state 与响应头。"""
        header = settings.app.request_id_header
        request_id = request.headers.get(header) or new_request_id()
        request.state.request_id = request_id

        content_length = request.headers.get("content-length")
        if content_length and content_length.isdigit():
            limit = settings.server.max_request_body_bytes
            if int(content_length) > limit:
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": {
                            "code": "PAYLOAD_TOO_LARGE",
                            "message": "请求体超过大小上限",
                            "user_visible": True,
                            "retryable": False,
                            "details": {
                                "content_length": int(content_length),
                                "max_request_body_bytes": limit,
                            },
                            "request_id": request_id,
                        }
                    },
                    headers={header: request_id},
                )

        response = await call_next(request)
        response.headers.setdefault(header, request_id)
        return response
