"""HTTP 请求上下文中间件。"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from utils.ids import new_request_id

__all__ = ["RequestContextMiddleware"]


class RequestContextMiddleware(BaseHTTPMiddleware):
    """为每个请求生成/透传 ``X-Request-ID`` 并写入 ``request.state``。

    错误处理器与日志都从 ``request.state.request_id`` 取值；客户端传入的
    id 会被保留，缺失时服务端生成。
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """生成或透传请求 id，写入 request.state 与响应头。"""
        request_id = request.headers.get("X-Request-ID") or new_request_id()
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers.setdefault("X-Request-ID", request_id)
        return response
