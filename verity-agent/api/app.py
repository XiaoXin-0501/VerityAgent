"""FastAPI 应用工厂。"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from exception import register_exception_handlers

from .lifespan import lifespan
from .middleware import RequestContextMiddleware
from .routers import router

__all__ = ["create_app", "app"]


def create_app() -> FastAPI:
    """构建 FastAPI 应用：中间件、异常处理器与路由前缀。"""
    app = FastAPI(
        title=settings.app.name,
        version=settings.app.version,
        debug=settings.app.debug,
        lifespan=lifespan,
        docs_url="/docs" if not settings.app.is_prod else None,
        redoc_url=None,
    )

    app.add_middleware(RequestContextMiddleware)
    # 异常处理器在请求上下文里读取该配置，避免 exception 包反向 import config。
    app.state.request_id_header = settings.app.request_id_header
    if settings.server.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.server.cors_origins,
            allow_credentials=settings.server.cors_allow_credentials,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    register_exception_handlers(app)
    app.include_router(router, prefix=settings.app.api_prefix)
    return app


app = create_app()
