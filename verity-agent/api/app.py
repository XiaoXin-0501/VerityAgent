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
    app = FastAPI(
        title=settings.app.name,
        version=settings.app.version,
        debug=settings.app.debug,
        lifespan=lifespan,
        docs_url="/docs" if not settings.app.is_prod else None,
        redoc_url=None,
    )

    app.add_middleware(RequestContextMiddleware)
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
