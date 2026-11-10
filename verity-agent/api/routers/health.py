"""健康检查路由。"""

from __future__ import annotations

from fastapi import APIRouter, Request

from config import settings
from store.pg import get_pg
from store.redis import get_redis

router = APIRouter(tags=["健康检查"])


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "name": settings.app.name, "version": settings.app.version}


@router.get("/ready")
async def ready(request: Request) -> dict[str, object]:
    pg_ok = False
    redis_ok = False
    try:
        pg_ok = await get_pg().ping()
    except Exception:
        pg_ok = False
    if getattr(request.app.state, "redis_ready", False):
        try:
            redis_ok = await get_redis().ping()
        except Exception:
            redis_ok = False
    return {
        "ready": pg_ok,
        "postgres": pg_ok,
        "redis": redis_ok,
        "minio": bool(getattr(request.app.state, "minio_ready", False)),
        "weaviate": bool(getattr(request.app.state, "weaviate_ready", False)),
        "neo4j": bool(getattr(request.app.state, "neo4j_ready", False)),
        "providers": len(getattr(request.app.state, "provider_registry", [])),
    }
