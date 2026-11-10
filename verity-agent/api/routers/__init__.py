"""API 路由聚合。"""

from __future__ import annotations

from fastapi import APIRouter

from .auth import router as auth_router
from .health import router as health_router

router = APIRouter()
router.include_router(health_router)
router.include_router(auth_router)

__all__ = ["router", "auth_router", "health_router"]
