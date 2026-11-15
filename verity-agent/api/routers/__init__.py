"""API 路由聚合。"""

from __future__ import annotations

from fastapi import APIRouter

from .auth import router as auth_router
from .conversations import router as conversations_router
from .health import router as health_router
from .runs import router as runs_router

router = APIRouter()
router.include_router(health_router)
router.include_router(auth_router)
router.include_router(conversations_router)
router.include_router(runs_router)

__all__ = [
    "router",
    "auth_router",
    "health_router",
    "conversations_router",
    "runs_router",
]
