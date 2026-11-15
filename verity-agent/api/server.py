"""开发启动入口。

从 ``verity-agent`` 目录运行::

    uv run python -m api.server

等价于 ``uvicorn api.app:app --reload``。生产环境使用进程管理器启动
``api.app:app``，不要开 reload。
"""

from __future__ import annotations

import asyncio
import logging
import sys

import uvicorn

from config import settings

from .app import app
from .logging_setup import setup_logging

__all__ = ["app", "main"]

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def main() -> None:
    """启动 uvicorn；Windows 使用 Selector 事件循环且不启用 reload。"""
    log_path = setup_logging()
    if log_path is not None:
        logging.getLogger(__name__).info("文件日志写入: %s", log_path)
    uvicorn.run(
        "api.app:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=settings.server.reload,
        workers=None if settings.server.reload else settings.server.workers,
        root_path=settings.server.root_path,
        loop="none" if sys.platform == "win32" else "auto",
        # 日志由 setup_logging 统一接管，避免 uvicorn 覆盖 handler。
        log_config=None,
    )


if __name__ == "__main__":
    main()
