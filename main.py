"""仓库根启动入口。

源码根目录是 ``verity-agent``（不是 Python 包名）。本文件只负责把源码根
加入 ``sys.path`` 并启动 ASGI 应用，避免在业务代码里散落路径 hack。

    uv run python main.py           # 开发（按 .env 的 SERVER__RELOAD）
    uv run uvicorn api.app:app      # 生产进程管理器直接指向 app
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SOURCE_ROOT = _ROOT / "verity-agent"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

# Windows 默认 ProactorEventLoop 不支持 psycopg 异步连接
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import uvicorn  # noqa: E402

from config import settings  # noqa: E402


def main() -> None:
    uvicorn.run(
        "api.app:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=settings.server.reload,
        workers=None if settings.server.reload else settings.server.workers,
        root_path=settings.server.root_path,
        # Windows 上 uvicorn 的 auto 会强制 ProactorEventLoop；none 沿用上面设置的
        # SelectorEventLoopPolicy，psycopg 异步连接才能工作。
        loop="none" if sys.platform == "win32" else "auto",
    )


if __name__ == "__main__":
    main()
