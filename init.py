"""仓库根初始化入口。

与 ``main.py`` 同级；真正的初始化逻辑在 ``verity-agent/initialize.py``，
应用启动链路也会调用同一个 ``ensure_initialized``，保证首次启动幂等补齐
表/bucket/collection，不删除或覆盖业务数据。

    uv run python init.py
    uv run python init.py --only pg redis
    uv run python init.py --weaviate-check-only
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SOURCE_ROOT = _ROOT / "verity-agent"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

# Windows 默认 ProactorEventLoop 不支持 psycopg 异步连接。
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from initialize import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
