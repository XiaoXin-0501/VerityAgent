"""应用日志初始化：控制台 + 可配置的本地轮转文件。

设计约束：

- 目录、文件名、大小、保留数量全部来自 ``AppConfig.log``，不在业务代码写死；
- 默认写入 ``D:/verityAgent日志/verity-agent.log``，可用
  ``APP__LOG__DIRECTORY`` 覆盖；
- 初始化失败（目录无权限/盘符不存在）只降级为控制台，不阻断应用启动；
- 幂等：重复调用不会叠加 handler；``shutdown_logging`` 用于测试/优雅退出。
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config import settings
from config.app import AppConfig
from config.base import SOURCE_ROOT

__all__ = ["setup_logging", "shutdown_logging", "LOG_FORMAT"]

# 统一格式：时间/级别/logger/进程/线程/消息；Hook 日志的 trace/run id 在消息体内。
LOG_FORMAT = (
    "%(asctime)s %(levelname)s %(name)s "
    "[pid=%(process)d thread=%(threadName)s] %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# uvicorn 自己的 logger 默认不向 root 传播；显式挂同一个文件 handler。
_UVICORN_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")

_file_handler: RotatingFileHandler | None = None
_log_path: Path | None = None


def _resolve_directory(raw: str) -> Path:
    """解析日志目录；相对路径按 verity-agent 源码根目录解析。"""
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = SOURCE_ROOT / path
    return path


def _is_managed(handler: logging.Handler) -> bool:
    """判断 handler 是否由本模块创建。"""
    return bool(getattr(handler, "_verity_managed", False))


def _remove_managed_file_handlers() -> None:
    """从 root/uvicorn 移除并关闭本模块创建的文件 handler。"""
    global _file_handler, _log_path
    targets = [logging.getLogger(), *(logging.getLogger(n) for n in _UVICORN_LOGGERS)]
    removed: set[int] = set()
    for logger in targets:
        for handler in list(logger.handlers):
            if not getattr(handler, "_verity_file", False):
                continue
            logger.removeHandler(handler)
            if id(handler) not in removed:
                removed.add(id(handler))
                handler.close()
    _file_handler = None
    _log_path = None


def setup_logging(config: AppConfig | None = None) -> Path | None:
    """按配置安装控制台/文件 handler；返回实际日志文件路径（未启用为 None）。"""
    global _file_handler, _log_path
    app_cfg = config or settings.app
    level = getattr(logging, app_cfg.log_level.upper(), logging.INFO)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=_DATE_FORMAT)
    root = logging.getLogger()
    root.setLevel(level)
    logging.captureWarnings(True)

    # 控制台：保留容器/终端可见；已存在则不重复添加。
    if app_cfg.log.console and not any(
        getattr(handler, "_verity_console", False) for handler in root.handlers
    ):
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console._verity_managed = True  # type: ignore[attr-defined]
        console._verity_console = True  # type: ignore[attr-defined]
        root.addHandler(console)

    if not app_cfg.log.to_file:
        _remove_managed_file_handlers()
        return None

    try:
        directory = _resolve_directory(app_cfg.log.directory)
        path = directory / app_cfg.log.file_name
        # 同一路径重复调用直接复用；换路径时先清理旧 handler。
        if _file_handler is not None and _log_path == path:
            return path
        _remove_managed_file_handlers()
        directory.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            path,
            maxBytes=app_cfg.log.max_bytes,
            backupCount=app_cfg.log.backup_count,
            encoding=app_cfg.log.encoding,
            delay=True,
        )
        handler.setFormatter(formatter)
        handler._verity_managed = True  # type: ignore[attr-defined]
        handler._verity_file = True  # type: ignore[attr-defined]
        root.addHandler(handler)
        for name in _UVICORN_LOGGERS:
            logger = logging.getLogger(name)
            # propagate=True 时 root 的 handler 已能收到，避免同一条写两次。
            if logger.propagate:
                continue
            if not any(
                getattr(item, "_verity_file", False) for item in logger.handlers
            ):
                logger.addHandler(handler)
    except OSError as exc:
        logging.getLogger(__name__).warning(
            "文件日志初始化失败，降级为控制台输出: %s", exc
        )
        return None

    _file_handler = handler
    _log_path = path
    return path


def shutdown_logging() -> None:
    """移除本模块安装的 handler；测试与进程退出时调用。"""
    global _log_path
    root = logging.getLogger()
    for handler in list(root.handlers):
        if _is_managed(handler):
            root.removeHandler(handler)
            handler.close()
    _remove_managed_file_handlers()
    _log_path = None
