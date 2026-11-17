"""应用日志初始化：控制台 + 分文件轮转。

日志按用途拆分，避免 HTTP 轮询、RAG 入库和普通业务日志混在一个文件里：

- ``verity-agent.log``：主应用日志（不含 access / RAG）；
- ``access.log``：uvicorn HTTP access 日志；
- ``error.log``：所有 WARNING/ERROR，集中排障；
- ``rag.log``：``rag.*`` 入库/检索组件日志。

设计约束：

- 目录、文件名、大小、保留数量全部来自 ``AppConfig.log``，不在业务代码写死；
- 默认写入 ``D:/verityAgent日志``，可用 ``APP__LOG__DIRECTORY`` 覆盖；
- 初始化失败（目录无权限/盘符不存在）只降级为控制台，不阻断应用启动；
- 幂等：重复调用不会叠加 handler；``shutdown_logging`` 用于测试/优雅退出。
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config import settings
from config.app import AppConfig, LoggingConfig
from config.base import SOURCE_ROOT
from utils.log_context import current_request_id

__all__ = ["setup_logging", "shutdown_logging", "LOG_FORMAT"]

# 统一格式：时间/级别/logger/请求 id/进程/线程/消息。
LOG_FORMAT = (
    "%(asctime)s %(levelname)s %(name)s "
    "[req=%(request_id)s pid=%(process)d thread=%(threadName)s] %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# uvicorn 自己的 logger 默认不向 root 传播；access 日志单独挂文件。
_UVICORN_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")
# 第三方 HTTP 客户端 INFO 日志对排障价值低，默认只保留 WARNING。
_NOISY_LOGGERS = ("httpx", "httpcore", "urllib3")

_file_handlers: dict[str, RotatingFileHandler] = {}
_log_signature: tuple[object, ...] | None = None
_log_path: Path | None = None


class _RequestIdFilter(logging.Filter):
    """把 contextvar 中的 request_id 注入日志记录。"""

    def filter(self, record: logging.LogRecord) -> bool:
        """写入 ``record.request_id``；没有请求上下文时为 ``-``。"""
        record.request_id = current_request_id()
        return True


class _LoggerNameFilter(logging.Filter):
    """按 logger 名称包含/排除记录，用于把 RAG 与 access 分流。"""

    def __init__(
        self,
        *,
        include_prefixes: tuple[str, ...] = (),
        exclude_names: tuple[str, ...] = (),
        exclude_prefixes: tuple[str, ...] = (),
    ) -> None:
        """保存前缀/名称规则。"""
        super().__init__()
        self.include_prefixes = include_prefixes
        self.exclude_names = frozenset(exclude_names)
        self.exclude_prefixes = exclude_prefixes

    @staticmethod
    def _matches(name: str, prefixes: tuple[str, ...]) -> bool:
        """logger 名是否命中某个前缀（按层级，不做裸字符串包含）。"""
        return any(
            name == prefix or name.startswith(f"{prefix}.")
            for prefix in prefixes
        )

    def filter(self, record: logging.LogRecord) -> bool:
        """include 为空表示不限制；exclude 优先。"""
        name = record.name
        if name in self.exclude_names:
            return False
        if self._matches(name, self.exclude_prefixes):
            return False
        if self.include_prefixes and not self._matches(
            name, self.include_prefixes
        ):
            return False
        return True


def _resolve_directory(raw: str) -> Path:
    """解析日志目录；相对路径按 verity-agent 源码根目录解析。"""
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = SOURCE_ROOT / path
    return path


def _is_managed(handler: logging.Handler) -> bool:
    """判断 handler 是否由本模块创建。"""
    return bool(getattr(handler, "_verity_managed", False))


def _remove_managed_console_handlers() -> None:
    """从 root 移除本模块创建的控制台 handler。"""
    root = logging.getLogger()
    for handler in list(root.handlers):
        if not getattr(handler, "_verity_console", False):
            continue
        root.removeHandler(handler)
        handler.close()


def _remove_managed_file_handlers() -> None:
    """从 root/uvicorn 移除并关闭本模块创建的文件 handler。"""
    global _log_signature, _log_path
    targets = [
        logging.getLogger(),
        *(logging.getLogger(name) for name in _UVICORN_LOGGERS),
    ]
    removed: set[int] = set()
    for logger in targets:
        for handler in list(logger.handlers):
            if not getattr(handler, "_verity_file", False):
                continue
            logger.removeHandler(handler)
            if id(handler) not in removed:
                removed.add(id(handler))
                handler.close()
    _file_handlers.clear()
    _log_signature = None
    _log_path = None


def _make_file_handler(
    path: Path,
    *,
    log_cfg: LoggingConfig,
    level: int,
    formatter: logging.Formatter,
    filters: tuple[logging.Filter, ...] = (),
) -> RotatingFileHandler:
    """创建带请求 id/分流过滤器的轮转文件 handler。"""
    handler = RotatingFileHandler(
        path,
        maxBytes=log_cfg.max_bytes,
        backupCount=log_cfg.backup_count,
        encoding=log_cfg.encoding,
        delay=True,
    )
    handler.setLevel(level)
    handler.setFormatter(formatter)
    handler.addFilter(_RequestIdFilter())
    for item in filters:
        handler.addFilter(item)
    handler._verity_managed = True  # type: ignore[attr-defined]
    handler._verity_file = True  # type: ignore[attr-defined]
    return handler


def setup_logging(config: AppConfig | None = None) -> Path | None:
    """按配置安装控制台/分文件 handler；返回主应用日志路径。"""
    global _log_signature, _log_path
    app_cfg = config or settings.app
    log_cfg = app_cfg.log
    level = getattr(logging, app_cfg.log_level.upper(), logging.INFO)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=_DATE_FORMAT)
    root = logging.getLogger()
    root.setLevel(level)
    logging.captureWarnings(True)
    # 默认压掉第三方 HTTP 客户端的逐请求 INFO；DEBUG 模式下保留，便于排查。
    noisy_level = logging.INFO if level <= logging.DEBUG else logging.WARNING
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(noisy_level)

    # 控制台：保留容器/终端可见；已存在则不重复添加。
    if log_cfg.console and not any(
        getattr(handler, "_verity_console", False) for handler in root.handlers
    ):
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console.addFilter(_RequestIdFilter())
        console._verity_managed = True  # type: ignore[attr-defined]
        console._verity_console = True  # type: ignore[attr-defined]
        root.addHandler(console)
    elif not log_cfg.console:
        _remove_managed_console_handlers()

    if not log_cfg.to_file:
        _remove_managed_file_handlers()
        return None

    try:
        directory = _resolve_directory(log_cfg.directory)
        paths = {
            "app": directory / log_cfg.file_name,
            "access": directory / log_cfg.access_file_name,
            "error": directory / log_cfg.error_file_name,
            "rag": directory / log_cfg.rag_file_name,
        }
        signature: tuple[object, ...] = (
            str(directory),
            log_cfg.file_name,
            log_cfg.access_file_name,
            log_cfg.error_file_name,
            log_cfg.rag_file_name,
            log_cfg.max_bytes,
            log_cfg.backup_count,
            log_cfg.encoding,
            level,
            log_cfg.console,
        )
        # 同一配置重复调用直接复用；换路径/参数时先清理旧 handler。
        if (
            _log_signature == signature
            and _log_path == paths["app"]
            and _file_handlers
        ):
            return paths["app"]
        _remove_managed_file_handlers()
        directory.mkdir(parents=True, exist_ok=True)

        app_handler = _make_file_handler(
            paths["app"],
            log_cfg=log_cfg,
            level=level,
            formatter=formatter,
            filters=(
                _LoggerNameFilter(
                    exclude_names=("uvicorn.access",),
                    exclude_prefixes=("rag", "uvicorn.access"),
                ),
            ),
        )
        rag_handler = _make_file_handler(
            paths["rag"],
            log_cfg=log_cfg,
            level=level,
            formatter=formatter,
            filters=(_LoggerNameFilter(include_prefixes=("rag",)),),
        )
        error_handler = _make_file_handler(
            paths["error"],
            log_cfg=log_cfg,
            level=logging.WARNING,
            formatter=formatter,
        )
        access_handler = _make_file_handler(
            paths["access"],
            log_cfg=log_cfg,
            level=logging.INFO,
            formatter=formatter,
        )

        root.addHandler(app_handler)
        root.addHandler(rag_handler)
        root.addHandler(error_handler)
        # uvicorn.access 单独挂文件；若它同时 propagate 到 root，
        # app_handler 已排除 access，不会重复写到主日志。
        logging.getLogger("uvicorn.access").addHandler(access_handler)

        _file_handlers.update(
            app=app_handler,
            rag=rag_handler,
            error=error_handler,
            access=access_handler,
        )
        _log_signature = signature
        _log_path = paths["app"]
        return paths["app"]
    except OSError as exc:
        _remove_managed_file_handlers()
        logging.getLogger(__name__).warning(
            "文件日志初始化失败，降级为控制台输出: %s", exc
        )
        return None


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
