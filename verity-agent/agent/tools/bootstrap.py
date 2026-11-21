"""工具模块装配入口。

启动时由 ``api.lifespan`` 调用。本模块统一从 ``agent.tools.manifest`` 读取
工具声明，按启动期权限与依赖装配全局 ``ToolRegistry``；具体执行器仍在
``agent.tools.builtin`` 中实现。

- **注册前过滤**：manifest 的 ``required_permission`` 不在启动期已知权限码
  集合中时，不注册该工具；
- 无权限码的工具（登录用户可用）始终尝试注册；
- manifest 的 ``required_services`` 在装配期统一校验；缺失时 warning 并跳过，
  工厂只负责自身的运行配置判断（如 recall_round 的 enabled）；
- 单次请求是否暴露由 ``ToolSelection`` 按 manifest 的 ``exposure`` 条件再过滤，
  执行期仍会校验工具是否在当前请求的暴露集合中。
"""

from __future__ import annotations

import logging
from typing import Any

from config.tools import ToolsConfig

from .declaration import ToolBuildContext
from .manifest import TOOL_MANIFEST, build_declared_tool
from .overflow import MinioOverflowWriter
from .pipeline import ToolPipeline
from .redis_state import build_redis_tool_state
from .registry import ToolRegistry

__all__ = ["build_tool_registry", "build_tool_pipeline"]

logger = logging.getLogger(__name__)


def build_tool_registry(
    *,
    services: Any | None = None,
    available_permissions: frozenset[str] | None = None,
    tools_config: ToolsConfig | None = None,
) -> ToolRegistry:
    """按 manifest 装配全局工具注册表；启动后只读。

    ``services`` 是通用依赖包（``app.state`` 或 Mapping）；工厂按自己的
    service key 取依赖，新增工具不需要修改本函数签名。
    ``available_permissions`` 是启动期已知的权限码集合（权限定义，不是某个
    用户的授权）；为空/None 时不做权限过滤。请求期仍会按用户实际权限过滤。
    """
    config = tools_config or ToolsConfig()
    ctx = ToolBuildContext(tools_config=config, services=services)
    registry = ToolRegistry()
    for entry in TOOL_MANIFEST:
        if (
            available_permissions is not None
            and entry.required_permission is not None
            and entry.required_permission not in available_permissions
        ):
            logger.info(
                "工具 %s 的权限码未定义，跳过注册: %s",
                entry.name,
                entry.required_permission,
            )
            continue
        missing_services = [
            name
            for name in entry.required_services
            if ctx.get_service(name) is None
        ]
        if missing_services:
            logger.warning(
                "工具 %s 缺少依赖服务，跳过注册: %s",
                entry.name,
                missing_services,
            )
            continue
        tool = build_declared_tool(entry, ctx)
        if tool is None:
            logger.info("工具不可用（依赖或配置），跳过注册: %s", entry.name)
            continue
        if tool.spec.name != entry.name:
            raise ValueError(
                f"manifest 工具名与实现不一致: {entry.name} != {tool.spec.name}"
            )
        registry.register(tool)
    return registry


def build_tool_pipeline(
    registry: ToolRegistry,
    config: ToolsConfig,
    *,
    cache: Any | None = None,
    idempotency: Any | None = None,
    circuit_breaker: Any | None = None,
    rate_limiter: Any | None = None,
    overflow_writer: Any | None = None,
) -> ToolPipeline:
    """创建执行流水线，并注入 Redis 缓存/幂等/熔断/限流与 MinIO 溢出。"""
    # 调用方已注入全部状态后端时不再创建 Redis 管理器，避免无意义的连接对象。
    needs_redis = any(
        item is None
        for item in (cache, idempotency, circuit_breaker, rate_limiter)
    )
    state = build_redis_tool_state() if needs_redis else {}
    return ToolPipeline(
        registry,
        config,
        cache=cache or state.get("cache"),
        idempotency=idempotency or state.get("idempotency"),
        circuit_breaker=circuit_breaker or state.get("circuit_breaker"),
        rate_limiter=rate_limiter or state.get("rate_limiter"),
        overflow_writer=overflow_writer or MinioOverflowWriter(),
    )
