"""内置 Hook 的统一构建入口。"""

from __future__ import annotations

from typing import Any

from config.hooks import HooksConfig
from store.redis import RedisConnectionManager

from ..counters import UsageCounterStore
from ..dispatcher import HookDispatcher
from ..redaction import RedactionPolicy
from .audit import AuditHook
from .logging import LoggingHook
from .metrics import MetricsHook
from .token_usage import TokenUsageHook
from .tracing import TracingHook

__all__ = ["build_dispatcher"]


def build_dispatcher(
    config: HooksConfig,
    *,
    conversation_repository: Any | None = None,
    redis: RedisConnectionManager | None = None,
) -> HookDispatcher:
    """按配置注册内置 Hook，返回 dispatcher。

    TokenUsageHook 的会话累计写入是可选的；不传 repository 时只记录日志/Redis。
    """
    from ..registry import HookRegistry

    registry = HookRegistry()
    redaction = RedactionPolicy(config.redaction)

    if config.logging.enabled:
        hook = LoggingHook(config.logging, redaction)
        registry.register(hook.spec, hook, source="builtin.logging")

    if config.audit.enabled:
        hook = AuditHook(config.audit, redaction)
        registry.register(hook.spec, hook, source="builtin.audit")

    if config.metrics.enabled:
        hook = MetricsHook(config.metrics)
        registry.register(hook.spec, hook, source="builtin.metrics")

    if config.tracing.enabled:
        hook = TracingHook(config.tracing)
        registry.register(hook.spec, hook, source="builtin.tracing")

    if config.token_usage.enabled:
        counters = (
            UsageCounterStore(config.token_usage, redis=redis)
            if config.token_usage.redis_counters
            else None
        )
        hook = TokenUsageHook(
            config.token_usage,
            counters=counters,
            conversation_repository=conversation_repository,
        )
        registry.register(hook.spec, hook, source="builtin.token_usage")

    return HookDispatcher(registry, config)
