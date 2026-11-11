"""Hooks 核心行为测试（不依赖 Redis/PG）。"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from uuid import uuid4

_ROOT = Path(__file__).resolve().parents[1]
_AGENT_ROOT = _ROOT / "verity-agent"
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

from agent.hooks.builtin.token_usage import TokenUsageHook  # noqa: E402
from agent.hooks.enum import (  # noqa: E402
    HookEvent,
    HookExecutionMode,
    HookFailurePolicy,
    HookStatus,
)
from agent.hooks.models import (  # noqa: E402
    HookContext,
    HookPayload,
    HookSpec,
    LLMUsagePayload,
    UsageKind,
)
from agent.hooks.redaction import RedactionPolicy  # noqa: E402
from agent.hooks.registry import HookRegistry  # noqa: E402
from agent.hooks.dispatcher import HookDispatcher  # noqa: E402
from config.hooks import (  # noqa: E402
    AuditHookConfig,
    HooksConfig,
    LoggingHookConfig,
    MetricsHookConfig,
    RedactionConfig,
    TokenUsageHookConfig,
    TracingHookConfig,
)
from agent.llm.types import LLMUsage  # noqa: E402


def _config() -> HooksConfig:
    return HooksConfig(
        logging=LoggingHookConfig(enabled=False),
        audit=AuditHookConfig(enabled=False),
        metrics=MetricsHookConfig(enabled=False),
        tracing=TracingHookConfig(enabled=False),
        token_usage=TokenUsageHookConfig(enabled=False),
    )


def _context(event: HookEvent, payload: HookPayload | None = None) -> HookContext:
    return HookContext(
        event=event,
        trace_id="trace-test",
        turn_id="turn-1",
        payload=payload or HookPayload(),
    )


def test_registry_rejects_duplicate() -> None:
    registry = HookRegistry()
    spec = HookSpec(name="dup", events=frozenset({HookEvent.RUN_START}))

    async def handler(_context: HookContext) -> None:
        return None

    registry.register(spec, handler)
    try:
        registry.register(spec, handler)
    except Exception as exc:
        assert type(exc).__name__ == "HookRegistrationError"
    else:
        raise AssertionError("duplicate registration should fail")


def test_dispatcher_order_and_background() -> None:
    async def run() -> None:
        registry = HookRegistry()
        calls: list[str] = []

        async def first(_context: HookContext) -> None:
            calls.append("first")

        async def second(_context: HookContext) -> None:
            calls.append("second")

        async def background(_context: HookContext) -> None:
            calls.append("background")

        registry.register(
            HookSpec(name="first", events=frozenset({HookEvent.RUN_START}), priority=10),
            first,
        )
        registry.register(
            HookSpec(name="second", events=frozenset({HookEvent.RUN_START}), priority=20),
            second,
        )
        registry.register(
            HookSpec(
                name="background",
                events=frozenset({HookEvent.RUN_START}),
                mode=HookExecutionMode.BACKGROUND,
            ),
            background,
        )
        dispatcher = HookDispatcher(registry, _config())
        try:
            result = await dispatcher.emit(_context(HookEvent.RUN_START))
            await dispatcher.flush()
        finally:
            await dispatcher.close()
        assert result.executed == 2
        assert calls[:2] == ["first", "second"]
        assert "background" in calls

    asyncio.run(run())


def test_timeout_is_fail_open() -> None:
    async def run() -> None:
        registry = HookRegistry()

        async def slow(_context: HookContext) -> None:
            await asyncio.sleep(0.1)

        registry.register(
            HookSpec(
                name="slow",
                events=frozenset({HookEvent.RUN_START}),
                timeout_s=0.01,
                failure_policy=HookFailurePolicy.FAIL_OPEN,
            ),
            slow,
        )
        dispatcher = HookDispatcher(registry, _config())
        try:
            result = await dispatcher.emit(_context(HookEvent.RUN_START))
        finally:
            await dispatcher.close()
        assert result.outcomes[0].status is HookStatus.TIMEOUT
        assert result.failed == 0

    asyncio.run(run())


def test_redaction() -> None:
    policy = RedactionPolicy(
        RedactionConfig(
            fields=["password"],
            patterns=[r"\d{11}"],
            max_text_length=100,
        )
    )
    value = policy.redact_value(
        {"password": "secret", "phone": "13800138000", "note": "ok"}
    )
    assert value["password"] == "***"
    assert value["phone"] == "***"
    assert value["note"] == "ok"


def test_token_usage_flush_without_redis() -> None:
    async def run() -> None:
        class Repo:
            def __init__(self) -> None:
                self.calls = []

            async def record_turn(self, conversation_id, **kwargs):
                self.calls.append((conversation_id, kwargs))

        repo = Repo()
        conversation_id = uuid4()
        hook = TokenUsageHook(
            TokenUsageHookConfig(redis_counters=False),
            conversation_repository=repo,
        )
        usage = LLMUsage(input_tokens=10, output_tokens=4, total_tokens=14)
        await hook(
            HookContext(
                event=HookEvent.LLM_USAGE,
                trace_id="trace",
                conversation_id=conversation_id,
                turn_id="turn",
                call_id="call-1",
                payload=LLMUsagePayload(
                    provider="p",
                    model="m",
                    kind=UsageKind.CHAT,
                    usage=usage,
                ),
            )
        )
        await hook(
            HookContext(
                event=HookEvent.TURN_END,
                trace_id="trace",
                conversation_id=conversation_id,
                turn_id="turn",
                payload=HookPayload(),
            )
        )
        assert repo.calls == [
            (conversation_id, {"input_tokens": 10, "output_tokens": 4})
        ]

    asyncio.run(run())
