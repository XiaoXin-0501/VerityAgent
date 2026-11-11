"""工具模块框架测试：注册、选择、执行、重试、缓存、幂等、截断。"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from uuid import UUID, uuid4

_ROOT = Path(__file__).resolve().parents[1]
_AGENT_ROOT = _ROOT / "verity-agent"
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

from agent.message.models import ToolCall  # noqa: E402
from agent.tools import (  # noqa: E402
    CachePolicy,
    IdempotencyMode,
    IdempotencyPolicy,
    RetryPolicy,
    ToolContext,
    ToolGroup,
    ToolKind,
    ToolPipeline,
    ToolPolicy,
    ToolRegistry,
    ToolResult,
    ToolResultStatus,
    ToolRoutingMode,
    ToolRoutingPolicy,
    ToolSelectionMode,
    ToolSpec,
    ToolSideEffect,
)
from agent.tools.models import IdempotencyRecord  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from config.tools import ToolsConfig  # noqa: E402
from exception.tool import ToolErrorCode, ToolExecutionError  # noqa: E402


def _spec(name: str, **overrides) -> ToolSpec:
    data = {
        "name": name,
        "description": name,
        "parameters": {"type": "object", "properties": {}},
    }
    data.update(overrides)
    return ToolSpec(**data)


def _context(call_id: str = "c1", **overrides) -> ToolContext:
    data = {
        "user_id": uuid4(),
        "conversation_id": uuid4(),
        "run_id": "run_1",
        "call_id": call_id,
        "tool_name": "echo",
        "trace_id": "trace_1",
    }
    data.update(overrides)
    return ToolContext(**data)


async def _echo(arguments, context):
    return ToolResult.success(
        call_id=context.call_id,
        tool_name=context.tool_name,
        content=str(arguments),
        data=arguments,
    )


def test_registry_permission_filter_and_duplicate() -> None:
    registry = ToolRegistry()
    registry.register_tool(_spec("public"), _echo)
    registry.register_tool(
        _spec("private", required_permission="tool:private"), _echo
    )
    selection = registry.snapshot().for_permissions(frozenset())
    assert [item.name for item in selection.descriptors] == ["public"]
    assert selection.mode is ToolSelectionMode.FLAT

    try:
        registry.register_tool(_spec("public"), _echo)
    except Exception as exc:
        assert getattr(exc, "code", "") == "TOOL_REGISTRATION_ERROR"
    else:  # pragma: no cover
        raise AssertionError("重复注册应失败")


def test_routed_selection_and_activation() -> None:
    registry = ToolRegistry()
    for index in range(16):
        registry.register_tool(_spec(f"tool_{index:02d}"), _echo)
    registry.register_group(
        ToolGroup(
            name="g",
            description="group g",
            tool_names=[f"tool_{index:02d}" for index in range(16)],
            router_tool_name="router_g",
        )
    )
    registry.register_tool(
        _spec(
            "router_g",
            description="router g",
            kind=ToolKind.ROUTER,
            router_group="g",
        ),
        _echo,
    )
    selection = registry.snapshot().for_permissions(
        frozenset(),
        ToolRoutingPolicy(
            mode=ToolRoutingMode.SINGLE_LEVEL,
            flat_threshold=15,
            max_exposed_tools=30,
        ),
    )
    assert selection.mode is ToolSelectionMode.ROUTED
    assert [item.name for item in selection.descriptors] == ["router_g"]

    async def run():
        pipeline = ToolPipeline(registry, ToolsConfig())
        result, updated = await pipeline.handle_router(
            selection,
            ToolCall(call_id="r1", name="router_g", arguments={}),
            _context("r1", tool_name="router_g"),
        )
        assert result.status is ToolResultStatus.SUCCEEDED
        assert updated.active_groups == {"g"}
        assert "router_g" not in updated.exposed_tool_names
        assert len(updated.exposed_tool_names) == 16

    asyncio.run(run())


def test_pipeline_validation_permission_and_not_found() -> None:
    registry = ToolRegistry()
    registry.register_tool(
        ToolSpec(
            name="needs_x",
            description="needs x",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            },
        ),
        _echo,
        argument_model=_NeedsXArgs,
    )
    registry.register_tool(
        _spec("needs_perm", required_permission="tool:perm"), _echo
    )
    pipeline = ToolPipeline(registry, ToolsConfig())

    async def run():
        context = _context()
        invalid = await pipeline.execute(
            ToolCall(call_id="c1", name="needs_x", arguments={}), context
        )
        assert invalid.status is ToolResultStatus.FAILED
        assert invalid.error is not None
        assert invalid.error.code is ToolErrorCode.INVALID_ARGUMENTS

        denied = await pipeline.execute(
            ToolCall(call_id="c2", name="needs_perm", arguments={}), context
        )
        assert denied.status is ToolResultStatus.FAILED
        assert denied.error is not None
        assert denied.error.code is ToolErrorCode.PERMISSION_DENIED

        missing = await pipeline.execute(
            ToolCall(call_id="c3", name="missing", arguments={}), context
        )
        assert missing.status is ToolResultStatus.FAILED
        assert missing.error is not None
        assert missing.error.code is ToolErrorCode.TOOL_NOT_FOUND

    asyncio.run(run())


def test_retry_and_timeout() -> None:
    attempts = {"n": 0}

    async def flaky(arguments, context):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise ToolExecutionError(
                "upstream", code=ToolErrorCode.UPSTREAM_ERROR
            )
        return ToolResult.success(
            call_id=context.call_id,
            tool_name=context.tool_name,
            content="ok",
        )

    async def slow(arguments, context):
        await asyncio.sleep(0.2)
        return ToolResult.success(
            call_id=context.call_id,
            tool_name=context.tool_name,
            content="late",
        )

    policy = ToolPolicy(
        retry=RetryPolicy(max_attempts=2, backoff_ms=0, jitter=False),
        timeout_s=0.05,
    )
    registry = ToolRegistry()
    registry.register_tool(_spec("flaky", policy=policy), flaky)
    registry.register_tool(_spec("slow", policy=policy), slow)
    pipeline = ToolPipeline(registry, ToolsConfig())

    async def run():
        context = _context()
        ok = await pipeline.execute(
            ToolCall(call_id="c1", name="flaky", arguments={}), context
        )
        assert ok.status is ToolResultStatus.SUCCEEDED
        assert ok.metrics.attempts == 2

        timeout = await pipeline.execute(
            ToolCall(call_id="c2", name="slow", arguments={}), context
        )
        assert timeout.status is ToolResultStatus.FAILED
        assert timeout.error is not None
        assert timeout.error.code is ToolErrorCode.TIMEOUT

    asyncio.run(run())


class _FakeCache:
    def __init__(self) -> None:
        self.values: dict[str, tuple[ToolResult, int]] = {}

    async def get(self, key: str) -> ToolResult | None:
        item = self.values.get(key)
        return item[0] if item else None

    async def put(self, key: str, result: ToolResult, ttl_s: int) -> None:
        self.values[key] = (result, ttl_s)


class _FakeIdempotency:
    def __init__(self) -> None:
        self.values: dict[str, IdempotencyRecord] = {}

    async def get(self, key: str) -> IdempotencyRecord | None:
        return self.values.get(key)

    async def acquire(self, record: IdempotencyRecord, *, lock_ttl_s: int) -> bool:
        if record.key in self.values:
            return False
        self.values[record.key] = record
        return True

    async def complete(self, record: IdempotencyRecord, *, ttl_s: int) -> None:
        self.values[record.key] = record


def test_cache_hit() -> None:
    calls = {"n": 0}

    async def counted(arguments, context):
        calls["n"] += 1
        return ToolResult.success(
            call_id=context.call_id,
            tool_name=context.tool_name,
            content=f"value={arguments.get('value')}",
        )

    policy = ToolPolicy(cache=CachePolicy(enabled=True, ttl_s=60))
    registry = ToolRegistry()
    registry.register_tool(
        ToolSpec(
            name="counted",
            description="counted",
            parameters={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
            policy=policy,
        ),
        counted,
        argument_model=_ValueArgs,
    )
    cache = _FakeCache()
    pipeline = ToolPipeline(registry, ToolsConfig(), cache=cache)

    async def run():
        context = _context()
        first = await pipeline.execute(
            ToolCall(call_id="c1", name="counted", arguments={"value": "a"}),
            context,
        )
        second = await pipeline.execute(
            ToolCall(call_id="c2", name="counted", arguments={"value": "a"}),
            context,
        )
        assert first.status is ToolResultStatus.SUCCEEDED
        assert second.status is ToolResultStatus.SUCCEEDED
        assert second.cache_hit is True
        assert calls["n"] == 1

    asyncio.run(run())


def test_idempotency_replay() -> None:
    calls = {"n": 0}

    async def counted(arguments, context):
        calls["n"] += 1
        return ToolResult.success(
            call_id=context.call_id,
            tool_name=context.tool_name,
            content=f"value={arguments.get('value')}",
        )

    policy = ToolPolicy(
        idempotency=IdempotencyPolicy(mode=IdempotencyMode.REQUIRED, ttl_s=60)
    )
    registry = ToolRegistry()
    registry.register_tool(
        ToolSpec(
            name="counted",
            description="counted",
            parameters={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
            policy=policy,
        ),
        counted,
        argument_model=_ValueArgs,
    )
    idempotency = _FakeIdempotency()
    pipeline = ToolPipeline(
        registry,
        ToolsConfig(),
        idempotency=idempotency,
    )

    async def run():
        context = _context()
        first = await pipeline.execute(
            ToolCall(call_id="c1", name="counted", arguments={"value": "a"}),
            context,
        )
        second = await pipeline.execute(
            ToolCall(call_id="c2", name="counted", arguments={"value": "a"}),
            context,
        )
        assert first.status is ToolResultStatus.SUCCEEDED
        assert second.status is ToolResultStatus.SUCCEEDED
        assert second.idempotent_replay is True
        assert calls["n"] == 1

    asyncio.run(run())


def test_result_truncation() -> None:
    async def huge(arguments, context):
        return ToolResult.success(
            call_id=context.call_id,
            tool_name=context.tool_name,
            content="x" * 5000,
            data={"rows": ["y" * 100] * 100},
        )

    policy = ToolPolicy(
        result_limit={
            "max_content_tokens": 50,
            "max_data_tokens": 50,
            "max_content_bytes": 200,
            "max_data_bytes": 200,
            "overflow_enabled": False,
        }
    )
    registry = ToolRegistry()
    registry.register_tool(_spec("huge", policy=policy), huge)
    pipeline = ToolPipeline(registry, ToolsConfig())

    async def run():
        result = await pipeline.execute(
            ToolCall(call_id="c1", name="huge", arguments={}), _context()
        )
        assert result.status is ToolResultStatus.SUCCEEDED
        assert result.truncated is True
        assert len(result.content.encode("utf-8")) <= 200
        assert result.data["truncated"] is True

    asyncio.run(run())


def test_execute_batch_preserves_order() -> None:
    async def fail(arguments, context):
        raise ToolExecutionError("boom")

    registry = ToolRegistry()
    registry.register_tool(_spec("ok"), _echo)
    registry.register_tool(_spec("fail"), fail)
    pipeline = ToolPipeline(registry, ToolsConfig())

    async def run():
        calls = [
            ToolCall(call_id="c1", name="ok", arguments={}),
            ToolCall(call_id="c2", name="fail", arguments={}),
        ]
        results = await pipeline.execute_batch(calls, _context())
        assert [item.call_id for item in results] == ["c1", "c2"]
        assert results[0].status is ToolResultStatus.SUCCEEDED
        assert results[1].status is ToolResultStatus.FAILED

    asyncio.run(run())


class _EchoArgs(BaseModel):
    x: str


class _NeedsXArgs(BaseModel):
    x: str


class _ValueArgs(BaseModel):
    value: str


def test_argument_model_derives_schema_and_strict_extra() -> None:
    registry = ToolRegistry()
    registry.register_tool(
        ToolSpec(name="typed", description="typed"),
        _echo,
        argument_model=_EchoArgs,
    )
    tool = registry.get("typed")
    assert tool is not None
    assert "x" in tool.spec.parameters.get("properties", {})
    assert tool.spec.strict is True

    pipeline = ToolPipeline(registry, ToolsConfig())

    async def run():
        result = await pipeline.execute(
            ToolCall(
                call_id="c1",
                name="typed",
                arguments={"x": "ok", "extra": 1},
            ),
            _context(),
        )
        assert result.status is ToolResultStatus.FAILED
        assert result.error is not None
        assert result.error.code is ToolErrorCode.INVALID_ARGUMENTS

    asyncio.run(run())


def test_schema_only_tool_uses_jsonschema() -> None:
    registry = ToolRegistry()
    registry.register_tool(
        ToolSpec(
            name="dynamic",
            description="dynamic",
            parameters={
                "type": "object",
                "properties": {"n": {"type": "integer"}},
                "required": ["n"],
            },
            metadata={"schema_only": True},
        ),
        _echo,
    )
    pipeline = ToolPipeline(registry, ToolsConfig())

    async def run():
        bad = await pipeline.execute(
            ToolCall(call_id="c1", name="dynamic", arguments={"n": "bad"}),
            _context(),
        )
        assert bad.status is ToolResultStatus.FAILED
        assert bad.error is not None
        assert bad.error.code is ToolErrorCode.INVALID_ARGUMENTS

        good = await pipeline.execute(
            ToolCall(call_id="c2", name="dynamic", arguments={"n": 2}),
            _context(),
        )
        assert good.status is ToolResultStatus.SUCCEEDED

    asyncio.run(run())


def test_executable_without_model_or_schema_only_is_rejected() -> None:
    registry = ToolRegistry()
    try:
        registry.register_tool(
            ToolSpec(
                name="bad",
                description="bad",
                parameters={
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                },
            ),
            _echo,
        )
    except Exception as exc:
        assert getattr(exc, "code", "") == "TOOL_REGISTRATION_ERROR"
    else:  # pragma: no cover
        raise AssertionError("有参数但没有 argument_model 的 executable 工具应拒绝注册")
