"""工具执行流水线。

流水线只做编排，执行器只做业务：

1. 解析工具 -> 2. 权限校验 -> 3. 参数校验 -> 4. 幂等 -> 5. 缓存
-> 6. 熔断 -> 7. 限流 -> 8. 执行（超时/重试） -> 9. 截断/溢出
-> 10. 状态回写 -> 11. ToolResult

缓存/幂等/熔断/限流后端通过窄协议注入；未注入时对应能力自动跳过（测试或
早期阶段可只跑校验+执行）。异常不会穿透到 Agent 主循环。
"""

from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime, timedelta
from time import perf_counter
from typing import Any

from agent.message.models import ToolCall
from config.tools import ToolsConfig
from exception.tool import (
    ToolCancelledError,
    ToolErrorCode,
    ToolException,
    ToolIdempotencyConflictError,
)
from store.redis.keys import RedisKeyPrefix, make_key
from utils.hashing import sha256_json
from utils.serialization import json_dumps
from utils.time import utcnow
from utils.tokens import estimate_tokens, truncate_to_tokens

from .enums import (
    IdempotencyMode,
    IdempotencyState,
    ToolKind,
    ToolResultStatus,
    ToolSideEffect,
)
from .errors import tool_error_from_exception
from .models import (
    IdempotencyRecord,
    RegisteredTool,
    ToolContext,
    ToolError,
    ToolResult,
    ToolSelection,
)
from .permissions import check_resource_permissions, has_required_permission
from .policies import RetryPolicy
from .registry import ToolRegistry
from .selection import activate_group
from .state import (
    CircuitBreakerBackend,
    IdempotencyBackend,
    OverflowWriter,
    RateLimitBackend,
    ToolCacheStore,
)
from .validation import stable_arguments, validate_arguments

__all__ = ["ToolPipeline"]


class ToolPipeline:
    """工具执行流水线；一个进程内通常只创建一个。"""

    def __init__(
        self,
        registry: ToolRegistry,
        config: ToolsConfig,
        *,
        cache: ToolCacheStore | None = None,
        idempotency: IdempotencyBackend | None = None,
        circuit_breaker: CircuitBreakerBackend | None = None,
        rate_limiter: RateLimitBackend | None = None,
        overflow_writer: OverflowWriter | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.registry = registry
        self.config = config
        self.cache = cache
        self.idempotency = idempotency
        self.circuit_breaker = circuit_breaker
        self.rate_limiter = rate_limiter
        self.overflow_writer = overflow_writer
        self._logger = logger or logging.getLogger(__name__)

    # ——————————————————— 单次调用 ———————————————————

    async def execute(self, call: ToolCall, context: ToolContext) -> ToolResult:
        """执行一次工具调用；任何失败都转成 ToolResult。"""
        started = perf_counter()
        started_at = utcnow()
        try:
            return await self._execute(call, context, started, started_at)
        except asyncio.CancelledError:
            error = ToolError.from_exception(
                ToolCancelledError(tool_name=call.name)
            )
            return self._result(
                ToolResult.failure(
                    call_id=call.call_id,
                    tool_name=call.name,
                    error=error,
                    started_at=started_at,
                    ended_at=utcnow(),
                ),
                started,
                attempts=1,
            )
        except ToolException as exc:
            error = tool_error_from_exception(exc, tool_name=call.name)
            return self._result(
                ToolResult.failure(
                    call_id=call.call_id,
                    tool_name=call.name,
                    error=error,
                    started_at=started_at,
                    ended_at=utcnow(),
                ),
                started,
                attempts=1,
            )
        except Exception as exc:  # pragma: no cover - 防御
            self._logger.exception("工具执行未处理异常: %s", call.name)
            error = tool_error_from_exception(exc, tool_name=call.name)
            return self._result(
                ToolResult.failure(
                    call_id=call.call_id,
                    tool_name=call.name,
                    error=error,
                    started_at=started_at,
                    ended_at=utcnow(),
                ),
                started,
                attempts=1,
            )

    async def _execute(
        self,
        call: ToolCall,
        context: ToolContext,
        started: float,
        started_at: datetime,
    ) -> ToolResult:
        tool = self.registry.get(call.name)
        if tool is None:
            return self._failure(
                call,
                started,
                started_at,
                ToolErrorCode.TOOL_NOT_FOUND,
                f"工具不存在: {call.name}",
            )

        spec = tool.spec
        if not spec.enabled:
            return self._failure(
                call,
                started,
                started_at,
                ToolErrorCode.TOOL_DISABLED,
                f"工具已禁用: {spec.name}",
                tool_name=spec.name,
            )
        if not has_required_permission(spec, context.permissions):
            return self._failure(
                call,
                started,
                started_at,
                ToolErrorCode.PERMISSION_DENIED,
                f"没有调用工具 {spec.name} 的权限",
                tool_name=spec.name,
                details={"required_permission": spec.required_permission},
            )
        if call.raw_arguments and not call.arguments:
            return self._failure(
                call,
                started,
                started_at,
                ToolErrorCode.INVALID_ARGUMENTS,
                "工具参数不是合法 JSON 对象",
                tool_name=spec.name,
                details={"raw_arguments": call.raw_arguments},
            )

        # 入参大小保护：在 Pydantic/JSON Schema 校验前拦住超大对象
        try:
            raw = json_dumps(call.arguments)
        except TypeError as exc:
            return self._failure(
                call,
                started,
                started_at,
                ToolErrorCode.INVALID_ARGUMENTS,
                "工具参数无法序列化",
                tool_name=spec.name,
                cause=str(exc),
            )
        if len(raw.encode("utf-8")) > self.config.max_argument_bytes:
            return self._failure(
                call,
                started,
                started_at,
                ToolErrorCode.INVALID_ARGUMENTS,
                "工具参数过大",
                tool_name=spec.name,
                details={"max_argument_bytes": self.config.max_argument_bytes},
            )

        try:
            arguments = validate_arguments(tool, call.arguments)
            allow_partial = bool(
                spec.metadata.get("allow_partial_resources", False)
            )
            check_resource_permissions(
                spec,
                arguments,
                context.permissions,
                allow_partial=allow_partial,
            )
        except ToolException as exc:
            return self._failure_from_exception(call, started, started_at, exc)

        if spec.kind is ToolKind.ROUTER:
            return self._failure(
                call,
                started,
                started_at,
                ToolErrorCode.INTERNAL_ERROR,
                "router 工具必须通过 handle_router 处理",
                tool_name=spec.name,
            )

        allowed, reason = self._write_allowed(tool)
        if not allowed:
            return self._failure(
                call,
                started,
                started_at,
                ToolErrorCode.TOOL_DISABLED,
                reason or "v1 不允许自动执行写/删除工具",
                tool_name=spec.name,
            )
        if spec.requires_confirmation:
            if not self.config.approval_enabled:
                return self._failure(
                    call,
                    started,
                    started_at,
                    ToolErrorCode.TOOL_DISABLED,
                    "危险工具确认流程未启用",
                    tool_name=spec.name,
                )
            # v1 只预留确认模型，不实现暂停/恢复；有确认要求就先返回结构化错误
            return self._failure(
                call,
                started,
                started_at,
                ToolErrorCode.CONFIRMATION_REQUIRED,
                "该工具需要用户确认",
                tool_name=spec.name,
            )

        # 幂等 -> 缓存 -> 熔断 -> 限流（文档 §9）
        idem_key, idem_record = await self._begin_idempotency(
            tool, arguments, context
        )
        if idem_record is not None:
            return self._finalize_idempotent(
                call, started, started_at, idem_record
            )

        cache_key = self._cache_key(tool, arguments, context)
        if cache_key is not None:
            cached = await self._cache_get(cache_key)
            if cached is not None:
                result = self._clone_result(
                    cached,
                    call,
                    started,
                    started_at,
                    cache_hit=True,
                )
                await self._complete_idempotency(
                    idem_key, tool, context, call, result, ttl_s=0
                )
                return result

        if self.circuit_breaker is not None and tool.spec.policy.circuit_breaker.enabled:
            try:
                ok, retry_after = await self.circuit_breaker.allow(
                    spec.name, spec.policy.circuit_breaker
                )
            except Exception as exc:  # 状态后端不可用时 fail-open
                self._logger.warning("熔断状态读取失败，按放行处理: %s", exc)
                ok, retry_after = True, None
            if not ok:
                result = self._failure(
                    call,
                    started,
                    started_at,
                    ToolErrorCode.CIRCUIT_OPEN,
                    "工具暂不可用（熔断）",
                    tool_name=spec.name,
                    retry_after_s=retry_after,
                )
                await self._complete_idempotency(
                    idem_key, tool, context, call, result, ttl_s=0
                )
                return result

        if self.rate_limiter is not None and spec.policy.rate_limit.enabled:
            rate_key = make_key(RedisKeyPrefix.RATE_LIMIT, "tool", spec.name)
            try:
                ok, retry_after = await self.rate_limiter.acquire(
                    rate_key, spec.policy.rate_limit
                )
            except Exception as exc:
                self._logger.warning("限流状态读取失败，按放行处理: %s", exc)
                ok, retry_after = True, None
            if not ok:
                result = self._failure(
                    call,
                    started,
                    started_at,
                    ToolErrorCode.RATE_LIMITED,
                    "工具调用过于频繁",
                    tool_name=spec.name,
                    retry_after_s=retry_after,
                )
                await self._complete_idempotency(
                    idem_key, tool, context, call, result, ttl_s=0
                )
                return result

        result, attempts = await self._run_with_retry(
            tool, arguments, context, call, started_at
        )
        result = await self._apply_result_limits(tool, result, context, call)
        result = self._result(result, started, attempts=attempts)

        await self._record_state(
            tool, context, call, arguments, result, idem_key, attempts
        )
        return result

    # ——————————————————— Router ———————————————————

    async def handle_router(
        self,
        selection: ToolSelection,
        call: ToolCall,
        context: ToolContext,
    ) -> tuple[ToolResult, ToolSelection]:
        """处理 router 调用：激活组，返回成功结果与新 selection。"""
        started = perf_counter()
        started_at = utcnow()
        tool = self.registry.get(call.name)
        if tool is None or tool.spec.kind is not ToolKind.ROUTER:
            result = self._failure(
                call,
                started,
                started_at,
                ToolErrorCode.TOOL_NOT_FOUND,
                f"路由工具不存在: {call.name}",
            )
            return result, selection
        if not has_required_permission(tool.spec, context.permissions):
            result = self._failure(
                call,
                started,
                started_at,
                ToolErrorCode.PERMISSION_DENIED,
                f"没有调用路由工具 {call.name} 的权限",
                tool_name=call.name,
            )
            return result, selection

        try:
            updated = activate_group(
                self.registry.snapshot(),
                selection,
                tool.spec.router_group or "",
                context.permissions,
            )
        except ToolException as exc:
            result = self._failure_from_exception(
                call, started, started_at, exc
            )
            return result, selection

        result = self._result(
            ToolResult.success(
                call_id=call.call_id,
                tool_name=call.name,
                content=f"已激活工具组: {tool.spec.router_group}",
                data={
                    "group": tool.spec.router_group,
                    "tools": sorted(
                        item.name
                        for item in updated.descriptors
                        if item.name != call.name
                    ),
                },
                started_at=started_at,
                ended_at=utcnow(),
            ),
            started,
            attempts=1,
        )
        return result, updated

    # ——————————————————— 批量 ———————————————————

    async def execute_batch(
        self,
        calls: list[ToolCall],
        context: ToolContext,
        *,
        max_concurrency: int | None = None,
    ) -> list[ToolResult]:
        """并行执行一批 ToolCall；返回顺序与入参一致。"""
        limit = max_concurrency or self.config.max_batch_concurrency
        semaphore = asyncio.Semaphore(limit)

        async def run(call: ToolCall) -> ToolResult:
            tool = self.registry.get(call.name)
            timeout = tool.spec.policy.timeout_s if tool else 30.0
            call_context = context.model_copy(
                update={
                    "call_id": call.call_id,
                    "tool_name": call.name,
                    "timeout_s": timeout,
                }
            )
            async with semaphore:
                return await self.execute(call, call_context)

        return list(await asyncio.gather(*(run(call) for call in calls)))

    # ——————————————————— 执行 + 重试 ———————————————————

    async def _run_with_retry(
        self,
        tool: RegisteredTool,
        arguments: dict[str, Any],
        context: ToolContext,
        call: ToolCall,
        started_at: datetime,
    ) -> tuple[ToolResult, int]:
        spec = tool.spec
        policy = spec.policy
        attempts_allowed = policy.retry.max_attempts
        if not self._retry_allowed(tool):
            attempts_allowed = 1

        last_error: ToolError | None = None
        for attempt in range(1, attempts_allowed + 1):
            timeout = self._timeout_for(context, policy.timeout_s)
            if timeout is not None and timeout <= 0:
                last_error = ToolError(
                    code=ToolErrorCode.TIMEOUT,
                    message="工具调用超过 deadline",
                    retryable=False,
                )
                break
            try:
                coro = tool.executor(arguments, context)
                raw = (
                    await asyncio.wait_for(coro, timeout=timeout)
                    if timeout is not None
                    else await coro
                )
                result = self._normalize_result(raw, call, started_at)
            except asyncio.TimeoutError:
                last_error = ToolError(
                    code=ToolErrorCode.TIMEOUT,
                    message=f"工具执行超时（{timeout:.1f}s）",
                    retryable=True,
                )
                result = ToolResult.failure(
                    call_id=call.call_id,
                    tool_name=call.name,
                    error=last_error,
                    started_at=started_at,
                    ended_at=utcnow(),
                )
            except ToolCancelledError as exc:
                last_error = tool_error_from_exception(exc, tool_name=call.name)
                result = ToolResult.cancelled(
                    call_id=call.call_id,
                    tool_name=call.name,
                    error=last_error,
                    started_at=started_at,
                    ended_at=utcnow(),
                )
            except ToolException as exc:
                last_error = tool_error_from_exception(exc, tool_name=call.name)
                result = ToolResult.failure(
                    call_id=call.call_id,
                    tool_name=call.name,
                    error=last_error,
                    started_at=started_at,
                    ended_at=utcnow(),
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_error = tool_error_from_exception(exc, tool_name=call.name)
                result = ToolResult.failure(
                    call_id=call.call_id,
                    tool_name=call.name,
                    error=last_error,
                    started_at=started_at,
                    ended_at=utcnow(),
                )

            if result.status is ToolResultStatus.SUCCEEDED:
                return result, attempt
            if result.status is ToolResultStatus.CANCELLED:
                return result, attempt

            error = result.error or last_error
            if (
                attempt < attempts_allowed
                and error is not None
                and error.code in policy.retry.retry_on
            ):
                await self._sleep_backoff(policy.retry, attempt)
                continue
            return result, attempt

        error = last_error or ToolError(
            code=ToolErrorCode.INTERNAL_ERROR,
            message="工具执行失败",
        )
        return (
            ToolResult.failure(
                call_id=call.call_id,
                tool_name=call.name,
                error=error,
                started_at=started_at,
                ended_at=utcnow(),
            ),
            attempts_allowed,
        )

    # ——————————————————— 结果处理 ———————————————————

    async def _apply_result_limits(
        self,
        tool: RegisteredTool,
        result: ToolResult,
        context: ToolContext,
        call: ToolCall,
    ) -> ToolResult:
        limits = tool.spec.policy.result_limit
        content = result.content
        data = result.data
        truncated = result.truncated

        if not content and data is not None:
            content = json_dumps(data)
        elif not content and result.status is ToolResultStatus.SUCCEEDED:
            content = "工具执行成功"

        if content:
            clipped = truncate_to_tokens(content, limits.max_content_tokens)
            if len(clipped.encode("utf-8")) > limits.max_content_bytes:
                clipped = _truncate_bytes(clipped, limits.max_content_bytes)
            if clipped != content:
                content = clipped
                truncated = True

        data_text = json_dumps(data) if data is not None else ""
        if data_text and (
            estimate_tokens(data_text) > limits.max_data_tokens
            or len(data_text.encode("utf-8")) > limits.max_data_bytes
        ):
            preview = _truncate_bytes(data_text, min(2048, limits.max_data_bytes))
            data = {
                "truncated": True,
                "original_bytes": len(data_text.encode("utf-8")),
                "preview": preview,
            }
            data_text = json_dumps(data)
            truncated = True

        files = list(result.files)
        images = list(result.images)
        if len(files) > limits.max_files:
            files = files[: limits.max_files]
            truncated = True
        if len(images) > limits.max_images:
            images = images[: limits.max_images]
            truncated = True

        overflow_file = result.overflow_file
        if truncated and overflow_file is None:
            if not limits.overflow_enabled:
                note = "结果已截断"
                content = f"{content}\n\n[{note}]" if content else f"[{note}]"
            elif self.overflow_writer is None:
                content = (
                    f"{content}\n\n[结果已截断；完整结果未能保存]"
                    if content
                    else "结果已截断；完整结果未能保存"
                )
            else:
                try:
                    overflow_file = await self.overflow_writer.write(
                        conversation_id=context.conversation_id,
                        call_id=call.call_id,
                        tool_name=call.name,
                        content=result.content,
                        data=result.data,
                    )
                    note = "结果已截断，完整结果见附件"
                    content = f"{content}\n\n[{note}]" if content else f"[{note}]"
                except Exception as exc:
                    self._logger.warning("工具结果溢出写 MinIO 失败: %s", exc)
                    content = (
                        f"{content}\n\n[结果已截断；完整结果保存失败]"
                        if content
                        else "结果已截断；完整结果保存失败"
                    )

        if truncated:
            # 追加提示后仍要遵守硬上限，避免提示本身把结果撑爆
            content = truncate_to_tokens(content, limits.max_content_tokens)
            content = _truncate_bytes(content, limits.max_content_bytes)

        metrics = result.metrics.model_copy(
            update={
                "truncated": truncated,
                "output_bytes": len(content.encode("utf-8"))
                + len(data_text.encode("utf-8")),
            }
        )
        return result.model_copy(
            update={
                "content": content,
                "data": data,
                "files": files,
                "images": images,
                "overflow_file": overflow_file,
                "truncated": truncated,
                "metrics": metrics,
            }
        )

    # ——————————————————— 状态 ———————————————————

    async def _begin_idempotency(
        self,
        tool: RegisteredTool,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> tuple[str | None, IdempotencyRecord | None]:
        policy = tool.spec.policy.idempotency
        if policy.mode is IdempotencyMode.NONE or self.idempotency is None:
            return None, None
        if (
            policy.mode is IdempotencyMode.OPTIONAL
            and not context.idempotency_key
        ):
            # OPTIONAL 只在调用方提供业务幂等键时生效，不自动用参数生成 key
            return None, None

        key = self._idempotency_key(tool, arguments, context)
        try:
            existing = await self.idempotency.get(key)
        except Exception as exc:
            if tool.spec.side_effect is not ToolSideEffect.READ:
                raise ToolIdempotencyConflictError(
                    "幂等状态不可用，写工具拒绝执行",
                    tool_name=tool.spec.name,
                    details={"idempotency_key": key},
                    cause=str(exc),
                ) from exc
            self._logger.warning("幂等记录读取失败: %s", exc)
            existing = None

        if existing is not None:
            if (
                existing.state is IdempotencyState.SUCCEEDED
                and policy.store_result
                and existing.result is not None
            ):
                return key, existing
            if existing.is_locked():
                raise ToolIdempotencyConflictError(
                    "同一幂等 key 正在执行",
                    tool_name=tool.spec.name,
                    details={"idempotency_key": key},
                )
            if existing.state is IdempotencyState.IN_PROGRESS:
                # 锁已过期但状态仍是 IN_PROGRESS：不能假设没执行，按 UNKNOWN 对账
                raise ToolIdempotencyConflictError(
                    "幂等锁已过期，执行状态未知，需要先对账",
                    tool_name=tool.spec.name,
                    details={"idempotency_key": key},
                )
            if existing.state is IdempotencyState.UNKNOWN:
                raise ToolIdempotencyConflictError(
                    "幂等状态未知，需要先对账",
                    tool_name=tool.spec.name,
                    details={"idempotency_key": key},
                )
            if existing.error is not None:
                return key, existing

        now = utcnow()
        record = IdempotencyRecord(
            key=key,
            tool_name=tool.spec.name,
            user_id=context.user_id,
            call_id=context.call_id,
            state=IdempotencyState.IN_PROGRESS,
            created_at=now,
            expires_at=now + timedelta(seconds=policy.ttl_s),
            locked_until=now + timedelta(seconds=policy.lock_ttl_s),
        )
        try:
            ok = await self.idempotency.acquire(
                record, lock_ttl_s=policy.lock_ttl_s
            )
        except Exception as exc:
            if tool.spec.side_effect is not ToolSideEffect.READ:
                raise ToolIdempotencyConflictError(
                    "幂等锁不可用，写工具拒绝执行",
                    tool_name=tool.spec.name,
                    details={"idempotency_key": key},
                    cause=str(exc),
                ) from exc
            self._logger.warning("幂等锁获取失败，读工具继续执行: %s", exc)
            ok = True
        if not ok:
            raise ToolIdempotencyConflictError(
                "同一幂等 key 正在执行",
                tool_name=tool.spec.name,
                details={"idempotency_key": key},
            )
        return key, None

    def _finalize_idempotent(
        self,
        call: ToolCall,
        started: float,
        started_at: datetime,
        record: IdempotencyRecord,
    ) -> ToolResult:
        if record.result is not None:
            return self._clone_result(
                record.result,
                call,
                started,
                started_at,
                idempotent_replay=True,
            )
        error = record.error or ToolError(
            code=ToolErrorCode.INTERNAL_ERROR,
            message="幂等记录缺少结果",
        )
        result = ToolResult.failure(
            call_id=call.call_id,
            tool_name=call.name,
            error=error,
            started_at=started_at,
            ended_at=utcnow(),
        )
        return self._result(result, started, attempts=1)

    async def _complete_idempotency(
        self,
        key: str | None,
        tool: RegisteredTool,
        context: ToolContext,
        call: ToolCall,
        result: ToolResult,
        *,
        ttl_s: int,
    ) -> None:
        if key is None or self.idempotency is None:
            return
        policy = tool.spec.policy.idempotency
        now = utcnow()
        record = IdempotencyRecord(
            key=key,
            tool_name=tool.spec.name,
            user_id=context.user_id,
            call_id=call.call_id,
            state=(
                IdempotencyState.SUCCEEDED
                if result.status is ToolResultStatus.SUCCEEDED
                else IdempotencyState.FAILED
            ),
            result=result if policy.store_result else None,
            error=result.error,
            created_at=now,
            expires_at=now + timedelta(seconds=ttl_s or policy.ttl_s),
            locked_until=None,
        )
        try:
            await self.idempotency.complete(record, ttl_s=ttl_s or policy.ttl_s)
        except Exception as exc:
            self._logger.warning("幂等终态写入失败: %s", exc)

    async def _record_state(
        self,
        tool: RegisteredTool,
        context: ToolContext,
        call: ToolCall,
        arguments: dict[str, Any],
        result: ToolResult,
        idem_key: str | None,
        attempts: int,
    ) -> None:
        if idem_key is not None:
            await self._complete_idempotency(
                idem_key,
                tool,
                context,
                call,
                result,
                ttl_s=tool.spec.policy.idempotency.ttl_s,
            )

        policy = tool.spec.policy
        if (
            self.circuit_breaker is not None
            and policy.circuit_breaker.enabled
        ):
            try:
                if result.status is ToolResultStatus.SUCCEEDED:
                    await self.circuit_breaker.record_success(
                        tool.spec.name, policy.circuit_breaker
                    )
                elif (
                    result.error is not None
                    and result.error.code in policy.circuit_breaker.count_errors
                ):
                    await self.circuit_breaker.record_failure(
                        tool.spec.name, policy.circuit_breaker
                    )
            except Exception as exc:
                self._logger.warning("熔断状态回写失败: %s", exc)

        cache = self.cache
        cache_key = self._cache_key(tool, arguments, context)
        if (
            cache is not None
            and cache_key is not None
            and result.status is ToolResultStatus.SUCCEEDED
            and (result.content or result.data is not None or policy.cache.cache_empty)
        ):
            try:
                await cache.put(cache_key, result, policy.cache.ttl_s)
            except Exception as exc:
                self._logger.warning("工具结果写缓存失败: %s", exc)

    # ——————————————————— 小工具 ———————————————————

    def _write_allowed(self, tool: RegisteredTool) -> tuple[bool, str | None]:
        spec = tool.spec
        if spec.side_effect is ToolSideEffect.READ:
            return True, None
        policy = spec.policy.idempotency
        if (
            spec.requires_confirmation
            and policy.mode is IdempotencyMode.REQUIRED
            and spec.supports_reconcile
        ):
            return True, None
        return False, None

    def _retry_allowed(self, tool: RegisteredTool) -> bool:
        spec = tool.spec
        if spec.side_effect is ToolSideEffect.READ:
            return True
        policy = spec.policy.idempotency
        return (
            policy.mode is IdempotencyMode.REQUIRED
            and spec.supports_reconcile
        )

    def _cache_key(
        self,
        tool: RegisteredTool,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> str | None:
        policy = tool.spec.policy.cache
        if not policy.enabled or self.cache is None:
            return None
        if tool.spec.side_effect is not ToolSideEffect.READ:
            return None
        stable = stable_arguments(
            arguments,
            key_fields=policy.key_fields,
            exclude_fields=policy.exclude_fields,
        )
        payload: dict[str, Any] = {"arguments": stable}
        if tool.spec.resource_permission_template:
            # 资源级工具的结果与权限相关，key 必须带权限摘要，避免跨用户泄露
            payload["permissions"] = sorted(context.permissions)
        return make_key(
            RedisKeyPrefix.TOOL_CACHE,
            tool.spec.name,
            sha256_json(payload),
        )

    def _idempotency_key(
        self,
        tool: RegisteredTool,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> str:
        if context.idempotency_key:
            digest = sha256_json(
                {
                    "user_id": str(context.user_id),
                    "tool": tool.spec.name,
                    "idempotency_key": context.idempotency_key,
                }
            )
            return make_key(RedisKeyPrefix.IDEMPOTENCY, tool.spec.name, digest)
        policy = tool.spec.policy.idempotency
        stable = stable_arguments(
            arguments,
            key_fields=policy.key_fields,
            exclude_fields=policy.exclude_fields,
        )
        digest = sha256_json(
            {
                "user_id": str(context.user_id),
                "tool": tool.spec.name,
                "arguments": stable,
            }
        )
        return make_key(RedisKeyPrefix.IDEMPOTENCY, tool.spec.name, digest)

    async def _cache_get(self, key: str) -> ToolResult | None:
        cache = self.cache
        if cache is None:
            return None
        try:
            return await cache.get(key)
        except Exception as exc:
            self._logger.warning("工具缓存读取失败: %s", exc)
            return None

    def _normalize_result(
        self,
        result: Any,
        call: ToolCall,
        started_at: datetime,
    ) -> ToolResult:
        if not isinstance(result, ToolResult):
            raise ToolException(
                "执行器必须返回 ToolResult",
                tool_name=call.name,
            )
        return result.model_copy(
            update={
                "call_id": call.call_id,
                "tool_name": call.name,
                "started_at": result.started_at or started_at,
                "ended_at": result.ended_at or utcnow(),
            }
        )

    def _clone_result(
        self,
        result: ToolResult,
        call: ToolCall,
        started: float,
        started_at: datetime,
        *,
        cache_hit: bool = False,
        idempotent_replay: bool = False,
    ) -> ToolResult:
        cloned = result.model_copy(
            update={
                "call_id": call.call_id,
                "tool_name": call.name,
                "cache_hit": cache_hit,
                "idempotent_replay": idempotent_replay,
                "started_at": started_at,
                "ended_at": utcnow(),
            }
        )
        return self._result(cloned, started, attempts=result.metrics.attempts)

    def _result(
        self,
        result: ToolResult,
        started: float,
        *,
        attempts: int,
    ) -> ToolResult:
        duration_ms = int((perf_counter() - started) * 1000)
        metrics = result.metrics.model_copy(
            update={
                "duration_ms": duration_ms,
                "attempts": attempts,
                "cache_hit": result.cache_hit,
                "idempotent_replay": result.idempotent_replay,
                "truncated": result.truncated,
            }
        )
        return result.model_copy(
            update={
                "metrics": metrics,
                "ended_at": result.ended_at or utcnow(),
            }
        )

    def _failure(
        self,
        call: ToolCall,
        started: float,
        started_at: datetime,
        code: ToolErrorCode,
        message: str,
        *,
        tool_name: str | None = None,
        details: dict[str, Any] | None = None,
        retry_after_s: float | None = None,
        cause: str | None = None,
    ) -> ToolResult:
        error = ToolError(
            code=code,
            message=message,
            details=details or {},
            retry_after_s=retry_after_s,
            cause=cause,
        )
        return self._result(
            ToolResult.failure(
                call_id=call.call_id,
                tool_name=tool_name or call.name,
                error=error,
                started_at=started_at,
                ended_at=utcnow(),
            ),
            started,
            attempts=1,
        )

    def _failure_from_exception(
        self,
        call: ToolCall,
        started: float,
        started_at: datetime,
        exc: ToolException,
    ) -> ToolResult:
        error = tool_error_from_exception(exc, tool_name=call.name)
        return self._result(
            ToolResult.failure(
                call_id=call.call_id,
                tool_name=call.name,
                error=error,
                started_at=started_at,
                ended_at=utcnow(),
            ),
            started,
            attempts=1,
        )

    @staticmethod
    def _timeout_for(context: ToolContext, timeout_s: float) -> float | None:
        limit = timeout_s
        if context.timeout_s and context.timeout_s > 0:
            limit = min(limit, context.timeout_s)
        if context.deadline is None:
            return limit
        remaining = (context.deadline - utcnow()).total_seconds()
        return min(limit, remaining)

    @staticmethod
    async def _sleep_backoff(policy: RetryPolicy, attempt: int) -> None:
        delay_ms = policy.backoff_ms * (policy.backoff_multiplier ** (attempt - 1))
        if policy.jitter:
            delay_ms = random.uniform(0, delay_ms)
        if delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000)


def _truncate_bytes(text: str, max_bytes: int) -> str:
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return text
    return raw[:max_bytes].decode("utf-8", errors="ignore")
