"""Hooks 数据模型。

事件、上下文、payload、Hook 声明、执行结果、统计、用量/成本、指标/追踪/审计。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent.llm import LLMUsage, ModelPricing, ModelPurpose
from utils.ids import uuid7
from utils.time import utcnow

from .enum import (
    AuditResult,
    HookEvent,
    HookExecutionMode,
    HookFailurePolicy,
    HookSource,
    HookStatus,
    MetricKind,
    SpanStatus,
    UsageCounterScope,
    UsageKind,
)

__all__ = [
    "HookPayload",
    "RunPayload",
    "TurnPayload",
    "LLMRequestPayload",
    "LLMUsagePayload",
    "LLMErrorPayload",
    "LLMStreamPayload",
    "ToolSelectionPayload",
    "ToolCallPayload",
    "ToolResultPayload",
    "ToolPolicyPayload",
    "MessagePayload",
    "CompressionPayload",
    "MemoryPayload",
    "RagPayload",
    "FilePayload",
    "ContextPayload",
    "EmbeddingRequestPayload",
    "EmbeddingUsagePayload",
    "RerankRequestPayload",
    "RerankUsagePayload",
    "PromptPayload",
    "PromptLogPayload",
    "HookContext",
    "HookSpec",
    "RegisteredHook",
    "HookOutcome",
    "HookDispatchResult",
    "HookStats",
    "CostRecord",
    "UsageRecord",
    "UsageAccumulator",
    "UsageSnapshot",
    "UsageCounter",
    "MetricRecord",
    "SpanRecord",
    "AuditEvent",
    "HookHandler",
]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class HookPayload(_Model):
    """payload 基类；子类按事件域定义字段。"""

    model_config = ConfigDict(
        extra="allow", frozen=True, populate_by_name=True
    )


class RunPayload(HookPayload):
    run_id: str | None = None
    status: str | None = None
    turn_count: int = 0
    usage: LLMUsage | None = None
    duration_ms: int | None = None


class TurnPayload(HookPayload):
    turn_id: str | None = None
    ordinal: int | None = None
    user_message_id: UUID | None = None
    message_count: int = 0
    usage: LLMUsage | None = None
    status: str | None = None
    duration_ms: int | None = None


class LLMRequestPayload(HookPayload):
    provider: str | None = None
    model: str | None = None
    purpose: ModelPurpose | None = None
    message_count: int = 0
    tool_count: int = 0
    stream: bool = False
    token_estimate: int | None = None


class LLMUsagePayload(HookPayload):
    provider: str | None = None
    model: str | None = None
    kind: UsageKind = UsageKind.CHAT
    purpose: ModelPurpose | None = None
    usage: LLMUsage | None = None
    latency_ms: int | None = None
    finish_reason: str | None = None
    cost: "CostRecord | None" = None
    model_id: str | None = None


class LLMErrorPayload(HookPayload):
    provider: str | None = None
    model: str | None = None
    purpose: ModelPurpose | None = None
    error_code: str | None = None
    retryable: bool = False
    attempt: int = 1


class LLMStreamPayload(HookPayload):
    provider: str | None = None
    model: str | None = None
    chunk_type: str | None = None
    chunk_length: int = 0
    index: int | None = None


class ToolSelectionPayload(HookPayload):
    mode: str | None = None
    tool_count: int = 0
    group_count: int = 0
    permissions_count: int = 0


class ToolCallPayload(HookPayload):
    tool_name: str | None = None
    call_id: str | None = None
    required_permission: str | None = None
    side_effect: str | None = None
    argument_keys: list[str] = Field(default_factory=list)
    argument_bytes: int = 0


class ToolResultPayload(HookPayload):
    tool_name: str | None = None
    call_id: str | None = None
    status: str | None = None
    duration_ms: int | None = None
    content_bytes: int = 0
    data_bytes: int = 0
    file_count: int = 0
    image_count: int = 0
    error_code: str | None = None
    cache_hit: bool = False
    idempotent_replay: bool = False
    truncated: bool = False


class ToolPolicyPayload(HookPayload):
    tool_name: str | None = None
    reason: str | None = None
    error_code: str | None = None
    retry_after_s: float | None = None


class MessagePayload(HookPayload):
    message_id: UUID | None = None
    role: str | None = None
    status: str | None = None
    content_bytes: int = 0
    image_count: int = 0
    file_count: int = 0
    tool_call_count: int = 0


class CompressionPayload(HookPayload):
    covered_before: int | None = None
    covered_after: int | None = None
    message_count: int = 0
    usage: LLMUsage | None = None
    status: str | None = None
    attempts: int = 0
    failure_reason: str | None = None
    retryable: bool = False
    user_visible: bool = False
    duration_ms: int | None = None


class MemoryPayload(HookPayload):
    user_id: UUID | None = None
    section: str | None = None
    revision: int | None = None
    fact_count: int | None = None
    change_count: int = 0
    duration_ms: int | None = None


class RagPayload(HookPayload):
    query_preview: str | None = None
    kb_ids: list[str] = Field(default_factory=list)
    top_k: int | None = None
    candidate_count: int = 0
    final_count: int = 0
    rerank_count: int = 0
    duration_ms: int | None = None


class FilePayload(HookPayload):
    file_id: UUID | None = None
    filename: str | None = None
    mime_type: str | None = None
    size_bytes: int = 0
    status: str | None = None
    duration_ms: int | None = None


class ContextPayload(HookPayload):
    section_count: int = 0
    sources: list[str] = Field(default_factory=list)
    total_tokens: int = 0
    dropped: int = 0
    truncated: bool = False
    warnings: list[str] = Field(default_factory=list)


class EmbeddingRequestPayload(HookPayload):
    model: str | None = None
    batch_size: int = 0
    input_type: str | None = None
    duration_ms: int | None = None


class EmbeddingUsagePayload(HookPayload):
    model: str | None = None
    input_tokens: int = 0
    total_tokens: int = 0
    source: str | None = None
    duration_ms: int | None = None


class RerankRequestPayload(HookPayload):
    model: str | None = None
    document_count: int = 0
    top_k: int | None = None
    duration_ms: int | None = None


class RerankUsagePayload(HookPayload):
    model: str | None = None
    query_tokens: int = 0
    document_tokens: int = 0
    total_tokens: int = 0
    document_count: int = 0
    duration_ms: int | None = None


class PromptPayload(HookPayload):
    message_count: int = 0
    system_section_count: int = 0
    tool_count: int = 0
    token_estimate: int | None = None


class PromptLogPayload(HookPayload):
    plan_id: str | None = None
    request_id: str | None = None
    provider: str | None = None
    model: str | None = None
    sections: list[dict[str, Any]] = Field(default_factory=list)
    working_messages: list[dict[str, Any]] = Field(default_factory=list)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    rendered_prompt: str | None = None
    redaction_version: str | None = None


class HookContext(_Model):
    """一次事件分发的公共信封；不可变。"""

    event: HookEvent
    event_id: UUID = Field(default_factory=uuid7)
    timestamp: datetime = Field(default_factory=utcnow)
    trace_id: str
    span_id: str | None = None
    parent_span_id: str | None = None
    source: HookSource = HookSource.SYSTEM
    user_id: UUID | None = None
    conversation_id: UUID | None = None
    turn_id: str | None = None
    run_id: str | None = None
    call_id: str | None = None
    payload: HookPayload = Field(default_factory=HookPayload)
    metadata: dict[str, Any] = Field(default_factory=dict)
    duration_ms: int | None = None


class HookSpec(_Model):
    name: str = Field(min_length=1)
    events: frozenset[HookEvent]
    priority: int = 100
    mode: HookExecutionMode = HookExecutionMode.INLINE
    failure_policy: HookFailurePolicy = HookFailurePolicy.FAIL_OPEN
    timeout_s: float = Field(default=1.0, gt=0)
    sampling_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    enabled: bool = True
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("events")
    @classmethod
    def _events_required(cls, value: frozenset[HookEvent]) -> frozenset[HookEvent]:
        """HookSpec 至少订阅一个事件。"""
        if not value:
            raise ValueError("HookSpec.events 至少订阅一个事件")
        return value


HookHandler = Callable[[HookContext], Awaitable[None]]


@dataclass(slots=True)
class RegisteredHook:
    spec: HookSpec
    handler: HookHandler
    registered_at: datetime = field(default_factory=utcnow)
    source: str | None = None
    sequence: int = 0


class HookOutcome(_Model):
    hook_name: str
    status: HookStatus
    duration_ms: int = 0
    error: str | None = None


class HookDispatchResult(_Model):
    model_config = ConfigDict(extra="forbid", frozen=False, populate_by_name=True)

    event: HookEvent
    selected: int = 0
    executed: int = 0
    failed: int = 0
    skipped: int = 0
    duration_ms: int = 0
    outcomes: list[HookOutcome] = Field(default_factory=list)


@dataclass(slots=True)
class HookStats:
    hook_name: str
    event: HookEvent
    executed: int = 0
    failed: int = 0
    timed_out: int = 0
    total_duration_ms: int = 0
    last_error: str | None = None


class CostRecord(_Model):
    input_cost: float = 0.0
    output_cost: float = 0.0
    cached_cost: float = 0.0
    total_cost: float = 0.0
    currency: str = "USD"
    pricing_version: str = "v1"


class UsageRecord(_Model):
    usage_id: UUID = Field(default_factory=uuid7)
    trace_id: str
    conversation_id: UUID | None = None
    turn_id: str | None = None
    call_id: str | None = None
    kind: UsageKind = UsageKind.CHAT
    purpose: ModelPurpose | None = None
    provider: str | None = None
    model: str | None = None
    usage: LLMUsage
    cost: CostRecord | None = None
    timestamp: datetime = Field(default_factory=utcnow)


@dataclass(slots=True)
class UsageAccumulator:
    key: str
    usage: LLMUsage = field(default_factory=LLMUsage)
    cost: CostRecord | None = None
    call_ids: set[str] = field(default_factory=set)
    updated_at: datetime = field(default_factory=utcnow)


class UsageSnapshot(_Model):
    conversation_id: UUID | None = None
    turn_id: str | None = None
    usage: LLMUsage = Field(default_factory=LLMUsage)
    cost: CostRecord | None = None
    call_count: int = 0


class UsageCounter(_Model):
    scope: UsageCounterScope
    key: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    total_cost: float = 0.0
    call_count: int = 0
    updated_at: datetime = Field(default_factory=utcnow)
    expires_at: datetime | None = None


class MetricRecord(_Model):
    name: str
    kind: MetricKind
    value: float
    unit: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=utcnow)


class SpanRecord(_Model):
    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    name: str
    started_at: datetime = Field(default_factory=utcnow)
    ended_at: datetime | None = None
    status: SpanStatus = SpanStatus.OK
    attributes: dict[str, Any] = Field(default_factory=dict)


class AuditEvent(_Model):
    audit_id: UUID = Field(default_factory=uuid7)
    timestamp: datetime = Field(default_factory=utcnow)
    trace_id: str
    actor_id: UUID | None = None
    actor_type: str = "user"
    action: str
    resource_type: str
    resource_id: str | None = None
    result: AuditResult = AuditResult.SUCCESS
    details: dict[str, Any] = Field(default_factory=dict)


def calculate_cost(
    usage: LLMUsage,
    pricing: ModelPricing | None,
) -> CostRecord | None:
    """按 ModelPricing 计算单次成本；无定价返回 None。"""
    if pricing is None:
        return None
    input_cost = usage.input_tokens / 1_000_000 * pricing.input_per_million
    output_cost = usage.output_tokens / 1_000_000 * pricing.output_per_million
    cached_cost = (
        usage.cached_input_tokens / 1_000_000 * pricing.cached_input_per_million
        if pricing.cached_input_per_million is not None
        else 0.0
    )
    return CostRecord(
        input_cost=input_cost,
        output_cost=output_cost,
        cached_cost=cached_cost,
        total_cost=input_cost + output_cost + cached_cost,
        currency=pricing.currency,
    )
