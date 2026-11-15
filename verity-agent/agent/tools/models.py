"""工具模块数据模型。

分层与《工具模块数据结构设计》§0.1 对齐：

- 定义层：``ToolSpec / ToolPolicy / ToolDescriptor``
- 注册层：``RegisteredTool / ToolCatalog``
- 选择层：``ToolGroup / ToolSelection``
- 调用层：``ToolInvocation / ToolContext / ToolBatch``
- 结果层：``ToolResult / ToolError / ToolMetrics``
- 状态层：``ToolCacheEntry / IdempotencyRecord / CircuitBreakerRecord / RateLimitRecord``

消息模块的 ``ToolCall / ToolFile / ToolImage / Artifact`` 直接复用，不重复定义。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agent.message.enum import MessageStatus, ToolCallStatus
from agent.message.models import Artifact, ToolFile, ToolImage
from exception.tool import (
    TOOL_ERROR_CATEGORIES,
    TOOL_ERROR_RETRYABLE,
    ToolErrorCategory,
    ToolErrorCode,
    ToolException,
    ToolNotFoundError,
)
from utils.time import utcnow

from .enums import (
    IdempotencyState,
    ToolApprovalStatus,
    ToolBatchStatus,
    ToolCircuitState,
    ToolKind,
    ToolResultStatus,
    ToolSelectionMode,
    ToolSideEffect,
)
from .policies import ToolPolicy, ToolRoutingPolicy

__all__ = [
    "ToolSpec",
    "ToolDescriptor",
    "ToolGroup",
    "ToolGroupDescriptor",
    "RegisteredTool",
    "ToolCatalog",
    "ToolSelection",
    "ToolExecutor",
    "ToolContext",
    "ToolError",
    "ToolMetrics",
    "ToolResult",
    "ToolInvocation",
    "ToolBatch",
    "ToolCacheEntry",
    "IdempotencyRecord",
    "CircuitBreakerRecord",
    "RateLimitRecord",
    "ToolExecutionRecord",
    "ToolApprovalRequest",
    "ToolApprovalDecision",
    "classify_batch_status",
]

_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_PERMISSION_RE = re.compile(r"^[A-Za-z0-9_*:{}.\-]+$")
_TEMPLATE_FIELD_RE = re.compile(r"{([a-zA-Z_][a-zA-Z0-9_]*)}")


class _Model(BaseModel):
    """不可变值对象。"""

    model_config = ConfigDict(
        extra="forbid", frozen=True, populate_by_name=True
    )


class _MutableModel(BaseModel):
    """运行期可变对象；只用于 ToolSelection / ToolInvocation / ToolBatch。"""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
    )


# ——————————————————— 定义层 ———————————————————

class ToolSpec(_Model):
    """工具的静态定义；不包含执行器，可序列化、可展示。"""

    # LLM 调用名；snake_case，注册表内唯一
    name: str = Field(min_length=1, max_length=64)
    # 给 LLM 的用途/边界/返回说明
    description: str = Field(min_length=1)
    # LLM-facing JSON Schema；有 argument_model 时由注册表从模型生成
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
    # 是否禁止额外参数；映射到 provider 的 strict
    strict: bool = True
    # 路由组名，如 kb / data / dev
    group: str | None = None
    # 静态能力码，如 tool:sql:read；为 null 表示登录用户可用
    required_permission: str | None = None
    # 资源级权限模板，如 kb:read:{kb_id}
    resource_permission_template: str | None = None
    # 副作用等级；v1 只自动执行 READ
    side_effect: ToolSideEffect = ToolSideEffect.READ
    # 是否执行前需要用户确认
    requires_confirmation: bool = False
    # 写工具是否支持按 idempotency_key 对账
    supports_reconcile: bool = False
    # 是否注册/暴露
    enabled: bool = True
    # 工具版本；破坏性 schema 变更时递增
    version: str = "1"
    # EXECUTABLE / ROUTER
    kind: ToolKind = ToolKind.EXECUTABLE
    # kind=ROUTER 时指向的 ToolGroup
    router_group: str | None = None
    # 检索/管理标签
    tags: list[str] = Field(default_factory=list)
    # 参数示例；默认不发给 LLM
    examples: list[dict[str, Any]] = Field(default_factory=list)
    # 横切策略
    policy: ToolPolicy = Field(default_factory=ToolPolicy)
    # 内部扩展；不发给 LLM
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        if not _NAME_RE.fullmatch(value):
            raise ValueError("工具名必须匹配 ^[a-zA-Z0-9_-]{1,64}$")
        return value

    @field_validator("required_permission")
    @classmethod
    def _valid_permission(cls, value: str | None) -> str | None:
        if value is not None and not _PERMISSION_RE.fullmatch(value):
            raise ValueError("required_permission 只能包含权限码字符")
        return value

    @field_validator("resource_permission_template")
    @classmethod
    def _valid_template(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not _PERMISSION_RE.fullmatch(value):
            raise ValueError("resource_permission_template 只能是权限码模板")
        if not _TEMPLATE_FIELD_RE.search(value):
            raise ValueError("resource_permission_template 必须包含 {field} 占位")
        return value

    @field_validator("parameters")
    @classmethod
    def _valid_schema(cls, value: dict[str, Any]) -> dict[str, Any]:
        if value.get("type") not in (None, "object"):
            raise ValueError("ToolSpec.parameters 必须是 object schema")
        properties = value.get("properties", {})
        if not isinstance(properties, dict):
            raise ValueError("ToolSpec.parameters.properties 必须是对象")
        required = value.get("required", [])
        if not isinstance(required, list) or any(
            not isinstance(item, str) for item in required
        ):
            raise ValueError("ToolSpec.parameters.required 必须是字符串数组")
        return value

    @model_validator(mode="after")
    def _router_consistent(self) -> "ToolSpec":
        if self.kind is ToolKind.ROUTER and not self.router_group:
            raise ValueError("kind=ROUTER 必须提供 router_group")
        if self.kind is ToolKind.EXECUTABLE and self.router_group:
            raise ValueError("kind=EXECUTABLE 不能设置 router_group")
        return self


class ToolDescriptor(_Model):
    """ToolSpec 的 LLM-facing 投影；不包含权限/策略/执行器。"""

    name: str
    description: str
    parameters: dict[str, Any]
    strict: bool = True
    kind: ToolKind = ToolKind.EXECUTABLE
    router_group: str | None = None

    @classmethod
    def from_spec(cls, spec: ToolSpec) -> "ToolDescriptor":
        return cls(
            name=spec.name,
            description=spec.description,
            parameters=spec.parameters,
            strict=spec.strict,
            kind=spec.kind,
            router_group=spec.router_group,
        )


# ——————————————————— 注册层 ———————————————————

@dataclass(slots=True, frozen=True)
class RegisteredTool:
    """进程内注册项：静态定义 + 执行器 + 参数模型。"""

    spec: ToolSpec
    executor: "ToolExecutor"
    argument_model: type[BaseModel] | None = None
    source: str | None = None
    registered_at: datetime = field(default_factory=utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


class ToolGroup(_Model):
    """工具路由组。"""

    name: str = Field(min_length=1, max_length=64)
    description: str = Field(min_length=1)
    # 组内工具名；仅用于路由展开，不改变 registry
    tool_names: list[str] = Field(default_factory=list)
    # 组级权限；为空继承各工具
    required_permission: str | None = None
    # 路由工具名，如 router_kb
    router_tool_name: str = Field(min_length=1, max_length=64)
    # 是否启用
    enabled: bool = True
    # 内部扩展
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        if not _NAME_RE.fullmatch(value):
            raise ValueError("组名必须匹配 ^[a-zA-Z0-9_-]{1,64}$")
        return value

    @field_validator("tool_names")
    @classmethod
    def _unique_tools(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("ToolGroup.tool_names 不能重复")
        invalid = [item for item in value if not _NAME_RE.fullmatch(item)]
        if invalid:
            raise ValueError(f"ToolGroup.tool_names 含非法名称: {invalid}")
        return value

    @field_validator("router_tool_name")
    @classmethod
    def _valid_router_name(cls, value: str) -> str:
        if not _NAME_RE.fullmatch(value):
            raise ValueError("router_tool_name 必须匹配工具名规则")
        return value

    @field_validator("required_permission")
    @classmethod
    def _valid_group_permission(cls, value: str | None) -> str | None:
        if value is not None and not _PERMISSION_RE.fullmatch(value):
            raise ValueError("required_permission 只能包含权限码字符")
        return value


class ToolGroupDescriptor(_Model):
    """组的只读投影；不单独下发给 LLM（D-35）。"""

    name: str
    description: str
    router_tool_name: str
    tool_names: list[str] = Field(default_factory=list)

    @classmethod
    def from_group(cls, group: ToolGroup) -> "ToolGroupDescriptor":
        return cls(
            name=group.name,
            description=group.description,
            router_tool_name=group.router_tool_name,
            tool_names=list(group.tool_names),
        )


@dataclass(slots=True, frozen=True)
class ToolCatalog:
    """注册表的只读快照；运行期不修改。"""

    tools: dict[str, RegisteredTool]
    groups: dict[str, ToolGroup]
    version: str
    created_at: datetime = field(default_factory=utcnow)

    def get(self, name: str) -> RegisteredTool | None:
        return self.tools.get(name)

    def require(self, name: str) -> RegisteredTool:
        tool = self.tools.get(name)
        if tool is None:
            raise ToolNotFoundError(name)
        return tool

    def get_group(self, name: str) -> ToolGroup | None:
        return self.groups.get(name)

    def for_permissions(
        self,
        permissions: frozenset[str],
        routing: ToolRoutingPolicy | None = None,
    ) -> "ToolSelection":
        # 延迟导入，避免 registry/selection 与 models 循环
        from .selection import build_selection

        return build_selection(self, permissions, routing or ToolRoutingPolicy())


# ——————————————————— 选择层 ———————————————————

class ToolSelection(_MutableModel):
    """某次运行实际暴露给 LLM 的工具集合。"""

    catalog_version: str
    mode: ToolSelectionMode = ToolSelectionMode.FLAT
    descriptors: list[ToolDescriptor] = Field(default_factory=list)
    # 只用于管理面/调试；不直接下发给 LLM
    groups: list[ToolGroupDescriptor] = Field(default_factory=list)
    active_groups: set[str] = Field(default_factory=set)
    exposed_tool_names: set[str] = Field(default_factory=set)
    # 路由轮次，从 0 开始
    round: int = Field(default=0, ge=0)

    @property
    def descriptor_by_name(self) -> dict[str, ToolDescriptor]:
        return {item.name: item for item in self.descriptors}

    def is_exposed(self, name: str) -> bool:
        return name in self.exposed_tool_names


# ——————————————————— 调用层 ———————————————————

class ToolContext(_Model):
    """执行器拿到的运行期上下文；不序列化。"""

    user_id: UUID
    username: str = ""
    permissions: frozenset[str] = Field(default_factory=frozenset)
    conversation_id: UUID
    # 当前回合（运行期）；v1 Run 与 Turn 1:1
    turn_id: str | None = None
    run_id: str
    call_id: str
    tool_name: str
    trace_id: str
    # 整次调用/整批的截止时间
    deadline: datetime | None = None
    # 本次工具超时；None 表示使用 ToolPolicy.timeout_s
    timeout_s: float | None = None
    # 幂等 key；由幂等层或调用方生成
    idempotency_key: str | None = None
    # 取消信号（运行期对象，如 asyncio.Event）
    cancel_token: Any = None
    # DB/Redis/检索/存储等依赖；执行器通过它取，不 import 全局单例
    services: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    logger: Any = None

    def has_permission(self, code: str | None) -> bool:
        return code is None or code in self.permissions


class ToolError(_Model):
    """结构化工具错误；进入 ToolResult，可回喂 LLM。"""

    code: ToolErrorCode
    category: ToolErrorCategory = ToolErrorCategory.SYSTEM
    message: str = "工具执行失败"
    retryable: bool = False
    retry_after_s: float | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    # 内部原因；只进日志，不返回 LLM
    cause: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _fill_defaults(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        value = data.get("code")
        try:
            code = value if isinstance(value, ToolErrorCode) else ToolErrorCode(value)
        except (TypeError, ValueError):
            return data
        data.setdefault("category", TOOL_ERROR_CATEGORIES[code])
        data.setdefault("retryable", code in TOOL_ERROR_RETRYABLE)
        return data

    @classmethod
    def from_exception(cls, exc: ToolException) -> "ToolError":
        return cls(
            code=exc.error_code,
            category=exc.category,
            message=exc.message,
            retryable=exc.retryable_value,
            retry_after_s=exc.retry_after_s,
            details=dict(exc.details),
            cause=exc.cause,
        )


class ToolMetrics(_Model):
    """单次工具调用的指标；只进日志/监控，不进 ToolMessage。"""

    duration_ms: int = 0
    queue_ms: int | None = None
    upstream_ms: int | None = None
    attempts: int = 1
    cache_hit: bool = False
    idempotent_replay: bool = False
    truncated: bool = False
    input_bytes: int | None = None
    output_bytes: int | None = None


class ToolResult(_Model):
    """一次工具调用的终态结果；工具模块与消息模块的桥。"""

    call_id: str
    tool_name: str
    status: ToolResultStatus
    # 给 LLM 的文本；成功是结果，失败是脱敏错误信息
    content: str = ""
    # 结构化数据；给 UI/审计/后处理
    data: Any = None
    files: list[ToolFile] = Field(default_factory=list)
    images: list[ToolImage] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)
    error: ToolError | None = None
    metrics: ToolMetrics = Field(default_factory=ToolMetrics)
    truncated: bool = False
    overflow_file: ToolFile | None = None
    cache_hit: bool = False
    idempotent_replay: bool = False
    started_at: datetime = Field(default_factory=utcnow)
    ended_at: datetime | None = None

    @model_validator(mode="after")
    def _error_consistent(self) -> "ToolResult":
        if self.status is not ToolResultStatus.SUCCEEDED and self.error is None:
            raise ValueError("FAILED/CANCELLED 的 ToolResult 必须带 ToolError")
        if self.status is ToolResultStatus.SUCCEEDED and self.error is not None:
            raise ValueError("SUCCEEDED 的 ToolResult 不能带 ToolError")
        return self

    @property
    def tool_call_status(self) -> ToolCallStatus:
        return {
            ToolResultStatus.SUCCEEDED: ToolCallStatus.SUCCEEDED,
            ToolResultStatus.FAILED: ToolCallStatus.FAILED,
            ToolResultStatus.CANCELLED: ToolCallStatus.CANCELLED,
        }[self.status]

    @property
    def message_status(self) -> MessageStatus:
        return (
            MessageStatus.COMPLETED
            if self.status is ToolResultStatus.SUCCEEDED
            else MessageStatus.FAILED
        )

    @classmethod
    def success(
        cls,
        *,
        call_id: str,
        tool_name: str,
        content: str = "",
        data: Any = None,
        **kwargs: Any,
    ) -> "ToolResult":
        return cls(
            call_id=call_id,
            tool_name=tool_name,
            status=ToolResultStatus.SUCCEEDED,
            content=content,
            data=data,
            **kwargs,
        )

    @classmethod
    def failure(
        cls,
        *,
        call_id: str,
        tool_name: str,
        error: ToolError,
        content: str | None = None,
        **kwargs: Any,
    ) -> "ToolResult":
        return cls(
            call_id=call_id,
            tool_name=tool_name,
            status=ToolResultStatus.FAILED,
            content=content if content is not None else error.message,
            error=error,
            **kwargs,
        )

    @classmethod
    def cancelled(
        cls,
        *,
        call_id: str,
        tool_name: str,
        error: ToolError | None = None,
        **kwargs: Any,
    ) -> "ToolResult":
        return cls(
            call_id=call_id,
            tool_name=tool_name,
            status=ToolResultStatus.CANCELLED,
            content="工具调用已取消",
            error=error
            or ToolError(
                code=ToolErrorCode.CANCELLED,
                message="工具调用已取消",
            ),
            **kwargs,
        )


class ToolInvocation(_MutableModel):
    """一次工具调用的运行期对象；不落库。"""

    call_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    raw_arguments: str | None = None
    context: ToolContext
    status: ToolCallStatus = ToolCallStatus.PENDING
    attempts: int = 0
    started_at: datetime = Field(default_factory=utcnow)
    ended_at: datetime | None = None
    result: ToolResult | None = None
    trace_id: str = ""
    cache_hit: bool = False
    idempotent_replay: bool = False

    def mark_running(self) -> None:
        self.status = ToolCallStatus.RUNNING
        self.started_at = utcnow()

    def finish(self, result: ToolResult) -> None:
        self.status = result.tool_call_status
        self.result = result
        self.ended_at = result.ended_at or utcnow()
        self.cache_hit = result.cache_hit
        self.idempotent_replay = result.idempotent_replay


class ToolBatch(_MutableModel):
    """同一 assistant 消息内的并行工具批次。"""

    invocations: list[ToolInvocation] = Field(default_factory=list)
    max_concurrency: int = Field(default=4, ge=1)
    deadline: datetime | None = None
    results: list[ToolResult] = Field(default_factory=list)
    status: ToolBatchStatus = ToolBatchStatus.RUNNING

    def finalize(self) -> "ToolBatch":
        """批次结束后按结果更新状态；全部调用终态后调用。"""
        self.results = [
            item.result for item in self.invocations if item.result is not None
        ]
        self.status = classify_batch_status(self.results)
        return self


def classify_batch_status(results: list[ToolResult]) -> ToolBatchStatus:
    """根据单工具结果推导批次状态。"""
    if not results:
        return ToolBatchStatus.COMPLETED
    statuses = [item.status for item in results]
    if any(item is ToolResultStatus.CANCELLED for item in statuses):
        return ToolBatchStatus.CANCELLED
    if all(item is ToolResultStatus.SUCCEEDED for item in statuses):
        return ToolBatchStatus.COMPLETED
    if all(item is ToolResultStatus.FAILED for item in statuses):
        return ToolBatchStatus.FAILED
    return ToolBatchStatus.PARTIAL


# ——————————————————— 状态层 ———————————————————

class ToolCacheEntry(_Model):
    key: str
    tool_name: str
    result: ToolResult
    created_at: datetime = Field(default_factory=utcnow)
    expires_at: datetime


class IdempotencyRecord(_Model):
    key: str
    tool_name: str
    user_id: UUID
    call_id: str
    state: IdempotencyState
    result: ToolResult | None = None
    error: ToolError | None = None
    created_at: datetime = Field(default_factory=utcnow)
    expires_at: datetime
    locked_until: datetime | None = None

    def is_locked(self, now: datetime | None = None) -> bool:
        return (
            self.state is IdempotencyState.IN_PROGRESS
            and self.locked_until is not None
            and self.locked_until > (now or utcnow())
        )


class CircuitBreakerRecord(_Model):
    tool_name: str
    state: ToolCircuitState = ToolCircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    # 当前窗口总调用数；用于 failure_rate_threshold，成功/失败都累加
    total_count: int = 0
    window_start: datetime = Field(default_factory=utcnow)
    opened_at: datetime | None = None
    next_probe_at: datetime | None = None
    half_open_in_flight: int = 0


class RateLimitRecord(_Model):
    key: str
    count: int = 0
    window_start: datetime = Field(default_factory=utcnow)
    expires_at: datetime


class ToolExecutionRecord(_Model):
    """日志/指标记录；不落 PG。"""

    trace_id: str
    user_id: UUID
    conversation_id: UUID
    turn_id: str | None = None
    call_id: str
    tool_name: str
    status: ToolResultStatus
    error_code: ToolErrorCode | None = None
    metrics: ToolMetrics = Field(default_factory=ToolMetrics)
    started_at: datetime
    ended_at: datetime
    cache_hit: bool = False
    idempotent_replay: bool = False


class ToolApprovalRequest(_Model):
    """危险工具确认请求；v1 只预留模型。"""

    call_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    side_effect: ToolSideEffect
    reason: str
    requested_at: datetime = Field(default_factory=utcnow)
    expires_at: datetime | None = None
    status: ToolApprovalStatus = ToolApprovalStatus.PENDING


class ToolApprovalDecision(_Model):
    """危险工具确认结果；v1 只预留模型。"""

    call_id: str
    approved: bool
    decided_by: UUID
    decided_at: datetime = Field(default_factory=utcnow)
    comment: str | None = None


class ToolExecutor(Protocol):
    """执行器协议：只写业务逻辑，不处理缓存/幂等/熔断/重试。"""

    async def __call__(
        self,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult: ...
