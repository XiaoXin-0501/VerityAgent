"""内置工具清单：JSON 声明 + 代码实现映射。

- ``manifest.json`` 集中声明名称、描述、分组、静态权限、暴露策略、策略预设
  与实现 id；
- 本模块把实现 id 映射到 ``builtin`` 的参数模型/执行器，构造 ``ToolSpec``；
- 注册表启动后常驻所有已装配工具；每次请求由 Agent 扫描注册表，结合用户
  权限 + ``ToolExposureContext`` 生成实际暴露的 ToolSelection；
- ``required_permission`` 为空表示登录用户即可使用；权限只用于注册/暴露前
  的过滤，不是运行期动态注册条件。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field

from config.tools import RecallRoundToolConfig, SearchKbToolConfig

from .builtin import (
    RecallRoundArgs,
    RecallRoundExecutor,
    ResolveConflictArgs,
    SearchKbArgs,
    build_search_kb_executor,
    execute_conflict_tool,
)
from .enums import ToolExposureMode, ToolKind, ToolSideEffect
from .models import RegisteredTool, ToolSpec
from .policies import (
    CachePolicy,
    CircuitBreakerPolicy,
    RateLimitPolicy,
    ToolPolicy,
)

__all__ = [
    "TOOL_MANIFEST",
    "ToolBuildContext",
    "ToolDeclaration",
    "build_declared_tool",
]

_MANIFEST_PATH = Path(__file__).with_name("manifest.json")


class _Model(BaseModel):
    """manifest 模型基类：拒绝未知字段，加载期即失败。"""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolDeclaration(_Model):
    """manifest.json 中的单个工具声明。"""

    name: str = Field(min_length=1, max_length=64)
    description: str = Field(min_length=1)
    group: str | None = None
    # 静态权限码；None 表示登录用户可用
    required_permission: str | None = None
    exposure: ToolExposureMode = ToolExposureMode.ALWAYS
    # AFTER_COMPRESSIONS 的默认阈值；具体实现可用运行配置覆盖
    min_compressions: int = Field(default=0, ge=0)
    side_effect: ToolSideEffect = ToolSideEffect.READ
    tags: list[str] = Field(default_factory=list)
    # 实现 id；必须在本模块 _IMPLEMENTATIONS 中存在
    implementation: str = Field(min_length=1, max_length=64)
    # 横切策略预设；实现只负责参数模型与执行器
    policy_preset: Literal["default", "rag_search", "cacheless"] = "default"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolManifestFile(_Model):
    """manifest.json 顶层结构。"""

    version: Literal[1] = 1
    tools: list[ToolDeclaration] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ToolBuildContext:
    """启动期装配依赖；只包含具体实现需要的外部对象/配置。"""

    retriever: Any | None
    chunk_conflict_service: Any | None
    search_kb: SearchKbToolConfig
    recall_round: RecallRoundToolConfig


def _load_manifest() -> tuple[ToolDeclaration, ...]:
    """读取并校验 JSON manifest；失败直接阻断启动。"""
    payload = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest = ToolManifestFile.model_validate(payload)
    names = [item.name for item in manifest.tools]
    if len(names) != len(set(names)):
        raise ValueError(f"manifest 工具名重复: {names}")
    for item in manifest.tools:
        if item.implementation not in _IMPLEMENTATIONS:
            raise ValueError(
                f"manifest 工具 {item.name} 引用了未知实现: {item.implementation}"
            )
    return tuple(manifest.tools)


def _spec_for(
    declaration: ToolDeclaration,
    *,
    policy: ToolPolicy,
    min_compressions: int | None = None,
) -> ToolSpec:
    """把 JSON 声明投影为运行期 ToolSpec。"""
    return ToolSpec(
        name=declaration.name,
        description=declaration.description,
        group=declaration.group,
        kind=ToolKind.EXECUTABLE,
        side_effect=declaration.side_effect,
        exposure=declaration.exposure,
        min_compressions=(
            declaration.min_compressions
            if min_compressions is None
            else min_compressions
        ),
        required_permission=declaration.required_permission,
        policy=policy,
        tags=list(declaration.tags),
        metadata=dict(declaration.metadata),
    )


def _policy_for(declaration: ToolDeclaration, ctx: ToolBuildContext) -> ToolPolicy:
    """按 JSON 的 policy_preset 构造横切策略。"""
    preset = declaration.policy_preset
    if preset == "rag_search":
        # 检索结果由 RAG RetrievalCache 负责；工具层只保留熔断与限流。
        return ToolPolicy(
            cache=CachePolicy(enabled=False),
            circuit_breaker=CircuitBreakerPolicy(enabled=True),
            rate_limit=RateLimitPolicy(
                enabled=True,
                max_calls=ctx.search_kb.rate_limit_calls,
                window_s=ctx.search_kb.rate_limit_window_s,
            ),
        )
    if preset == "cacheless":
        # 结果只对当前会话有效；缓存 key 缺 conversation_id，直接关闭。
        return ToolPolicy(cache=CachePolicy(enabled=False))
    if preset == "default":
        return ToolPolicy()
    raise ValueError(
        f"工具 {declaration.name} 的 policy_preset 未知: {preset}"
    )


def _build_search_kb(
    declaration: ToolDeclaration,
    ctx: ToolBuildContext,
) -> RegisteredTool | None:
    """装配 ``search_kb``：RAG 工具模式入口。"""
    if ctx.retriever is None:
        return None
    return RegisteredTool(
        spec=_spec_for(declaration, policy=_policy_for(declaration, ctx)),
        executor=build_search_kb_executor(ctx.search_kb),
        argument_model=SearchKbArgs,
        source="builtin",
    )


def _build_recall_round(
    declaration: ToolDeclaration,
    ctx: ToolBuildContext,
) -> RegisteredTool | None:
    """装配 ``recall_round``：摘要压缩达到阈值后才暴露。"""
    if not ctx.recall_round.enabled:
        return None
    return RegisteredTool(
        spec=_spec_for(
            declaration,
            policy=_policy_for(declaration, ctx),
            # JSON 中的 min_compressions 是部署默认值；运行配置可覆盖。
            min_compressions=ctx.recall_round.min_compressions,
        ),
        executor=RecallRoundExecutor(ctx.recall_round),
        argument_model=RecallRoundArgs,
        source="builtin",
    )


def _build_chunk_conflict(
    declaration: ToolDeclaration,
    ctx: ToolBuildContext,
) -> RegisteredTool | None:
    """装配 ``resolve_chunk_conflict``：仅 RAG 召回请求中暴露。"""
    if ctx.chunk_conflict_service is None:
        return None
    return RegisteredTool(
        spec=_spec_for(declaration, policy=_policy_for(declaration, ctx)),
        executor=execute_conflict_tool,
        argument_model=ResolveConflictArgs,
        source="builtin",
    )


_IMPLEMENTATIONS: dict[
    str, Callable[[ToolDeclaration, ToolBuildContext], RegisteredTool | None]
] = {
    "search_kb": _build_search_kb,
    "recall_round": _build_recall_round,
    "chunk_conflict": _build_chunk_conflict,
}


# 启动期统一工具表；新增工具：改 manifest.json + 在这里注册实现 id。
TOOL_MANIFEST: tuple[ToolDeclaration, ...] = _load_manifest()


def build_declared_tool(
    declaration: ToolDeclaration,
    ctx: ToolBuildContext,
) -> RegisteredTool | None:
    """按实现 id 装配单个工具；实现缺失在加载期已校验。"""
    builder = _IMPLEMENTATIONS.get(declaration.implementation)
    if builder is None:
        raise ValueError(
            f"工具 {declaration.name} 引用了未知实现: {declaration.implementation}"
        )
    return builder(declaration, ctx)
