"""工具工厂协议与通用装配逻辑。

新增工具时只需要：

1. 在实现模块提供一个 ``ToolFactory``（通常继承 ``SimpleToolFactory``）；
2. 在 ``manifest.json`` 的 ``implementation`` 填 ``module:FactoryClass``。

工厂负责依赖可用性、执行器和少量 spec 覆盖；描述、权限、暴露条件、横切策略
默认值由 manifest/config 统一提供，不需要改 manifest.py 或 bootstrap.py。
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel

from .declaration import (
    POLICY_OVERRIDE_FIELDS,
    ToolBuildContext,
    ToolDeclaration,
)
from .enums import ToolKind
from .models import RegisteredTool, ToolExposureRule, ToolSpec
from .policies import ToolPolicy

__all__ = [
    "SimpleToolFactory",
    "ToolFactory",
    "build_policy",
    "with_exposure_param",
]

class ToolFactory(Protocol):
    """工具工厂协议；manifest 通过 ``module:FactoryClass`` 解析。"""

    def build(
        self,
        declaration: ToolDeclaration,
        ctx: ToolBuildContext,
    ) -> RegisteredTool | None:
        """装配工具；依赖不满足时返回 None。"""
        ...


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并策略覆盖；未出现的字段继承顶层默认值。"""
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def build_policy(
    declaration: ToolDeclaration,
    ctx: ToolBuildContext,
) -> ToolPolicy:
    """顶层默认策略 + 单工具差异字段 -> ToolPolicy。"""
    base = ctx.tools_config.policy.model_dump(mode="python")
    overrides = {
        key: getattr(declaration, key)
        for key in POLICY_OVERRIDE_FIELDS
        if key in declaration.model_fields_set
    }
    return ToolPolicy.model_validate(_deep_merge(base, overrides))


def with_exposure_param(
    rule: ToolExposureRule,
    key: str,
    value: Any,
) -> ToolExposureRule:
    """覆盖条件参数，保持 kind 与其它参数不变。"""
    params = dict(rule.params)
    params[key] = value
    return ToolExposureRule(kind=rule.kind, **params)


class SimpleToolFactory:
    """通用工厂：处理 spec/策略/元数据，子类只写依赖与执行器。"""

    # 参数模型由实现模块提供；registry 用它生成 LLM schema
    argument_model: type[BaseModel] | None = None
    # 默认普通执行工具；router 工厂可覆盖并补 router_group
    kind: ToolKind = ToolKind.EXECUTABLE
    source: str = "builtin"

    def available(
        self,
        declaration: ToolDeclaration,
        ctx: ToolBuildContext,
    ) -> bool:
        """依赖是否满足；默认总是可用。"""
        return True

    def create_executor(
        self,
        declaration: ToolDeclaration,
        ctx: ToolBuildContext,
    ) -> Any:
        """创建执行器；子类必须实现。"""
        raise NotImplementedError

    def spec_overrides(
        self,
        declaration: ToolDeclaration,
        ctx: ToolBuildContext,
    ) -> dict[str, Any]:
        """少量 spec 动态覆盖（如 recall_round 的压缩阈值）。"""
        return {}

    def build(
        self,
        declaration: ToolDeclaration,
        ctx: ToolBuildContext,
    ) -> RegisteredTool | None:
        """通用装配：可用性 -> 策略 -> spec -> executor。"""
        if not self.available(declaration, ctx):
            return None
        policy = build_policy(declaration, ctx)
        spec_kwargs: dict[str, Any] = {
            "name": declaration.name,
            "description": declaration.description,
            "group": declaration.group,
            "kind": self.kind,
            "side_effect": declaration.side_effect,
            "exposure": declaration.exposure,
            "required_permission": declaration.required_permission,
            "policy": policy,
            "tags": list(declaration.tags),
            "metadata": dict(declaration.metadata),
        }
        spec_kwargs.update(
            self.spec_overrides(declaration, ctx)
        )
        return RegisteredTool(
            spec=ToolSpec(**spec_kwargs),
            executor=self.create_executor(declaration, ctx),
            argument_model=self.argument_model,
            source=self.source,
        )
