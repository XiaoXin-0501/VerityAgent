"""进程内工具注册表。

启动时装配，运行期只读；路由只改 ``ToolSelection``，不改注册表。
工具定义/执行器不落 PG，也不返回给 LLM。
"""

from __future__ import annotations

from dataclasses import replace
from threading import RLock
from typing import Any

from pydantic import BaseModel

from exception.tool import ToolNotFoundError, ToolRegistrationError
from utils.hashing import sha256_json

from .enums import ToolKind
from .models import (
    RegisteredTool,
    ToolCatalog,
    ToolExposureContext,
    ToolGroup,
    ToolSpec,
)
from .policies import ToolRoutingPolicy
from .selection import build_selection
from .validation import (
    model_tool_schema,
    strict_for_model,
    validate_tool_schema,
)

__all__ = ["ToolRegistry"]


class ToolRegistry:
    """工具注册表；线程安全，建议启动后只读。"""

    def __init__(self) -> None:
        """初始化工具、工具组与注册锁。"""
        self._tools: dict[str, RegisteredTool] = {}
        self._groups: dict[str, ToolGroup] = {}
        self._lock = RLock()

    # ——————————————————— 注册 ———————————————————

    def register(self, registered: RegisteredTool) -> RegisteredTool:
        """注册一个已经装配好的工具。"""
        with self._lock:
            spec = registered.spec
            if spec.name in self._tools:
                raise ToolRegistrationError(
                    f"工具重复注册: {spec.name}",
                    tool_name=spec.name,
                )
            self._validate_spec(spec)
            prepared_spec = self._prepare_spec(spec, registered.argument_model)
            if prepared_spec is not spec:
                registered = replace(registered, spec=prepared_spec)
            self._tools[spec.name] = registered
            return registered

    def register_tool(
        self,
        spec: ToolSpec,
        executor: Any,
        *,
        argument_model: type[BaseModel] | None = None,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RegisteredTool:
        """便捷注册：由 spec + executor 组 RegisteredTool。"""
        return self.register(
            RegisteredTool(
                spec=spec,
                executor=executor,
                argument_model=argument_model,
                source=source,
                metadata=dict(metadata or {}),
            )
        )

    def register_group(self, group: ToolGroup) -> ToolGroup:
        """注册路由组；组内工具/路由工具可以稍后注册。"""
        with self._lock:
            if group.name in self._groups:
                raise ToolRegistrationError(
                    f"工具组重复注册: {group.name}",
                    details={"group": group.name},
                )
            router = self._tools.get(group.router_tool_name)
            if router is not None and (
                router.spec.kind is not ToolKind.ROUTER
                or router.spec.router_group != group.name
            ):
                raise ToolRegistrationError(
                    f"路由工具与组不匹配: {group.router_tool_name}",
                    details={"group": group.name},
                )
            self._groups[group.name] = group
            return group

    def unregister(self, name: str) -> bool:
        """移除工具；已经生成的 catalog 快照不受影响。"""
        with self._lock:
            return self._tools.pop(name, None) is not None

    # ——————————————————— 查询 ———————————————————

    def get(self, name: str) -> RegisteredTool | None:
        """按名称取工具；不存在返回 None。"""
        return self._tools.get(name)

    def require(self, name: str) -> RegisteredTool:
        """按名称取工具；不存在抛 ToolNotFoundError。"""
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFoundError(name)
        return tool

    def all(self) -> list[RegisteredTool]:
        """返回全部已注册工具。"""
        return list(self._tools.values())

    def groups(self) -> list[ToolGroup]:
        """返回全部工具组。"""
        return list(self._groups.values())

    def snapshot(self) -> ToolCatalog:
        """生成不可变快照；一次运行只使用一个版本。"""
        with self._lock:
            tools = dict(self._tools)
            groups = dict(self._groups)
            version = self._catalog_version(tools, groups)
        return ToolCatalog(
            tools=tools,
            groups=groups,
            version=version,
        )

    def for_permissions(
        self,
        permissions: frozenset[str],
        routing: ToolRoutingPolicy | None = None,
        exposure: ToolExposureContext | None = None,
    ):
        """按权限与请求暴露策略生成 ToolSelection；内部使用当前快照。"""
        return build_selection(
            self.snapshot(),
            permissions,
            routing or ToolRoutingPolicy(),
            exposure,
        )

    def clear(self) -> None:
        """清空工具与工具组；仅用于测试/重载。"""
        with self._lock:
            self._tools.clear()
            self._groups.clear()

    def __len__(self) -> int:
        """已注册工具数量。"""
        return len(self._tools)

    # ——————————————————— 校验 ———————————————————

    @staticmethod
    def _prepare_spec(
        spec: ToolSpec,
        model: type[BaseModel] | None,
    ) -> ToolSpec:
        """以 Pydantic 模型为参数事实源；schema-only 工具保留手写 schema。"""
        if spec.kind is ToolKind.ROUTER:
            return spec
        if model is not None:
            schema = model_tool_schema(model)
            strict = spec.strict and strict_for_model(model)
            if strict:
                schema.setdefault("additionalProperties", False)
            return spec.model_copy(
                update={"parameters": schema, "strict": strict}
            )
        if spec.metadata.get("schema_only"):
            validate_tool_schema(spec.parameters)
            return spec
        if spec.parameters.get("properties"):
            raise ToolRegistrationError(
                f"工具 {spec.name} 必须提供 argument_model，"
                "或显式设置 metadata.schema_only=true",
                tool_name=spec.name,
            )
        return spec

    def _validate_spec(self, spec: ToolSpec) -> None:
        """校验 router 工具与目标组声明一致。"""
        if spec.kind is ToolKind.ROUTER:
            group = self._groups.get(spec.router_group or "")
            if group is None:
                raise ToolRegistrationError(
                    f"router_group 不存在: {spec.router_group}",
                    tool_name=spec.name,
                    details={"router_group": spec.router_group},
                )
            if group.router_tool_name != spec.name:
                raise ToolRegistrationError(
                    f"路由工具名与组声明不一致: {spec.name}",
                    tool_name=spec.name,
                    details={"expected": group.router_tool_name},
                )

    @staticmethod
    def _catalog_version(
        tools: dict[str, RegisteredTool],
        groups: dict[str, ToolGroup],
    ) -> str:
        """根据工具与组定义计算稳定的 catalog 版本号。"""
        payload = {
            "tools": [
                item.spec.model_dump(mode="json")
                for item in sorted(tools.values(), key=lambda tool: tool.spec.name)
            ],
            "groups": [
                group.model_dump(mode="json")
                for group in sorted(groups.values(), key=lambda item: item.name)
            ],
        }
        return f"toolcat_{sha256_json(payload)[:16]}"
