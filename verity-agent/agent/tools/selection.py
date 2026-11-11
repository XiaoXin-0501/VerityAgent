"""工具选择与路由。

选择层只改 ``ToolSelection``，不修改 ``ToolRegistry``：

- 可用工具少于阈值：``FLAT``，全量暴露；
- 达到阈值且开启路由：``ROUTED``，只暴露各组的 router 工具；
- router 调用后 ``activate_group`` 展开组内工具，并移除该 router。
"""

from __future__ import annotations

from exception.tool import ToolNotFoundError

from .enums import ToolKind, ToolRoutingMode, ToolSelectionMode
from .models import (
    RegisteredTool,
    ToolCatalog,
    ToolDescriptor,
    ToolGroup,
    ToolGroupDescriptor,
    ToolSelection,
)
from .permissions import group_allowed, spec_allowed
from .policies import ToolRoutingPolicy

__all__ = [
    "build_selection",
    "activate_group",
    "is_router_tool",
]


def build_selection(
    catalog: ToolCatalog,
    permissions: frozenset[str],
    routing: ToolRoutingPolicy,
) -> ToolSelection:
    """按权限与路由策略生成一次运行的 ToolSelection。"""
    allowed_tools = [
        tool for tool in catalog.tools.values() if spec_allowed(tool.spec, permissions)
    ]
    allowed_groups = [
        group
        for group in catalog.groups.values()
        if group_allowed(group, catalog, permissions)
    ]

    routed = (
        routing.mode is not ToolRoutingMode.NONE
        and len(allowed_tools) >= routing.flat_threshold
    )
    if not routed:
        return _flat_selection(catalog, allowed_tools, allowed_groups)

    groups = allowed_groups[: routing.max_groups]
    descriptors: list[ToolDescriptor] = []
    for group in groups:
        router = catalog.tools.get(group.router_tool_name)
        if (
            router is not None
            and router.spec.kind is ToolKind.ROUTER
            and router.spec.router_group == group.name
            and spec_allowed(router.spec, permissions)
        ):
            descriptors.append(ToolDescriptor.from_spec(router.spec))

    if not descriptors:
        # 没有可用的 router 工具时退回 FLAT，不能把工具藏起来导致模型无工具可用
        return _flat_selection(catalog, allowed_tools, allowed_groups)

    descriptors.sort(key=lambda item: item.name)
    names = {item.name for item in descriptors}
    return ToolSelection(
        catalog_version=catalog.version,
        mode=ToolSelectionMode.ROUTED,
        descriptors=descriptors,
        groups=[ToolGroupDescriptor.from_group(group) for group in groups],
        active_groups=set(),
        exposed_tool_names=names,
        round=0,
    )


def activate_group(
    catalog: ToolCatalog,
    selection: ToolSelection,
    group_name: str,
    permissions: frozenset[str],
) -> ToolSelection:
    """router 调用后激活组内工具，返回新的 selection。

    不原地修改入参，Agent 可用返回值替换当前快照。
    """
    group = catalog.get_group(group_name)
    if group is None or not group_allowed(group, catalog, permissions):
        raise ToolNotFoundError(
            group_name,
            message=f"路由组不存在或不可用: {group_name}",
        )
    if group.router_tool_name not in selection.exposed_tool_names:
        raise ToolNotFoundError(
            group.router_tool_name,
            message=f"路由工具未暴露: {group.router_tool_name}",
        )

    updated = selection.model_copy(deep=True)
    updated.descriptors = [
        item
        for item in updated.descriptors
        if item.name != group.router_tool_name
    ]
    updated.exposed_tool_names.discard(group.router_tool_name)

    for name in group.tool_names:
        tool = catalog.tools.get(name)
        if tool is None or not spec_allowed(tool.spec, permissions):
            continue
        if name in updated.exposed_tool_names:
            continue
        updated.descriptors.append(ToolDescriptor.from_spec(tool.spec))
        updated.exposed_tool_names.add(name)

    updated.descriptors.sort(key=lambda item: item.name)
    updated.active_groups.add(group_name)
    updated.round += 1
    return updated


def is_router_tool(catalog: ToolCatalog, name: str) -> bool:
    tool = catalog.tools.get(name)
    return tool is not None and tool.spec.kind is ToolKind.ROUTER


def _flat_selection(
    catalog: ToolCatalog,
    allowed_tools: list[RegisteredTool],
    allowed_groups: list[ToolGroup],
) -> ToolSelection:
    descriptors = sorted(
        (ToolDescriptor.from_spec(tool.spec) for tool in allowed_tools),
        key=lambda item: item.name,
    )
    return ToolSelection(
        catalog_version=catalog.version,
        mode=ToolSelectionMode.FLAT,
        descriptors=descriptors,
        groups=[ToolGroupDescriptor.from_group(group) for group in allowed_groups],
        active_groups=set(),
        exposed_tool_names={item.name for item in descriptors},
        round=0,
    )
