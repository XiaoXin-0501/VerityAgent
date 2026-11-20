"""工具权限校验。

同一套权限码同时覆盖静态能力与资源：

- ``required_permission``：如 ``tool:kb:search``，选择时过滤；
- ``resource_permission_template``：如 ``kb:read:{kb_id}``，执行时按参数展开；
- 多资源工具（如 ``search_kb(kb_ids=[...])``）逐资源检查；v1 默认全有权限
  才执行，工具若需“过滤未授权资源”可传 ``allow_partial=True``。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from exception.tool import ToolArgumentValidationError, ToolPermissionDeniedError

from .enums import ToolExposureMode
from .models import (
    TEMPLATE_FIELD_RE,
    ToolCatalog,
    ToolExposureContext,
    ToolGroup,
    ToolSpec,
)

__all__ = [
    "has_required_permission",
    "spec_allowed",
    "group_allowed",
    "required_resource_permissions",
    "missing_resource_permissions",
    "check_resource_permissions",
]

def has_required_permission(
    spec: ToolSpec,
    permissions: frozenset[str],
) -> bool:
    """静态能力码校验；无 required_permission 表示登录用户可用。"""
    return (
        spec.required_permission is None
        or spec.required_permission in permissions
    )


def exposure_allowed(
    spec: ToolSpec,
    exposure: ToolExposureContext | None,
) -> bool:
    """按单次请求上下文判断工具暴露策略；权限不参与这里。"""
    if exposure is None:
        return True
    if spec.exposure == ToolExposureMode.RAG_REQUIRED:
        return exposure.rag_required
    if spec.exposure == ToolExposureMode.AFTER_COMPRESSIONS:
        return exposure.compression_count >= spec.min_compressions
    return True


def spec_allowed(
    spec: ToolSpec,
    permissions: frozenset[str],
    exposure: ToolExposureContext | None = None,
) -> bool:
    """工具是否可暴露：启用 + 静态权限通过 + 请求暴露策略通过。"""
    return (
        spec.enabled
        and has_required_permission(spec, permissions)
        and exposure_allowed(spec, exposure)
    )


def group_allowed(
    group: ToolGroup,
    catalog: ToolCatalog,
    permissions: frozenset[str],
    exposure: ToolExposureContext | None = None,
) -> bool:
    """组是否可暴露：组启用、组权限通过，且至少一个组内工具可用。"""
    if not group.enabled:
        return False
    if (
        group.required_permission is not None
        and group.required_permission not in permissions
    ):
        return False
    for name in group.tool_names:
        tool = catalog.tools.get(name)
        if tool is not None and spec_allowed(tool.spec, permissions, exposure):
            return True
    return False


def _field_values(arguments: Mapping[str, Any], field: str) -> list[Any]:
    """读取模板字段值；兼容 kb_id -> kb_ids 的复数参数。"""
    value = arguments.get(field)
    if value is None and field.endswith("_id"):
        # 模板写 {kb_id}、参数常传 kb_ids；做一次兼容映射
        value = arguments.get(f"{field}s")
    if value is None:
        raise ToolArgumentValidationError(
            f"资源权限模板缺少参数: {field}",
            details={"field": field},
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return list(value)
    return [value]


def required_resource_permissions(
    spec: ToolSpec,
    arguments: Mapping[str, Any],
) -> list[str]:
    """把资源权限模板展开成具体权限码；去重并保持顺序。"""
    template = spec.resource_permission_template
    if not template:
        return []
    fields = TEMPLATE_FIELD_RE.findall(template)
    if not fields:
        return [template]

    codes: list[str] = []
    for field in fields:
        for value in _field_values(arguments, field):
            code = template.format(**{field: value})
            if code not in codes:
                codes.append(code)
    return codes


def missing_resource_permissions(
    spec: ToolSpec,
    arguments: Mapping[str, Any],
    permissions: frozenset[str],
) -> list[str]:
    """返回当前权限集合缺失的资源级权限码。"""
    return [
        code
        for code in required_resource_permissions(spec, arguments)
        if code not in permissions
    ]


def check_resource_permissions(
    spec: ToolSpec,
    arguments: Mapping[str, Any],
    permissions: frozenset[str],
    *,
    allow_partial: bool = False,
) -> list[str]:
    """执行前资源级权限校验。

    返回本次允许的资源权限码；``allow_partial=True`` 时只要有一个资源可用就
    放行，由执行器自行过滤未授权资源（如多 KB 检索）。
    """
    if not has_required_permission(spec, permissions):
        raise ToolPermissionDeniedError(
            f"没有调用工具 {spec.name} 的权限",
            tool_name=spec.name,
            details={"required_permission": spec.required_permission},
        )

    codes = required_resource_permissions(spec, arguments)
    missing = [code for code in codes if code not in permissions]
    if not missing:
        return codes
    allowed = [code for code in codes if code in permissions]
    if allow_partial and allowed:
        return allowed
    raise ToolPermissionDeniedError(
        f"没有访问工具 {spec.name} 所需资源的权限",
        tool_name=spec.name,
        details={"missing_permissions": missing},
    )
