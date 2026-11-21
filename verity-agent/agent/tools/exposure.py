"""工具暴露条件解释器。

工具 manifest 只声明 ``exposure: {"kind": ..., ...params}``；本模块把
``kind`` 映射到可注册的 evaluator。内置条件：

- ``always``：无条件暴露；
- ``rag_required``：本次查询规划需要 RAG 召回；
- ``after_compressions``：``compression_count >= min_compressions``；
- ``all_of`` / ``any_of``：组合子条件（``conditions: [...]``）。

未来新增动态暴露条件时，只需：

1. 在请求上下文 ``ToolExposureContext.signals`` 中放入所需信号；
2. 调用 ``register_exposure_condition(kind, evaluator)``；
3. manifest 使用新的 ``kind``。

不需要修改 ToolSpec、Selection 或 AgentLoop。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from .models import ToolExposureContext, ToolExposureRule

__all__ = [
    "ExposureEvaluator",
    "register_exposure_condition",
    "rule_allowed",
    "validate_exposure_rule",
]

logger = logging.getLogger(__name__)

ExposureEvaluator = Callable[[ToolExposureRule, ToolExposureContext], bool]

_EVALUATORS: dict[str, ExposureEvaluator] = {}


def register_exposure_condition(kind: str, evaluator: ExposureEvaluator) -> None:
    """注册一个暴露条件 kind；重复注册直接报错。"""
    key = (kind or "").strip()
    if not key:
        raise ValueError("暴露条件 kind 不能为空")
    if key in _EVALUATORS:
        raise ValueError(f"暴露条件重复注册: {key}")
    _EVALUATORS[key] = evaluator


def _nested_conditions(rule: ToolExposureRule) -> list[ToolExposureRule]:
    """解析 all_of/any_of 的 conditions 参数。"""
    raw = rule.params.get("conditions")
    if not isinstance(raw, list):
        return []
    result: list[ToolExposureRule] = []
    for item in raw:
        if isinstance(item, ToolExposureRule):
            result.append(item)
        elif isinstance(item, dict):
            result.append(ToolExposureRule.model_validate(item))
        else:
            raise ValueError(f"暴露条件 {rule.kind} 的 conditions 元素非法")
    return result


def _always(_rule: ToolExposureRule, _ctx: ToolExposureContext) -> bool:
    return True


def _rag_required(_rule: ToolExposureRule, ctx: ToolExposureContext) -> bool:
    return bool(ctx.rag_required)


def _after_compressions(
    rule: ToolExposureRule,
    ctx: ToolExposureContext,
) -> bool:
    try:
        threshold = int(rule.params.get("min_compressions") or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError("after_compressions.min_compressions 必须是整数") from exc
    return ctx.compression_count >= max(0, threshold)


def _all_of(rule: ToolExposureRule, ctx: ToolExposureContext) -> bool:
    return all(rule_allowed(item, ctx) for item in _nested_conditions(rule))


def _any_of(rule: ToolExposureRule, ctx: ToolExposureContext) -> bool:
    return any(rule_allowed(item, ctx) for item in _nested_conditions(rule))


def _register_builtins() -> None:
    """注册内置条件；只在模块导入时执行一次。"""
    register_exposure_condition("always", _always)
    register_exposure_condition("rag_required", _rag_required)
    register_exposure_condition("after_compressions", _after_compressions)
    register_exposure_condition("all_of", _all_of)
    register_exposure_condition("any_of", _any_of)


def validate_exposure_rule(rule: ToolExposureRule) -> None:
    """启动期校验条件 kind 与嵌套结构；未知 kind 直接失败。"""
    if rule.kind not in _EVALUATORS:
        raise ValueError(f"未知工具暴露条件: {rule.kind}")
    if rule.kind == "after_compressions":
        try:
            threshold = int(rule.params.get("min_compressions") or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "after_compressions.min_compressions 必须是整数"
            ) from exc
        if threshold < 0:
            raise ValueError(
                "after_compressions.min_compressions 不能为负数"
            )
    if rule.kind in {"all_of", "any_of"}:
        conditions = _nested_conditions(rule)
        if not conditions:
            raise ValueError(f"暴露条件 {rule.kind} 至少需要一个子条件")
        for item in conditions:
            validate_exposure_rule(item)


def rule_allowed(rule: ToolExposureRule, ctx: ToolExposureContext) -> bool:
    """执行暴露条件；未知 kind 在启动期已校验，这里 fail-closed。"""
    evaluator = _EVALUATORS.get(rule.kind)
    if evaluator is None:
        logger.warning("未知工具暴露条件，按不暴露处理: %s", rule.kind)
        return False
    return bool(evaluator(rule, ctx))


_register_builtins()
