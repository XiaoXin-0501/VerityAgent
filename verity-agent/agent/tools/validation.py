"""工具参数校验与规范化。

第一方工具以 Pydantic ``argument_model`` 为唯一运行时校验源；JSON Schema
由模型生成（``model_tool_schema``），不再维护自研的简化 JSON Schema 校验器。

只有显式标记 ``metadata={"schema_only": True}`` 的动态/第三方工具才允许
没有 Pydantic 模型，此时用 ``jsonschema`` 校验其手写 schema。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import jsonschema
from jsonschema.exceptions import SchemaError
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from pydantic import BaseModel, ValidationError as PydanticValidationError

from exception.tool import ToolArgumentValidationError, ToolRegistrationError
from utils.serialization import to_jsonable

from .models import RegisteredTool

__all__ = [
    "model_tool_schema",
    "validate_tool_schema",
    "strict_for_model",
    "validate_arguments",
    "stable_arguments",
]


def model_tool_schema(model: type[BaseModel]) -> dict[str, Any]:
    """由 Pydantic 模型生成 LLM-facing JSON Schema。"""
    schema = model.model_json_schema()
    schema.setdefault("type", "object")
    return schema


def validate_tool_schema(schema: Mapping[str, Any]) -> None:
    """注册期检查 schema-only 工具的 JSON Schema 是否合法。"""
    try:
        jsonschema.Draft202012Validator.check_schema(dict(schema))
    except SchemaError as exc:
        raise ToolRegistrationError(
            "工具 JSON Schema 不合法",
            details={"path": list(exc.path), "message": exc.message},
        ) from exc


def strict_for_model(model: type[BaseModel]) -> bool:
    """OpenAI strict 要求所有 properties 必填；有默认值/可选字段时降级。"""
    return all(field.is_required() for field in model.model_fields.values())


def validate_arguments(
    tool: RegisteredTool,
    arguments: Mapping[str, Any],
) -> dict[str, Any]:
    """校验并返回规范化的参数字典；失败抛 ToolArgumentValidationError。"""
    if not isinstance(arguments, Mapping):
        raise ToolArgumentValidationError(
            "工具参数必须是 JSON 对象",
            tool_name=tool.spec.name,
            details={"type": type(arguments).__name__},
        )

    if tool.argument_model is not None:
        if tool.spec.strict:
            fields = tool.argument_model.model_fields
            aliases = set(fields) | {
                item.alias for item in fields.values() if item.alias
            }
            extra = sorted(set(arguments) - aliases)
            if extra:
                raise ToolArgumentValidationError(
                    "存在未声明的参数",
                    tool_name=tool.spec.name,
                    details={"fields": extra},
                )
        try:
            model = tool.argument_model.model_validate(dict(arguments))
        except PydanticValidationError as exc:
            raise ToolArgumentValidationError(
                "工具参数校验失败",
                tool_name=tool.spec.name,
                details={"errors": to_jsonable(exc.errors(include_url=False))},
            ) from exc
        return model.model_dump(mode="python", by_alias=False)

    if tool.spec.metadata.get("schema_only"):
        try:
            jsonschema.Draft202012Validator(
                dict(tool.spec.parameters),
                format_checker=jsonschema.FormatChecker(),
            ).validate(dict(arguments))
        except JsonSchemaValidationError as exc:
            raise ToolArgumentValidationError(
                "工具参数不符合 JSON Schema",
                tool_name=tool.spec.name,
                details={
                    "path": list(exc.absolute_path),
                    "validator": exc.validator,
                    "message": exc.message,
                },
            ) from exc
        return dict(arguments)

    properties = tool.spec.parameters.get("properties")
    if not properties:
        # 无参数工具：不接受任何字段
        if arguments:
            raise ToolArgumentValidationError(
                f"工具 {tool.spec.name} 不接受参数",
                tool_name=tool.spec.name,
                details={"fields": sorted(arguments)},
            )
        return {}

    raise ToolArgumentValidationError(
        f"工具 {tool.spec.name} 未配置 argument_model",
        tool_name=tool.spec.name,
        details={"schema_only": False},
    )


def stable_arguments(
    arguments: Mapping[str, Any],
    *,
    key_fields: Sequence[str] | None = None,
    exclude_fields: Sequence[str] = (),
) -> dict[str, Any]:
    """提取参与缓存/幂等 key 的稳定参数。

    ``key_fields`` 为 None 表示全部参数；``exclude_fields`` 永远优先排除。
    """
    if key_fields is None:
        result = dict(arguments)
    else:
        result = {key: arguments[key] for key in key_fields if key in arguments}
    for key in exclude_fields:
        result.pop(key, None)
    return result
