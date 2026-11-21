"""JSON 工具 manifest 加载与工厂解析。

manifest.json 负责声明工具；具体工厂通过 ``implementation`` 的
``module:FactoryClass`` 路径动态解析。新增工具不需要修改本文件或
bootstrap，只需要新增实现模块并在 JSON 增加声明。
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

from .declaration import ToolBuildContext, ToolDeclaration, ToolManifestFile
from .exposure import validate_exposure_rule
from .factory import ToolFactory
from .models import RegisteredTool
from .services import ToolServiceKey

__all__ = ["TOOL_MANIFEST", "build_declared_tool"]

_MANIFEST_PATH = Path(__file__).with_name("manifest.json")
_FACTORY_BY_NAME: dict[str, ToolFactory] = {}


def _resolve_factory(implementation: str) -> ToolFactory:
    """解析 ``module:FactoryClass``；实例化并校验 build 协议。"""
    module_name, separator, factory_name = implementation.partition(":")
    if not separator or not module_name or not factory_name:
        raise ValueError(
            f"工具 implementation 必须是 module:FactoryClass: {implementation!r}"
        )
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - 包装成启动期配置错误
        raise ValueError(
            f"工具实现模块导入失败: {module_name}: {exc}"
        ) from exc
    factory_obj = getattr(module, factory_name, None)
    if factory_obj is None:
        raise ValueError(f"工具实现类不存在: {implementation}")
    try:
        factory = factory_obj() if isinstance(factory_obj, type) else factory_obj
    except Exception as exc:  # noqa: BLE001 - 启动期工厂构造失败
        raise ValueError(
            f"工具工厂实例化失败: {implementation}: {exc}"
        ) from exc
    if not callable(getattr(factory, "build", None)):
        raise ValueError(f"工具实现不是合法工厂: {implementation}")
    return factory


def _load_manifest() -> tuple[ToolDeclaration, ...]:
    """读取并校验 JSON manifest；失败直接阻断启动。"""
    payload = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest = ToolManifestFile.model_validate(payload)
    names = [item.name for item in manifest.tools]
    if len(names) != len(set(names)):
        raise ValueError(f"manifest 工具名重复: {names}")
    for item in manifest.tools:
        # 暴露条件是运行期解释的；未知 kind/参数必须在启动期失败，
        # 不能等到请求期才按“不暴露”静默降级。
        validate_exposure_rule(item.exposure)
        # required_services 与运行期读取共用 ToolServiceKey，启动期拦截
        # 拼写漂移，避免工具再次因服务名不一致被跳过。
        for service_name in item.required_services:
            try:
                ToolServiceKey(service_name)
            except ValueError as exc:
                raise ValueError(
                    f"工具 {item.name} 声明了未知服务: {service_name}"
                ) from exc
        _FACTORY_BY_NAME[item.name] = _resolve_factory(item.implementation)
    return tuple(manifest.tools)


TOOL_MANIFEST: tuple[ToolDeclaration, ...] = _load_manifest()


def build_declared_tool(
    declaration: ToolDeclaration,
    ctx: ToolBuildContext,
) -> RegisteredTool | None:
    """按 manifest 声明调用工厂装配工具。"""
    factory = _FACTORY_BY_NAME.get(declaration.name)
    if factory is None:
        raise ValueError(f"工具没有已解析工厂: {declaration.name}")
    tool = factory.build(declaration, ctx)
    if tool is not None and not isinstance(tool, RegisteredTool):
        raise ValueError(
            f"工具工厂返回类型非法: {declaration.name} -> {type(tool).__name__}"
        )
    return tool
