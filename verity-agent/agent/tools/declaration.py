"""工具声明 DTO。

JSON manifest 与代码工厂之间只通过这里的稳定结构交互：

- ``ToolDeclaration``：manifest.json 里的静态声明与单工具横切覆盖；
- ``ToolManifestFile``：manifest 顶层结构；
- ``ToolBuildContext``：启动期工厂装配所需的外部依赖。

工厂实现只依赖本模块，不反向依赖 manifest/registry，避免循环。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from config.tools import ToolsConfig

from .enums import ToolSideEffect
from .models import ToolExposureRule
from .services import get_tool_service

__all__ = [
    "POLICY_OVERRIDE_FIELDS",
    "ToolBuildContext",
    "ToolDeclaration",
    "ToolManifestFile",
]

# 单工具可覆盖的横切能力字段；新增能力时只在这里和 ToolDeclaration 增加。
POLICY_OVERRIDE_FIELDS = (
    "timeout_s",
    "retry",
    "cache",
    "circuit_breaker",
    "rate_limit",
    "idempotency",
    "result_limit",
    "max_concurrency",
)


class _Model(BaseModel):
    """声明模型基类：拒绝未知字段，加载期即失败。"""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolDeclaration(_Model):
    """manifest.json 中的单个工具声明。"""

    name: str = Field(min_length=1, max_length=64)
    description: str = Field(min_length=1)
    group: str | None = None
    # 静态权限码；None 表示登录用户可用
    required_permission: str | None = None
    # 暴露条件：kind + 参数；具体解释器见 agent.tools.exposure
    exposure: ToolExposureRule = Field(
        default_factory=lambda: ToolExposureRule(kind="always")
    )
    side_effect: ToolSideEffect = ToolSideEffect.READ
    tags: list[str] = Field(default_factory=list)
    # 装配期必须存在的服务 key；缺失时 bootstrap 明确警告并跳过，不静默
    required_services: list[str] = Field(default_factory=list)
    # 工厂路径：module:FactoryClass；新增工具只改 JSON + 自身实现模块
    implementation: str = Field(min_length=3, max_length=256)
    # 横切能力全部铺开；只写需要修改的字段，装配时与顶层默认值深合并。
    timeout_s: float | None = Field(default=None, gt=0)
    retry: dict[str, Any] | None = None
    cache: dict[str, Any] | None = None
    circuit_breaker: dict[str, Any] | None = None
    rate_limit: dict[str, Any] | None = None
    idempotency: dict[str, Any] | None = None
    result_limit: dict[str, Any] | None = None
    max_concurrency: int | None = Field(default=None, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolManifestFile(_Model):
    """manifest.json 顶层结构。"""

    version: Literal[1] = 1
    tools: list[ToolDeclaration] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ToolBuildContext:
    """启动期装配上下文。

    ``services`` 是通用依赖包（``app.state`` 或 Mapping），工厂按需通过
    ``get_service`` 取自己的依赖；新增工具不需要改 bootstrap 签名。
    """

    tools_config: ToolsConfig
    services: Any = None

    def get_service(self, name: str, default: Any = None) -> Any:
        """从 Mapping 或 app.state 风格对象读取服务；缺失返回默认值。"""
        return get_tool_service(self.services, name, default)
