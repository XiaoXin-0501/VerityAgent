"""Prompt 管理配置。

Prompt 文本本身放在 ``verity-agent/prompt/templates``（内置）或外部覆盖目录；
这里只放启动期行为参数，符合 D-92“配置类集中在 config 包”。
"""

from __future__ import annotations

from .base import ConfigModel

__all__ = ["PromptConfig"]


class PromptConfig(ConfigModel):
    """Prompt 注册表加载与渲染策略。"""

    # 外部覆盖目录（必须包含 manifest.json）；相对路径按 verity-agent 解析。
    # 为空时只使用内置模板。
    directory: str | None = None
    # 是否同时加载内置模板；False 时目录必须提供全部所需 prompt。
    include_builtin: bool = True
    # 渲染策略：缺少必填变量时是否 fail-fast；生产建议 true。
    # 模板未声明的额外入参始终忽略（允许覆盖模板只使用变量子集）。
    strict: bool = True
