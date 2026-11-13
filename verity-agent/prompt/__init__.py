"""统一 Prompt 管理模块。

- 模板正文与元数据集中在 ``prompt/templates``，不写死在业务代码里；
- 支持静态 prompt（无变量）与动态 prompt（变量在运行时注入）；
- 支持外部覆盖目录（``PromptConfig.directory``），覆盖项可只改正文；
- 渲染结果带 id/version/checksum，可写入 Prompt 审计日志。

设计说明见 ``doc/Prompt模块设计.md``。
"""

from .ids import PromptId
from .models import PromptTemplate, PromptVariable, RenderedPrompt
from .registry import (
    BUILTIN_TEMPLATE_DIR,
    PromptRegistry,
    build_prompt_registry,
    get_default_prompt_registry,
)

__all__ = [
    "BUILTIN_TEMPLATE_DIR",
    "PromptId",
    "PromptRegistry",
    "PromptTemplate",
    "PromptVariable",
    "RenderedPrompt",
    "build_prompt_registry",
    "get_default_prompt_registry",
]
