"""配置包公共基类与路径约定。

所有配置类集中在 ``config`` 包内：

- 基础设施：``config.infrastructure``
- Provider：``config.provider``
- 模型清单：``config.model``
- LLM：``config.llm``
- 消息：``config.message``
- Agent：``config.agent``
- RAG：``config.rag``
- Hooks：``config.hooks``

启动层只负责 env → 模块配置的映射，不重复模块默认值（《README》§4.1）。
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["PROJECT_ROOT", "SOURCE_ROOT", "ENV_FILE", "ConfigModel", "EnvSettings"]

# config/base.py -> config -> verity-agent -> 仓库根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"


class ConfigModel(BaseModel):
    """模块配置基类：拒绝未知字段，避免拼写错误静默生效。"""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class EnvSettings(BaseSettings):
    """env 读取基类：``GROUP__FIELD`` 映射到嵌套字段。"""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        env_ignore_empty=True,
        extra="ignore",
        case_sensitive=False,
    )
