"""密钥解析工具。

`.env` 由 pydantic-settings 读入配置对象，但不会导出到 ``os.environ``；
因此 provider/model 的 ``api_key_env`` 需要显式回读 `.env`。
"""

from __future__ import annotations

import os
from typing import Mapping

from pydantic import SecretStr

from .base import ENV_FILE

__all__ = ["resolve_api_key"]


def resolve_api_key(
    api_key: SecretStr | str | None,
    api_key_env: str | None,
    *,
    environ: Mapping[str, str] | None = None,
) -> str | None:
    """按 直接 key → 显式 environ → 进程环境 → .env 的顺序解析密钥。"""
    if isinstance(api_key, SecretStr):
        direct = api_key.get_secret_value()
    else:
        direct = api_key or ""
    if direct:
        return direct
    if not api_key_env:
        return None
    if environ is not None and environ.get(api_key_env):
        return environ[api_key_env]
    if os.environ.get(api_key_env):
        return os.environ[api_key_env]
    try:
        from dotenv import dotenv_values

        value = dotenv_values(ENV_FILE).get(api_key_env)
        return str(value) if value else None
    except Exception:
        return None
