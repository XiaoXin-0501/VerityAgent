"""DomainProfile 加载器。

解析顺序（先命中先用）：

1. ``path``：调用方显式指定的 JSON 文件；
2. ``directory``：配置的领域目录，按 ``{name}.json`` 查找；
3. 内置目录 ``rag/indexer/enrichment/domains``。

外部接入新领域时，把 JSON 放进 ``domain_profile_dir`` 并改
``IndexerConfig.domain_profile`` 即可，不需要改代码。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import ValidationError

from exception import RagError, RagErrorCode

from .models import DomainProfile

__all__ = [
    "BUILTIN_DOMAIN_DIR",
    "load_domain_profile",
    "resolve_domain_profile_path",
]

# 内置领域画像目录；随代码发布，不依赖工作目录。
BUILTIN_DOMAIN_DIR = Path(__file__).resolve().parent.parent / "domains"

# 领域名只允许字母/数字/下划线/短横线，避免路径穿越。
_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def resolve_domain_profile_path(
    name: str | None = None,
    *,
    path: str | Path | None = None,
    directory: str | Path | None = None,
) -> Path:
    """解析画像文件路径；不存在或名字非法时抛 ``RagError``。"""
    if path:
        candidate = Path(path).expanduser()
        if candidate.is_dir():
            candidate = candidate / _file_name(name or "medical")
        if not candidate.is_file():
            raise _error(f"领域画像文件不存在: {candidate}", name=name)
        return candidate

    domain_name = (name or "medical").strip()
    if domain_name.lower().endswith(".json"):
        domain_name = domain_name[:-5]
    if not _NAME_RE.match(domain_name):
        raise _error(f"领域名不合法: {domain_name!r}", name=domain_name)
    file_name = _file_name(domain_name)

    base_dir = Path(directory).expanduser() if directory else BUILTIN_DOMAIN_DIR
    candidate = base_dir / file_name
    if not candidate.is_file():
        raise _error(
            f"领域画像不存在: {candidate}",
            name=domain_name,
            details={"directory": str(base_dir)},
        )
    return candidate


def load_domain_profile(
    name: str | None = None,
    *,
    path: str | Path | None = None,
    directory: str | Path | None = None,
) -> DomainProfile:
    """加载并校验领域画像；切换领域只改 ``name/path/directory``。"""
    file = resolve_domain_profile_path(name, path=path, directory=directory)
    try:
        payload = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise _error(
            f"领域画像读取失败: {file}: {exc}",
            name=name,
            details={"path": str(file)},
        ) from exc
    if not isinstance(payload, dict):
        raise _error(
            f"领域画像必须是 JSON 对象: {file}",
            name=name,
            details={"path": str(file)},
        )
    try:
        return DomainProfile.model_validate(payload)
    except ValidationError as exc:
        raise _error(
            f"领域画像不合法: {file}: {exc}",
            name=name,
            details={"path": str(file)},
        ) from exc


def _file_name(name: str) -> str:
    """领域名 → 文件名；允许调用方直接写 ``medical.json``。"""
    return name if name.lower().endswith(".json") else f"{name}.json"


def _error(
    message: str,
    *,
    name: str | None = None,
    details: dict[str, str] | None = None,
) -> RagError:
    """统一收敛为 RagError(INVALID_CONFIG)，便于启动期直接暴露配置问题。"""
    merged: dict[str, str] = {"profile": name or "medical"}
    merged.update(details or {})
    return RagError(
        RagErrorCode.INVALID_CONFIG,
        message,
        details=merged,
    )
