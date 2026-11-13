"""Prompt 注册表：内置模板 + 外部覆盖目录。

目录结构（内置与覆盖目录一致）：::

    templates/
    ├── manifest.json          # id → 文件/版本/变量等元数据
    └── rag/
        ├── description.system.md
        ├── description.user.md
        ├── graph.system.md
        └── graph.user.md

覆盖策略：

- 覆盖目录按 ``PromptConfig.directory`` 指定，manifest 中的同名 id 覆盖内置；
- 覆盖项只需写要改的字段（常用：``file`` 换正文），未写字段继承内置；
- 新增 prompt 必须提供 ``file``，变量声明必须与模板占位符一致；
- 所有模板在构建期一次性加载并校验，运行期只读、无 I/O。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from config.base import SOURCE_ROOT
from config.prompt import PromptConfig
from exception import PromptError, PromptErrorCode

from .ids import PromptId
from .models import PromptTemplate, PromptVariable, RenderedPrompt

__all__ = [
    "BUILTIN_TEMPLATE_DIR",
    "PromptRegistry",
    "build_prompt_registry",
    "get_default_prompt_registry",
]

logger = logging.getLogger(__name__)

# 内置模板目录；随代码发布，不依赖工作目录。
BUILTIN_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
MANIFEST_FILE = "manifest.json"


class PromptEntry(BaseModel):
    """manifest 中单个 prompt 的元数据。

    覆盖目录允许只写部分字段，因此这里全部可选；合并后再校验必填项。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    file: str | None = None
    version: int | None = Field(default=None, ge=1)
    description: str | None = None
    purpose: str | None = None
    static: bool | None = None
    cacheable: bool | None = None
    variables: list[PromptVariable] | None = None


class PromptManifest(BaseModel):
    """prompt 目录的 manifest。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int = 1
    prompts: dict[str, PromptEntry] = Field(default_factory=dict)


class PromptRegistry:
    """只读 Prompt 注册表；构造完成后可安全共享。"""

    def __init__(
        self,
        templates: Mapping[str, PromptTemplate],
        *,
        strict: bool = True,
        sources: Iterable[str] = (),
    ) -> None:
        """保存模板与渲染策略；``sources`` 只用于日志/排障。"""
        self._templates = dict(templates)
        self.strict = strict
        self.sources = tuple(sources)

    @property
    def prompt_ids(self) -> tuple[str, ...]:
        """全部可用 prompt id，按字母序稳定输出。"""
        return tuple(sorted(self._templates))

    def has(self, prompt_id: str) -> bool:
        """判断 prompt 是否存在，避免调用方靠异常探测。"""
        return prompt_id in self._templates

    def get(self, prompt_id: str) -> PromptTemplate:
        """按 id 取模板；不存在时抛出 ``PROMPT_NOT_FOUND``。"""
        template = self._templates.get(prompt_id)
        if template is None:
            raise PromptError(
                PromptErrorCode.NOT_FOUND,
                f"Prompt 不存在: {prompt_id}",
                details={
                    "prompt_id": prompt_id,
                    "available": list(self.prompt_ids),
                },
            )
        return template

    def render(
        self,
        prompt_id: str,
        /,
        **values: Any,
    ) -> RenderedPrompt:
        """按 id 渲染，返回带版本/checksum 的快照。"""
        return self.get(prompt_id).render(values, strict=self.strict)

    def render_text(self, prompt_id: str, /, **values: Any) -> str:
        """只取渲染文本的便捷入口。"""
        return self.render(prompt_id, **values).text


def build_prompt_registry(
    config: PromptConfig | None = None,
    *,
    builtin_dir: str | Path | None = None,
) -> PromptRegistry:
    """按配置加载内置模板与可选覆盖目录。

    启动期完成全部读取与校验：目录缺失、manifest 非法、占位符与变量不一致
    都会直接抛 ``PromptError``，不允许带病启动。
    """
    cfg = config or PromptConfig()
    builtin_root = _path(builtin_dir) if builtin_dir else BUILTIN_TEMPLATE_DIR
    overlay_root = _resolve_overlay(cfg.directory)

    if not cfg.include_builtin and overlay_root is None:
        raise PromptError(
            PromptErrorCode.INVALID_CONFIG,
            "PromptConfig.include_builtin=false 时必须提供 directory",
        )

    entries: dict[str, tuple[PromptEntry, Path]] = {}
    sources: list[str] = []
    if cfg.include_builtin:
        entries = _load_entries(builtin_root)
        sources.append(str(builtin_root))
    if overlay_root is not None:
        overlay = _load_entries(overlay_root)
        entries = _merge_entries(entries, overlay, overlay_root)
        sources.append(str(overlay_root))

    templates = {
        prompt_id: _build_template(prompt_id, entry, root)
        for prompt_id, (entry, root) in sorted(entries.items())
    }
    # PromptId 声明了系统当前需要的全部 prompt；缺失时启动期直接报错，
    # 避免运行到某个功能才发现模板没覆盖。
    missing = sorted(
        str(prompt_id)
        for prompt_id in PromptId
        if str(prompt_id) not in templates
    )
    if missing:
        raise PromptError(
            PromptErrorCode.LOAD_FAILED,
            f"Prompt 注册表缺少必需模板: {missing}",
            details={"missing": missing, "sources": sources},
        )
    logger.info(
        "Prompt 注册表加载完成: %d 个模板, sources=%s",
        len(templates),
        sources,
    )
    return PromptRegistry(templates, strict=cfg.strict, sources=sources)


@lru_cache(maxsize=1)
def get_default_prompt_registry() -> PromptRegistry:
    """内置模板的进程级单例。

    RAG 等模块在未显式注入 registry 时使用它；自定义目录请在应用启动时
    ``build_prompt_registry(settings.prompts)`` 后显式传入。
    模板更新需要重启进程（注册表运行期只读）。
    """
    return build_prompt_registry()


def _load_entries(root: Path) -> dict[str, tuple[PromptEntry, Path]]:
    """读取一个目录的 manifest；返回 id → (元数据, 文件根目录)。"""
    manifest_path = root / MANIFEST_FILE
    if not manifest_path.is_file():
        raise PromptError(
            PromptErrorCode.LOAD_FAILED,
            f"Prompt manifest 不存在: {manifest_path}",
            details={"path": str(manifest_path)},
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PromptError(
            PromptErrorCode.LOAD_FAILED,
            f"Prompt manifest 读取失败: {manifest_path}: {exc}",
            details={"path": str(manifest_path)},
        ) from exc
    try:
        manifest = PromptManifest.model_validate(payload)
    except ValidationError as exc:
        raise PromptError(
            PromptErrorCode.LOAD_FAILED,
            f"Prompt manifest 不合法: {manifest_path}: {exc}",
            details={"path": str(manifest_path)},
        ) from exc
    if not manifest.prompts:
        raise PromptError(
            PromptErrorCode.LOAD_FAILED,
            f"Prompt manifest 没有任何 prompt: {manifest_path}",
            details={"path": str(manifest_path)},
        )
    return {
        prompt_id: (entry, root)
        for prompt_id, entry in manifest.prompts.items()
    }


def _merge_entries(
    builtin: dict[str, tuple[PromptEntry, Path]],
    overlay: dict[str, tuple[PromptEntry, Path]],
    overlay_root: Path,
) -> dict[str, tuple[PromptEntry, Path]]:
    """把覆盖目录合并到内置结果；同名 id 按字段覆盖。"""
    merged = dict(builtin)
    for prompt_id, (entry, _root) in overlay.items():
        base = merged.get(prompt_id)
        if base is None:
            if entry.file is None:
                raise PromptError(
                    PromptErrorCode.LOAD_FAILED,
                    f"新增 Prompt {prompt_id} 必须提供 file",
                    details={"prompt_id": prompt_id},
                )
            merged[prompt_id] = (entry, overlay_root)
            continue

        base_entry, base_root = base
        # 只覆盖显式提供的字段；未提供的继承内置。
        update = entry.model_dump(exclude_none=True)
        try:
            merged_entry = PromptEntry.model_validate(
                {**base_entry.model_dump(), **update}
            )
        except ValidationError as exc:
            raise PromptError(
                PromptErrorCode.LOAD_FAILED,
                f"覆盖 Prompt {prompt_id} 的元数据不合法: {exc}",
                details={"prompt_id": prompt_id},
            ) from exc
        if merged_entry.file is None:
            raise PromptError(
                PromptErrorCode.LOAD_FAILED,
                f"Prompt {prompt_id} 缺少 file",
                details={"prompt_id": prompt_id},
            )
        # 覆盖项显式给了 file，则正文从覆盖目录读；否则仍用内置文件。
        entry_root = overlay_root if entry.file is not None else base_root
        merged[prompt_id] = (merged_entry, entry_root)
    return merged


def _build_template(
    prompt_id: str,
    entry: PromptEntry,
    root: Path,
) -> PromptTemplate:
    """读取正文并构造/校验模板。"""
    if entry.file is None:
        raise PromptError(
            PromptErrorCode.LOAD_FAILED,
            f"Prompt {prompt_id} 缺少 file",
            details={"prompt_id": prompt_id},
        )
    path = _safe_file(root, entry.file)
    try:
        body = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PromptError(
            PromptErrorCode.LOAD_FAILED,
            f"Prompt 文件读取失败: {path}: {exc}",
            details={"prompt_id": prompt_id, "path": str(path)},
        ) from exc
    try:
        return PromptTemplate(
            id=prompt_id,
            version=entry.version or 1,
            description=entry.description or "",
            purpose=entry.purpose or "",
            body=body,
            variables=list(entry.variables or []),
            static=bool(entry.static),
            cacheable=True if entry.cacheable is None else entry.cacheable,
            source=str(path),
        )
    except (ValidationError, ValueError) as exc:
        raise PromptError(
            PromptErrorCode.LOAD_FAILED,
            f"Prompt {prompt_id} 不合法: {exc}",
            details={"prompt_id": prompt_id, "path": str(path)},
        ) from exc


def _safe_file(root: Path, relative: str) -> Path:
    """解析模板文件并阻止 ``../`` 越出模板根目录。"""
    base = root.resolve()
    path = (base / relative).resolve()
    if not path.is_relative_to(base):
        raise PromptError(
            PromptErrorCode.LOAD_FAILED,
            f"Prompt 文件越出模板目录: {relative}",
            details={"root": str(base), "file": relative},
        )
    if not path.is_file():
        raise PromptError(
            PromptErrorCode.LOAD_FAILED,
            f"Prompt 文件不存在: {path}",
            details={"path": str(path)},
        )
    return path


def _resolve_overlay(directory: str | None) -> Path | None:
    """解析覆盖目录；相对路径按 ``verity-agent`` 源码根目录解析。"""
    if not directory:
        return None
    path = Path(directory).expanduser()
    if not path.is_absolute():
        path = SOURCE_ROOT / path
    if not path.is_dir():
        raise PromptError(
            PromptErrorCode.INVALID_CONFIG,
            f"Prompt 覆盖目录不存在: {path}",
            details={"path": str(path)},
        )
    return path


def _path(value: str | Path) -> Path:
    """把内置目录参数统一成 Path。"""
    return value if isinstance(value, Path) else Path(value)
