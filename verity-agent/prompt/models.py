"""Prompt 模板与渲染结果模型。

设计约定：

- **模板正文**放在 ``templates/*.md``，不写进代码；manifest 只描述元数据；
- **变量**使用标准库 ``string.Template`` 的 ``${name}`` 语法；
- **静态 prompt** 不允许声明变量，``static=true``；
- **动态 prompt** 由 manifest 声明变量，运行时注入；未传的可选变量用 manifest
  默认值补位；未声明的入参忽略，便于覆盖模板只使用变量子集；
- **checksum** 是正文 SHA256，进入 ``RenderedPrompt``，供审计/缓存比对。

加载期会校验“模板占位符”与“manifest 变量声明”完全一致，避免模板改了但
manifest 忘记更新导致运行时才报错。
"""

from __future__ import annotations

import re
import string
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from exception import PromptError, PromptErrorCode
from utils.hashing import sha256_text

__all__ = ["PromptVariable", "PromptTemplate", "RenderedPrompt"]

# Prompt id 采用点分层级：<模块>.<用途>.<类型>，如 rag.graph.system。
_PROMPT_ID_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)+$")


class _Model(BaseModel):
    """本模块模型基类：拒绝未知字段，加载后只读。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
    )


class PromptVariable(_Model):
    """一个模板变量声明。"""

    name: str = Field(min_length=1)
    description: str = ""
    # 缺失且无 default 时是否报错；有 default 时通常设为 false
    required: bool = True
    # 默认可用于“领域画像未提供时”的兜底文案（如通用角色/规则）
    default: str | None = None


class RenderedPrompt(_Model):
    """一次渲染结果；用于发送给模型，也可写入 Prompt 审计日志。"""

    prompt_id: str
    version: int
    text: str
    # 模板正文的 SHA256，变量变化不改变 checksum
    checksum: str
    # 实际参与渲染的变量（含默认值）；便于定位“这段文案来自哪里”
    variables: dict[str, str] = Field(default_factory=dict)
    # 模板文件路径；不暴露给前端，只用于服务端日志/排障
    source: str = ""


class PromptTemplate(_Model):
    """一个不可变 Prompt 模板。"""

    id: str
    version: int = Field(default=1, ge=1)
    description: str = ""
    # 用途标签，例如 rag.document_description / rag.graph_extraction
    purpose: str = ""
    # 模板正文（从文件读取）
    body: str
    variables: list[PromptVariable] = Field(default_factory=list)
    # 静态模板不允许变量；动态模板由调用方传值
    static: bool = False
    # 是否适合 prompt caching；只是 hint，不做实际缓存
    cacheable: bool = True
    # 模板来源文件
    source: str = ""
    # 正文 SHA256；留空时自动计算
    checksum: str = ""

    @model_validator(mode="after")
    def _validate_template(self) -> "PromptTemplate":
        """校验 id/正文/变量声明与占位符一致，并补齐 checksum。"""
        if not _PROMPT_ID_RE.match(self.id):
            raise ValueError(f"Prompt id 不合法: {self.id!r}")
        if not self.body.strip():
            raise ValueError(f"Prompt {self.id} 正文为空")

        names = [item.name for item in self.variables]
        if len(names) != len(set(names)):
            raise ValueError(f"Prompt {self.id} 存在重复变量声明")
        declared = set(names)
        if self.static and declared:
            raise ValueError(f"静态 Prompt {self.id} 不应声明变量")

        try:
            placeholders = set(string.Template(self.body).get_identifiers())
        except ValueError as exc:
            raise ValueError(
                f"Prompt {self.id} 模板语法不合法: {exc}"
            ) from exc
        missing = sorted(placeholders - declared)
        extra = sorted(declared - placeholders)
        if missing or extra:
            raise ValueError(
                f"Prompt {self.id} 的模板占位符与 manifest 变量不一致: "
                f"未声明={missing}, 未使用={extra}"
            )

        if not self.checksum:
            object.__setattr__(self, "checksum", sha256_text(self.body))
        return self

    def render(
        self,
        values: Mapping[str, Any] | None = None,
        *,
        strict: bool = True,
    ) -> RenderedPrompt:
        """渲染模板；``strict`` 控制缺少必填变量是否报错。

        模板未声明的入参会被忽略：覆盖目录可以用同一调用方、同一组变量
        提供“只使用其中一部分”的模板变体；真正需要防的是缺变量导致 Prompt
        静默少一段内容。
        """
        provided = {
            key: _stringify(value)
            for key, value in (values or {}).items()
        }
        declared = {item.name: item for item in self.variables}

        merged: dict[str, str] = {}
        missing: list[str] = []
        for name, spec in declared.items():
            if name in provided:
                merged[name] = provided[name]
            elif spec.default is not None:
                merged[name] = spec.default
            elif spec.required:
                missing.append(name)
                # 非 strict 模式也要补空串，否则 substitute 会抛 KeyError。
                merged[name] = ""
            else:
                merged[name] = ""
        if missing and strict:
            raise PromptError(
                PromptErrorCode.RENDER_FAILED,
                f"Prompt {self.id} 缺少必填变量: {missing}",
                details={"prompt_id": self.id, "missing": missing},
            )

        try:
            # substitute（非 safe_substitute）保证非 strict 模式下也不会残留
            # 无法识别的 `$` 片段；缺失项已用空串补齐。
            text = string.Template(self.body).substitute(merged)
        except (KeyError, ValueError) as exc:
            raise PromptError(
                PromptErrorCode.RENDER_FAILED,
                f"Prompt {self.id} 渲染失败: {exc}",
                details={"prompt_id": self.id},
            ) from exc

        return RenderedPrompt(
            prompt_id=self.id,
            version=self.version,
            text=text,
            checksum=self.checksum,
            variables=merged,
            source=self.source,
        )


def _stringify(value: Any) -> str:
    """变量统一转字符串；None 视为空串，避免渲染出 "None"。"""
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)
