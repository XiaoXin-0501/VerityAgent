"""垂直领域画像（DomainProfile）：只辅助 LLM 抽取，不直接产出图节点。

边界约定（重要）：

- **实体抽取必须由 LLM 完成**。领域画像中的 ``whitelist`` 只是补充词/别名
  参考，用于文本命中时提示、别名归一、查询期 seed 实体匹配；
- ``blacklist`` 是明确禁止进入图的关键词，提示词与结果后处理都会使用；
- ``entity_types`` 声明垂直领域的类型体系，类型示例用于告诉 LLM 粒度；
- 关系只能取 ``relations`` 中声明的类型。

这样切换领域只需要换一份 JSON；新增/更换文档时，词典未覆盖的实体仍由
LLM 从正文中抽取，不会被词典边界限制。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    model_validator,
)

__all__ = [
    "DomainEntityType",
    "DomainEntity",
    "DomainRelationType",
    "DomainPrompt",
    "DomainProfile",
    "DomainMatch",
]

# 别名中的水平空白统一压成一个空格，避免 “阿司匹林 片” 匹配不到 “阿司匹林片”。
_WHITESPACE = re.compile(r"\s+")


def _key(value: str) -> str:
    """匹配键：去首尾空白、压缩空白，英文统一小写。"""
    return _WHITESPACE.sub(" ", (value or "").strip()).lower()


def _alias_pattern(alias: str) -> str:
    """把单个别名转成正则片段；ASCII 边界避免 HTN 命中 THTN 这类子串。"""
    pattern = re.escape(alias)
    if alias and alias[0].isascii() and alias[0].isalnum():
        pattern = rf"(?<![A-Za-z0-9]){pattern}"
    if alias and alias[-1].isascii() and alias[-1].isalnum():
        pattern = rf"{pattern}(?![A-Za-z0-9])"
    return pattern


@dataclass(frozen=True, slots=True)
class DomainMatch:
    """一次词典命中；``name`` 是归一后的标准名，不是原文表面形式。"""

    name: str
    entity_type: str
    surface: str
    start: int
    end: int


class _Model(BaseModel):
    """领域画像模型基类：拒绝未知字段并保持只读。"""

    model_config = ConfigDict(extra="forbid", frozen=True)


class DomainEntityType(_Model):
    """实体类型定义；``examples`` 是垂直领域的粒度示例，不是抽取白名单。"""

    name: str = Field(min_length=1)
    label: str = ""
    description: str = ""
    examples: list[str] = Field(default_factory=list)


class DomainEntity(_Model):
    """一个领域补充词（whitelist）；只放新词/别名，用于归一与提示，不直接建节点。"""

    name: str = Field(min_length=1)
    type: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    description: str = ""


class DomainRelationType(_Model):
    """允许写入图的关系类型；LLM 只能从这里取值。"""

    name: str = Field(min_length=1)
    label: str = ""
    description: str = ""


class DomainPrompt(_Model):
    """领域提示词片段；具体 Prompt 骨架与输出格式由 PromptRegistry 模板追加。"""

    role: str = ""
    entity_guidance: str = ""
    relation_guidance: str = ""
    extra_instructions: list[str] = Field(default_factory=list)
    # 好例子：每行一个“实体名（类型）：说明”，帮助 LLM 理解粒度
    entity_examples: list[str] = Field(default_factory=list)
    # 噪音例子：每行一类不应进入图节点的文本
    noise_examples: list[str] = Field(default_factory=list)


class DomainProfile(_Model):
    """一个垂直领域的完整画像；加载后只读。

    别名索引、匹配正则、类型查找表与内容签名只构建一次。签名参与图抽取
    缓存 key，画像内容变化后无需手动改版本号也能让旧缓存失效。
    """

    name: str = Field(min_length=1)
    display_name: str = ""
    version: str = "1"
    language: str = "zh"
    description: str = ""

    entity_types: list[DomainEntityType] = Field(default_factory=list)
    # 补充词白名单：只放新词/别名；文本命中时提示 + 别名归一 + 查询 seed 匹配
    whitelist: list[DomainEntity] = Field(default_factory=list)
    # 明确禁止进入图的关键词（泛化词、版式词、叙述词等）
    blacklist: list[str] = Field(default_factory=list)
    relations: list[DomainRelationType] = Field(default_factory=list)
    prompt: DomainPrompt = Field(default_factory=DomainPrompt)

    # 运行期索引：别名键 → (标准名, 类型)
    _alias_index: dict[str, tuple[str, str]] = PrivateAttr(default_factory=dict)
    # 运行期索引：全部原始别名（用于构建最长优先匹配正则）
    _surfaces: list[str] = PrivateAttr(default_factory=list)
    _blacklist_keys: frozenset[str] = PrivateAttr(default_factory=frozenset)
    _entity_type_names: frozenset[str] = PrivateAttr(default_factory=frozenset)
    _type_lookup: dict[str, str] = PrivateAttr(default_factory=dict)
    _relation_names: frozenset[str] = PrivateAttr(default_factory=frozenset)
    _pattern: re.Pattern[str] | None = PrivateAttr(default=None)
    _signature: str = PrivateAttr(default="")

    @model_validator(mode="after")
    def _validate_profile(self) -> "DomainProfile":
        """加载期 fail-fast：重复类型、冲突别名、未声明类型直接报错。"""
        type_names: set[str] = set()
        type_lookup: dict[str, str] = {}
        for item in self.entity_types:
            if item.name in type_names:
                raise ValueError(f"实体类型重复: {item.name}")
            type_names.add(item.name)
            name_key = _key(item.name)
            name_owner = type_lookup.get(name_key)
            if name_owner is not None and name_owner != item.name:
                raise ValueError(
                    f"实体类型名称冲突: {item.name!r} 与类型标签 "
                    f"{name_owner} 冲突"
                )
            type_lookup[name_key] = item.name
            if item.label:
                label_key = _key(item.label)
                owner = type_lookup.get(label_key)
                if owner is not None and owner != item.name:
                    raise ValueError(
                        f"实体类型标签冲突: {item.label!r} 同时属于 "
                        f"{owner} 与 {item.name}"
                    )
                type_lookup[label_key] = item.name
        if not type_names:
            raise ValueError("领域画像至少需要声明一个 entity_type")

        relation_names: set[str] = set()
        for item in self.relations:
            if item.name in relation_names:
                raise ValueError(f"关系类型重复: {item.name}")
            relation_names.add(item.name)

        entity_names: set[str] = set()
        alias_owner: dict[str, str] = {}
        surfaces: list[str] = []
        for entity in self.whitelist:
            if entity.name in entity_names:
                raise ValueError(f"补充词重复: {entity.name}")
            entity_names.add(entity.name)
            if entity.type not in type_names:
                raise ValueError(
                    f"补充词 {entity.name} 的类型未声明: {entity.type}"
                )
            for surface in [entity.name, *entity.aliases]:
                key = _key(surface)
                if not key:
                    raise ValueError(f"补充词 {entity.name} 存在空别名")
                # 单字中文词在中文文本里几乎总是子串（“肝”命中“肝炎”），
                # 作为匹配面噪声大于收益；与图关键词过滤口径保持一致。
                if len(key) < 2:
                    raise ValueError(
                        f"补充词 {entity.name} 的别名过短: {surface!r}"
                    )
                owner = alias_owner.get(key)
                if owner is not None and owner != entity.name:
                    raise ValueError(
                        f"别名冲突: {surface!r} 同时属于 {owner} 与 "
                        f"{entity.name}"
                    )
                alias_owner[key] = entity.name
                surfaces.append(surface)

        blacklist_keys = frozenset(
            _key(item) for item in self.blacklist if _key(item)
        )
        conflict = blacklist_keys & set(alias_owner)
        if conflict:
            raise ValueError(
                f"补充词别名不能同时出现在 blacklist: {sorted(conflict)}"
            )

        # 别名索引在这里构建：model_post_init 早于 after-validator，放在
        # 那里会在非法 type 上先抛 KeyError，掩盖真正的校验错误。
        index: dict[str, tuple[str, str]] = {}
        for entity in self.whitelist:
            for surface in [entity.name, *entity.aliases]:
                index[_key(surface)] = (entity.name, entity.type)

        object.__setattr__(self, "_surfaces", surfaces)
        object.__setattr__(self, "_blacklist_keys", blacklist_keys)
        object.__setattr__(self, "_entity_type_names", frozenset(type_names))
        object.__setattr__(self, "_type_lookup", type_lookup)
        object.__setattr__(self, "_relation_names", frozenset(relation_names))
        object.__setattr__(self, "_alias_index", index)
        payload = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
        )
        object.__setattr__(
            self,
            "_signature",
            hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16],
        )
        return self

    # ——————————————————— 只读属性 ———————————————————

    @property
    def signature(self) -> str:
        """画像内容签名；任何字段变化都会改变，用于图抽取缓存失效。"""
        return self._signature

    @property
    def relation_names(self) -> frozenset[str]:
        """领域声明的关系类型；抽取结果只能从这里取值。"""
        return self._relation_names

    @property
    def entity_type_names(self) -> frozenset[str]:
        """领域声明的实体类型。"""
        return self._entity_type_names

    @property
    def blacklist_keys(self) -> frozenset[str]:
        """归一化后的领域关键词黑名单。"""
        return self._blacklist_keys

    def is_blacklisted(self, value: str) -> bool:
        """是否命中领域关键词黑名单。"""
        return _key(value) in self._blacklist_keys

    # ——————————————————— 匹配与归一 ———————————————————

    @property
    def pattern(self) -> re.Pattern[str]:
        """最长优先的实体匹配正则；画像加载后只编译一次。"""
        if self._pattern is None:
            if self._surfaces:
                # Python 正则的 alternation 从左到右尝试，按长度降序保证
                # “2型糖尿病”先于“糖尿病”命中，避免被短词截断。
                aliases = sorted(self._surfaces, key=len, reverse=True)
                compiled = re.compile(
                    "|".join(_alias_pattern(item) for item in aliases),
                    re.IGNORECASE,
                )
            else:
                # 没有补充词时使用永不匹配的正则，避免空 alternation
                # 编译成“匹配任意位置”。
                compiled = re.compile(r"(?!x)x")
            object.__setattr__(self, "_pattern", compiled)
        pattern = self._pattern
        assert pattern is not None
        return pattern

    def match(self, text: str) -> list[DomainMatch]:
        """按最长优先扫描文本，返回不重叠的领域补充词命中。"""
        if not text or not self._alias_index:
            return []
        matches: list[DomainMatch] = []
        for found in self.pattern.finditer(text):
            info = self._alias_index.get(_key(found.group(0)))
            if info is None:  # 理论上不会发生；防御正则/索引不同步
                continue
            name, entity_type = info
            matches.append(
                DomainMatch(
                    name=name,
                    entity_type=entity_type,
                    surface=found.group(0),
                    start=found.start(),
                    end=found.end(),
                )
            )
        return matches

    def canonicalize(self, value: str) -> str | None:
        """把标准名/别名归一为标准名；词典外返回 None。"""
        info = self._alias_index.get(_key(value))
        return info[0] if info else None

    def canonicalize_entity(self, value: str) -> tuple[str, str] | None:
        """归一为 (标准名, 类型)；词典外返回 None。"""
        return self._alias_index.get(_key(value))

    def resolve_entity_type(self, value: str) -> str | None:
        """把 LLM 返回的类型名/标签解析为画像声明的类型名。"""
        key = _key(value)
        return self._type_lookup.get(key) if key else None

    def entity_type_of(self, value: str) -> str | None:
        """补充词的类型；词典外返回 None。"""
        info = self._alias_index.get(_key(value))
        return info[1] if info else None

    def relation_prompt_lines(self) -> list[str]:
        """LLM 提示词中的关系类型清单。"""
        return [
            f"- {item.name}: {item.label or item.description}"
            for item in self.relations
        ]

    def entity_type_prompt_lines(self) -> list[str]:
        """LLM 提示词中的实体类型清单；包含垂直领域粒度示例。"""
        lines: list[str] = []
        for item in self.entity_types:
            label = item.label or item.name
            line = f"- {item.name}（{label}）"
            if item.description and item.description != label:
                line += f"：{item.description}"
            if item.examples:
                line += f"；示例：{'、'.join(item.examples)}"
            lines.append(line)
        return lines

    def reference_terms(
        self,
        text: str,
        *,
        limit: int = 20,
    ) -> list[dict[str, str]]:
        """返回文本中实际出现的补充词（标准名/类型）。

        白名单不是抽取边界，也不要求穷举领域：这里只挑出当前 chunk 文本
        里出现的参考词，用于提示 LLM 统一新词/别名的标准名。文本中没有
        出现的词不会进入提示词。
        """
        if limit <= 0:
            return []
        terms: list[dict[str, str]] = []
        seen: set[str] = set()
        for match in self.match(text):
            key = match.name.casefold()
            if key in seen:
                continue
            seen.add(key)
            terms.append({"name": match.name, "type": match.entity_type})
            if len(terms) >= limit:
                break
        return terms
