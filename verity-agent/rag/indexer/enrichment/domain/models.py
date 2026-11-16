"""垂直领域画像（DomainProfile）：图抽取的领域知识入口。

设计目标：**一个部署只接入一个垂直领域**。领域差异全部收敛到 JSON 画像：

- 实体词典：标准名 + 别名 + 类型 + 权重；
- 领域停用词：该领域无检索价值的通用词；
- 关系类型：允许写入图的关系白名单；
- LLM 提示词：领域角色与抽取要求。

切换领域只需要换一份 JSON（或改 ``domain_profile`` 配置），
抽取代码、流水线、存储都不需要改。默认宠物医疗画像见
``rag/indexer/enrichment/domains/pet_medical.json``。
"""

from __future__ import annotations

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
    "DomainRelationRule",
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
    weight: float


class _Model(BaseModel):
    """领域画像模型基类：拒绝未知字段并保持只读。"""

    model_config = ConfigDict(extra="forbid", frozen=True)


class DomainEntityType(_Model):
    """实体类型定义；权重参与关键词排序（疾病/药物通常高于解剖部位）。"""

    name: str = Field(min_length=1)
    label: str = ""
    weight: float = Field(default=1.0, gt=0.0)
    description: str = ""


class DomainEntity(_Model):
    """一个领域实体；``aliases`` 包含缩写、俗称、英文名。"""

    name: str = Field(min_length=1)
    type: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    # 单实体权重；为空时继承实体类型权重
    weight: float | None = Field(default=None, gt=0.0)
    description: str = ""


class DomainRelationType(_Model):
    """允许写入图的关系类型；LLM 只能从这里取值。"""

    name: str = Field(min_length=1)
    label: str = ""
    description: str = ""


class DomainRelationRule(_Model):
    """词典模式的关系抽取规则。

    同一句中出现 ``subject_types`` 与 ``object_types`` 的实体，且命中
    ``keywords``（为空表示只看类型组合）时，生成一条 ``name`` 关系。
    规则数据放在领域画像 JSON，不在代码里写死医学词表。
    """

    # 关系名必须在 relations 中声明
    name: str = Field(min_length=1)
    # 触发词；为空表示“类型组合即关系”
    keywords: list[str] = Field(default_factory=list)
    # 主语/宾语实体类型
    subject_types: list[str] = Field(default_factory=list)
    object_types: list[str] = Field(default_factory=list)
    # 规则置信度，需 >= graph_min_confidence 才会入图
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class DomainPrompt(_Model):
    """领域提示词片段；具体 Prompt 骨架与输出格式由 PromptRegistry 模板追加。"""

    role: str = ""
    entity_guidance: str = ""
    relation_guidance: str = ""
    extra_instructions: list[str] = Field(default_factory=list)


class DomainProfile(_Model):
    """一个垂直领域的完整画像。

    画像加载后视为只读；别名索引与匹配正则只构建一次。
    """

    name: str = Field(min_length=1)
    display_name: str = ""
    version: str = "1"
    language: str = "zh"
    description: str = ""

    entity_types: list[DomainEntityType] = Field(default_factory=list)
    entities: list[DomainEntity] = Field(default_factory=list)
    stopwords: list[str] = Field(default_factory=list)
    relations: list[DomainRelationType] = Field(default_factory=list)
    # 词典模式关系规则；未配置时 lexicon 模式只抽关键词
    relation_rules: list[DomainRelationRule] = Field(default_factory=list)
    prompt: DomainPrompt = Field(default_factory=DomainPrompt)

    # 领域词命中不足时是否补充通用名词关键词（由配置再叠加开关）
    include_generic_terms: bool = True
    # 补充通用关键词的候选倍数；最终仍受 graph_max_keywords_per_chunk 限制
    generic_candidate_factor: int = Field(default=2, ge=1)

    # 运行期索引：别名键 → (标准名, 类型, 权重)
    _alias_index: dict[str, tuple[str, str, float]] = PrivateAttr(
        default_factory=dict
    )
    # 运行期索引：全部原始别名（用于构建最长优先匹配正则）
    _surfaces: list[str] = PrivateAttr(default_factory=list)
    _stopword_keys: frozenset[str] = PrivateAttr(default_factory=frozenset)
    _entity_type_names: frozenset[str] = PrivateAttr(default_factory=frozenset)
    _relation_names: frozenset[str] = PrivateAttr(default_factory=frozenset)
    _pattern: re.Pattern[str] | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _validate_profile(self) -> "DomainProfile":
        """加载期 fail-fast：重复实体、冲突别名、未声明类型都直接报错。"""
        type_weights: dict[str, float] = {}
        for item in self.entity_types:
            if item.name in type_weights:
                raise ValueError(f"实体类型重复: {item.name}")
            type_weights[item.name] = item.weight
        if not type_weights:
            raise ValueError("领域画像至少需要声明一个 entity_type")
        if not self.entities:
            raise ValueError("领域画像至少需要一个实体")

        relation_names: set[str] = set()
        for item in self.relations:
            if item.name in relation_names:
                raise ValueError(f"关系类型重复: {item.name}")
            relation_names.add(item.name)

        rule_names: set[str] = set()
        for rule in self.relation_rules:
            if rule.name not in relation_names:
                raise ValueError(
                    f"关系规则 {rule.name} 未在 relations 中声明"
                )
            if not rule.subject_types or not rule.object_types:
                raise ValueError(
                    f"关系规则 {rule.name} 必须声明 subject_types/object_types"
                )
            unknown = sorted(
                {
                    *rule.subject_types,
                    *rule.object_types,
                }
                - set(type_weights)
            )
            if unknown:
                raise ValueError(
                    f"关系规则 {rule.name} 引用了未声明实体类型: {unknown}"
                )
            if rule.name in rule_names:
                raise ValueError(f"关系规则重复: {rule.name}")
            rule_names.add(rule.name)

        entity_names: set[str] = set()
        alias_owner: dict[str, str] = {}
        surfaces: list[str] = []
        for entity in self.entities:
            if entity.name in entity_names:
                raise ValueError(f"实体重复: {entity.name}")
            entity_names.add(entity.name)
            if entity.type not in type_weights:
                raise ValueError(
                    f"实体 {entity.name} 的类型未声明: {entity.type}"
                )
            for surface in [entity.name, *entity.aliases]:
                key = _key(surface)
                if not key:
                    raise ValueError(f"实体 {entity.name} 存在空别名")
                # 单字中文词在中文文本里几乎总是子串（“肝”命中“肝炎”），
                # 作为图节点噪声大于收益；与通用关键词过滤口径保持一致。
                if len(key) < 2:
                    raise ValueError(
                        f"实体 {entity.name} 的别名过短: {surface!r}"
                    )
                owner = alias_owner.get(key)
                if owner is not None and owner != entity.name:
                    raise ValueError(
                        f"别名冲突: {surface!r} 同时属于 {owner} 与 "
                        f"{entity.name}"
                    )
                alias_owner[key] = entity.name
                surfaces.append(surface)

        object.__setattr__(self, "_surfaces", surfaces)
        object.__setattr__(
            self,
            "_stopword_keys",
            frozenset(_key(item) for item in self.stopwords if _key(item)),
        )
        object.__setattr__(
            self, "_entity_type_names", frozenset(type_weights)
        )
        object.__setattr__(
            self, "_relation_names", frozenset(relation_names)
        )
        # 别名索引也在这里构建：model_post_init 早于 after-validator，
        # 放在那里会在非法 type 上先抛 KeyError，掩盖真正的校验错误。
        index: dict[str, tuple[str, str, float]] = {}
        for entity in self.entities:
            weight = (
                entity.weight
                if entity.weight is not None
                else type_weights[entity.type]
            )
            for surface in [entity.name, *entity.aliases]:
                index[_key(surface)] = (entity.name, entity.type, weight)
        object.__setattr__(self, "_alias_index", index)
        return self

    # ——————————————————— 只读属性 ———————————————————

    @property
    def relation_names(self) -> frozenset[str]:
        """领域声明的关系类型；抽取结果只能从这里取值。"""
        return self._relation_names

    @property
    def entity_type_names(self) -> frozenset[str]:
        """领域声明的实体类型。"""
        return self._entity_type_names

    @property
    def stopword_keys(self) -> frozenset[str]:
        """归一化后的领域停用词集合。"""
        return self._stopword_keys

    # ——————————————————— 匹配与归一 ———————————————————

    @property
    def pattern(self) -> re.Pattern[str]:
        """最长优先的实体匹配正则；画像加载后只编译一次。"""
        if self._pattern is None:
            # Python 正则的 alternation 从左到右尝试，因此按长度降序保证
            # “2型糖尿病”先于“糖尿病”命中，避免被短词截断。
            aliases = sorted(self._surfaces, key=len, reverse=True)
            compiled = re.compile(
                "|".join(_alias_pattern(item) for item in aliases),
                re.IGNORECASE,
            )
            object.__setattr__(self, "_pattern", compiled)
        pattern = self._pattern
        assert pattern is not None
        return pattern

    def match(self, text: str) -> list[DomainMatch]:
        """按最长优先扫描文本，返回不重叠的领域实体命中。"""
        if not text:
            return []
        matches: list[DomainMatch] = []
        for found in self.pattern.finditer(text):
            info = self._alias_index.get(_key(found.group(0)))
            if info is None:  # 理论上不会发生；防御正则/索引不同步
                continue
            name, entity_type, weight = info
            matches.append(
                DomainMatch(
                    name=name,
                    entity_type=entity_type,
                    surface=found.group(0),
                    start=found.start(),
                    end=found.end(),
                    weight=weight,
                )
            )
        return matches

    def canonicalize(self, value: str) -> str | None:
        """把标准名/别名归一为标准名；词典外返回 None。"""
        info = self._alias_index.get(_key(value))
        return info[0] if info else None

    def is_stopword(self, value: str) -> bool:
        """是否领域停用词。"""
        return _key(value) in self._stopword_keys

    def relation_prompt_lines(self) -> list[str]:
        """LLM 提示词中的关系类型清单。"""
        return [
            f"- {item.name}: {item.label or item.description}"
            for item in self.relations
        ]

    def entity_type_prompt_lines(self) -> list[str]:
        """LLM 提示词中的实体类型清单。"""
        return [
            f"- {item.name}: {item.label or item.description}"
            for item in self.entity_types
        ]
