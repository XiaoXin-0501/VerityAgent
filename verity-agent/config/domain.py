"""垂直领域画像：主画像/子画像模型、加载器与只读索引。

画像数据随代码放在 ``config/domains``；一个部署可加载多个画像（按知识库
绑定子画像），主画像奠定垂直领域主题。图抽取器实例仍按合并后的画像复用。

职责边界（重要）：

- **画像只做辅助**：提供类型体系、补充词/别名、黑名单、关系白名单与提示词；
  **不是实体枚举，也不负责生成检索 seed**；实体必须由 LLM 抽取或由
  BM25 explainScore 从实际召回中提取；
- **主画像**（``master`` 为空）只给全局使用：Agent system prompt、
  查询规划领域目录、未绑定 KB 的兜底；不要求绑定到具体知识库；
- **子画像**（``master`` 非空）绑定知识库，用于具体知识库的图构建、
  检索期关系白名单与实体归一；
- 画像**不是**图构建专属资产：图抽取、检索 seed 归一、Agent 领域目录/
  system prompt、管理端知识库绑定都会消费它；因此模型/加载保留在
  ``config``，各消费方只依赖配置资产，真正的图抽取实现仍在
  ``rag/indexer/enrichment/``。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    model_validator,
)

from exception import RagError, RagErrorCode
from utils.text import normalize_whitespace

__all__ = [
    "BUILTIN_DOMAIN_DIR",
    "DomainEntity",
    "DomainEntityType",
    "DomainMatch",
    "DomainProfile",
    "DomainPrompt",
    "DomainRelationType",
    "domain_catalog_text",
    "list_domain_profile_names",
    "load_domain_profile",
    "load_master_profile",
    "resolve_domain_profile_path",
]

# ——————————————————— 画像模型 ———————————————————

def _key(value: str) -> str:
    """匹配键：去首尾空白、压缩空白，英文统一小写。"""
    # 水平空白统一压成一个空格，避免 “阿司匹林 片” 匹配不到 “阿司匹林片”。
    return normalize_whitespace((value or "").strip(), keep_newlines=False).lower()


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
    # 主画像名；空字符串表示自身就是主画像。子画像加载时会把主画像的
    # 公共类型/关系/黑名单/提示词合并进来，合并结果仍是完整画像。
    master: str = ""
    # 主画像展示名；由加载器在合并后写入，便于提示词与目录展示。
    master_display_name: str = ""

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
        if self.master and self.master == self.name:
            raise ValueError("画像不能把自己声明为 master")

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
    def is_master(self) -> bool:
        """是否为主画像（奠定垂直领域主题、不直接绑定知识库）。"""
        return not self.master

    @property
    def master_name(self) -> str:
        """所属主画像名；主画像返回自身名字，便于统一生成目录/提示词。"""
        return self.master or self.name

    @property
    def master_display(self) -> str:
        """所属主画像展示名；主画像未设置时退回 display_name/name。"""
        return self.master_display_name or self.display_name or self.name

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


# ——————————————————— 画像加载器 ———————————————————

# 内置领域画像目录；随代码发布，不依赖工作目录。
BUILTIN_DOMAIN_DIR = Path(__file__).resolve().parent / "domains"

# 领域名只允许字母/数字/下划线/短横线，避免路径穿越。
_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")

# 子画像可覆盖主画像的标量字段；列表字段单独合并。
_SCALAR_FIELDS = (
    "display_name",
    "version",
    "language",
    "description",
)
_NAMED_LIST_FIELDS = ("entity_types", "whitelist", "relations")


def resolve_domain_profile_path(
    name: str | None = None,
    *,
    path: str | Path | None = None,
    directory: str | Path | None = None,
) -> Path:
    """解析画像文件路径；不存在或名字非法时抛 ``RagError``。"""
    base_dir = _resolve_base_dir(directory)
    if path:
        candidate = Path(path).expanduser()
        if candidate.is_dir():
            candidate = candidate / _file_name(name or "pet_medical")
        if not candidate.is_file():
            raise _error(f"领域画像文件不存在: {candidate}", name=name)
        return candidate

    domain_name = _normalize_name(name)
    candidate = base_dir / _file_name(domain_name)
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
    """加载并校验领域画像；子画像会自动合并主画像。

    ``path`` 显式指向文件时，其主画像优先在同一目录查找；这样把一套
    ``master + child`` 放进自定义目录后可以直接使用。
    """
    base_dir = _resolve_base_dir(directory)
    profile_path = resolve_domain_profile_path(
        name,
        path=path,
        directory=base_dir,
    )
    payload = _load_payload_with_master(
        profile_path,
        directory=base_dir,
        seen=(),
    )
    return _validate(payload, source=profile_path, name=name)


def load_master_profile(
    name: str | None = None,
    *,
    path: str | Path | None = None,
    directory: str | Path | None = None,
) -> DomainProfile:
    """加载主画像；若指向子画像则返回其所属主画像。

    用于 Agent system prompt、查询规划等领域主题描述，保证配置写子画像名
    时也能拿到“宠物医疗”这一层的主题。
    """
    base_dir = _resolve_base_dir(directory)
    profile_path = resolve_domain_profile_path(name, path=path, directory=base_dir)
    payload = _load_payload_with_master(
        profile_path,
        directory=base_dir,
        seen=(),
    )
    master_name = str(payload.get("master") or "").strip()
    if not master_name:
        return _validate(payload, source=None, name=name)
    master_path = _resolve_master_path(
        master_name,
        current_file=profile_path,
        directory=base_dir,
    )
    master_payload = _load_payload_with_master(
        master_path,
        directory=base_dir,
        seen=(),
    )
    return _validate(master_payload, source=master_path, name=master_name)


def list_domain_profile_names(
    *,
    path: str | Path | None = None,
    directory: str | Path | None = None,
) -> list[str]:
    """列出目录中可用的画像名（含主画像）；按名称排序、去重。"""
    base_dir = _resolve_base_dir(directory)
    names: set[str] = set()
    if path:
        candidate = Path(path).expanduser()
        if candidate.is_file():
            # path 指向单个文件时，同目录下的兄弟画像也应可选。
            names.add(candidate.stem)
            base_dir = candidate.parent
        else:
            base_dir = candidate
    for item in base_dir.glob("*.json"):
        if item.is_file():
            names.add(item.stem)
    return sorted(names)


def domain_catalog_text(profiles: list[DomainProfile]) -> str:
    """把一组画像渲染成给 LLM 的领域目录文本。

    主画像在前、子画像缩进在后：主画像奠定主题，子画像说明知识库覆盖的
    细分方向。没有画像时返回空字符串。
    """
    if not profiles:
        return ""
    masters = [item for item in profiles if item.is_master]
    children: dict[str, list[DomainProfile]] = {}
    for item in profiles:
        if item.is_master:
            continue
        children.setdefault(item.master, []).append(item)
    # 没有主画像但传入了子画像时，按子画像的 master 名兜底展示。
    lines: list[str] = []
    emitted: set[str] = set()
    for master in masters:
        emitted.add(master.name)
        label = master.display_name or master.name
        lines.append(f"- {label}（{master.name}）：{master.description or '领域主题'}")
        for child in children.get(master.name, []):
            child_label = child.display_name or child.name
            lines.append(
                f"  - {child_label}（{child.name}）：{child.description or '细分方向'}"
            )
    for master_name, items in children.items():
        if master_name in emitted:
            continue
        emitted.add(master_name)
        lines.append(f"- {master_name}（主画像未加载）：细分方向")
        for child in items:
            child_label = child.display_name or child.name
            lines.append(
                f"  - {child_label}（{child.name}）：{child.description or '细分方向'}"
            )
    return "\n".join(lines)


def _resolve_base_dir(directory: str | Path | None) -> Path:
    """解析画像目录；相对路径按当前工作目录展开，未配置用内置目录。"""
    return Path(directory).expanduser() if directory else BUILTIN_DOMAIN_DIR


def _normalize_name(name: str | None) -> str:
    """校验并归一画像名；允许调用方直接写 ``pet_medical.json``。"""
    domain_name = (name or "pet_medical").strip()
    if domain_name.lower().endswith(".json"):
        domain_name = domain_name[:-5]
    if not _NAME_RE.match(domain_name):
        raise _error(f"领域名不合法: {domain_name!r}", name=domain_name)
    return domain_name


def _file_name(name: str) -> str:
    """领域名 → 文件名；允许调用方直接写 ``pet_medical.json``。"""
    return name if name.lower().endswith(".json") else f"{name}.json"


def _resolve_master_path(
    name: str,
    *,
    current_file: Path,
    directory: Path,
) -> Path:
    """解析主画像文件：优先与子画像同目录，其次配置目录，最后内置目录。

    这样把一套 ``master + child`` 放进自定义目录、或用 ``path`` 指向其中
    一个文件时，都能找到相邻的主画像。
    """
    domain_name = _normalize_name(name)
    candidates = [
        current_file.parent / _file_name(domain_name),
        directory / _file_name(domain_name),
    ]
    if BUILTIN_DOMAIN_DIR not in candidates:
        candidates.append(BUILTIN_DOMAIN_DIR / _file_name(domain_name))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise _error(
        f"主画像不存在: {domain_name}",
        name=domain_name,
        details={"directory": str(current_file.parent)},
    )


def _load_payload_with_master(
    file: Path,
    *,
    directory: Path,
    seen: tuple[str, ...],
) -> dict[str, Any]:
    """读取 JSON；若声明了 master，则递归合并主画像的公共底座。"""
    payload = _read_payload(file)
    master_name = str(payload.get("master") or "").strip()
    if not master_name:
        return payload
    if master_name in seen:
        raise _error(
            f"领域画像主画像存在循环引用: {' -> '.join((*seen, master_name))}",
            name=str(payload.get("name") or file.stem),
            details={"path": str(file)},
        )
    master_path = _resolve_master_path(
        master_name,
        current_file=file,
        directory=directory,
    )
    master_payload = _load_payload_with_master(
        master_path,
        directory=directory,
        seen=(*seen, master_name),
    )
    merged = _merge_payload(master_payload, payload)
    # 记录主画像展示名，供目录/提示词展示；不要求用户手写。
    merged["master_display_name"] = str(
        master_payload.get("display_name")
        or master_payload.get("name")
        or master_name
    )
    return merged


def _read_payload(file: Path) -> dict[str, Any]:
    """读取并做 JSON 对象初检；解析失败统一收敛为 RagError。"""
    try:
        payload = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise _error(
            f"领域画像读取失败: {file}: {exc}",
            details={"path": str(file)},
        ) from exc
    if not isinstance(payload, dict):
        raise _error(
            f"领域画像必须是 JSON 对象: {file}",
            details={"path": str(file)},
        )
    return payload


def _merge_payload(
    master: dict[str, Any],
    child: dict[str, Any],
) -> dict[str, Any]:
    """把主画像与子画像合并成完整 payload（子画像优先）。

    列表字段按 ``name`` 去重合并，子画像同名项覆盖主画像；``blacklist``
    按字符串合并；``prompt`` 字符串子画像非空覆盖，列表拼接。
    """
    merged: dict[str, Any] = dict(master)
    for key, value in child.items():
        if key in _NAMED_LIST_FIELDS:
            continue
        if key == "blacklist":
            continue
        if key == "prompt":
            continue
        if key in _SCALAR_FIELDS:
            if value not in (None, ""):
                merged[key] = value
            continue
        # name / master / 其他扩展字段以子画像为准；空值不覆盖。
        if value not in (None, ""):
            merged[key] = value

    for key in _NAMED_LIST_FIELDS:
        merged[key] = _merge_named_list(master.get(key), child.get(key))
    merged["blacklist"] = _merge_blacklist(
        master.get("blacklist"), child.get("blacklist")
    )
    merged["prompt"] = _merge_prompt(master.get("prompt"), child.get("prompt"))
    # name/master 必须来自子画像，避免合并后仍叫主画像名。
    merged["name"] = child.get("name") or master.get("name")
    merged["master"] = child.get("master") or ""
    return merged


def _merge_named_list(
    base: Any,
    extra: Any,
) -> list[dict[str, Any]]:
    """按 ``name`` 合并对象列表；子画像覆盖同名项，顺序保持主画像在前。"""
    result: list[dict[str, Any]] = []
    index: dict[str, int] = {}
    for source in (base, extra):
        if not isinstance(source, list):
            continue
        for item in source:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if name and name in index:
                result[index[name]] = item
                continue
            if name:
                index[name] = len(result)
            result.append(item)
    return result


def _merge_blacklist(base: Any, extra: Any) -> list[str]:
    """合并黑名单并保序去重。"""
    values: list[str] = []
    for source in (base, extra):
        if not isinstance(source, list):
            continue
        for item in source:
            value = str(item).strip()
            if value and value not in values:
                values.append(value)
    return values


def _merge_prompt(base: Any, extra: Any) -> dict[str, Any]:
    """合并提示词片段：字符串子画像非空覆盖，列表拼接去重。"""
    if not isinstance(base, dict):
        base = {}
    if not isinstance(extra, dict):
        extra = {}
    merged: dict[str, Any] = dict(base)
    for key, value in extra.items():
        if isinstance(value, list):
            values = [
                str(item)
                for item in [*(merged.get(key) or []), *value]
                if str(item).strip()
            ]
            merged[key] = list(dict.fromkeys(values))
        elif value not in (None, ""):
            merged[key] = value
    return merged


def _validate(
    payload: dict[str, Any],
    *,
    source: Path | None,
    name: str | None,
) -> DomainProfile:
    """Pydantic 校验；所有失败统一收敛为 ``RagError(INVALID_CONFIG)``。"""
    try:
        return DomainProfile.model_validate(payload)
    except ValidationError as exc:
        message = f"领域画像不合法: {source}: {exc}" if source else f"领域画像不合法: {exc}"
        raise _error(
            message,
            name=name or str(payload.get("name") or ""),
            details={"path": str(source)} if source else None,
        ) from exc


def _error(
    message: str,
    *,
    name: str | None = None,
    details: dict[str, str] | None = None,
) -> RagError:
    """统一收敛为 RagError(INVALID_CONFIG)，便于启动期直接暴露配置问题。"""
    merged: dict[str, str] = {"profile": name or "pet_medical"}
    merged.update(details or {})
    return RagError(
        RagErrorCode.INVALID_CONFIG,
        message,
        details=merged,
    )
