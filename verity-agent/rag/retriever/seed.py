"""图检索 seed 实体提取。

边界（重要）：

- **领域画像不是实体枚举，也不负责生成 seed**；它只做标准名归一与黑名单过滤；
- Agent 查询规划（LLM）提取到实体时直接使用；
- 规则路径没有规划实体时，用主召回 BM25 的 ``explainScore`` 提取查询词作为
  seed；explainScore 不可用就返回空，不退回领域词典扫描。

这样 seed 来自“用户问题和实际召回”，而不是词典覆盖范围，避免画像未枚举
的实体直接丢失。
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from config.domain import DomainProfile
from rag.shared.models import RetrievedChunk
from rag.shared.noise import is_noise_term

__all__ = [
    "SeedExtraction",
    "canonicalize_entities",
    "expand_bilingual_entities",
    "extract_seeds_from_bm25",
    "is_entity_blocked",
    "normalize_blacklist",
]

# 最多从多少个高排名 BM25 chunk 的 explain 中收集候选。
_EXPLAIN_CHUNKS = 8
# 单次最多返回多少 seed，避免图扩展 hop 爆炸。
_DEFAULT_LIMIT = 8

# explainScore 常见形态：括号项、quoted 项、``term: 1.23``、``term (tf=...)``。
_PAREN_RE = re.compile(r"[\(（]([^()（）]{1,60})[\)）]")
_ASSIGN_RE = re.compile(
    r"(?:^|[\s,;])([^\s,;:=()（）]{1,40})\s*(?:[:=：])\s*[-+]?\d"
)
_QUOTED_RE = re.compile(r"[\"'“”‘’]([^\"'“”‘’]{1,40})[\"'“”‘’]")
_LINE_TERM_RE = re.compile(r"^\s*([^\s:：=()（）]{2,40})\s*\(")
# Weaviate 实际格式：`, BM25F_parvovirus_frequency:1, BM25F_..._propLength:9`
_BM25F_TERM_RE = re.compile(
    r"BM25F_(.+?)_(?:frequency|propLength|idf|tf|boost)",
    re.IGNORECASE,
)
_TOKEN_SPLIT_RE = re.compile(r"[\s,;:：=()（）\[\]{}<>\"'“”‘’/|]+")

# 查询中常见但不适合作为 seed 的高频虚词/泛化词。
# 这里只放语言层虚词；领域类型词（疾病/药物/疫苗…）不写死，运行时从
# DomainProfile.entity_types 派生，避免换领域后停用词表失真。
_STOPWORDS = frozenset(
    {
        "的", "了", "是", "在", "和", "与", "及", "或", "有", "无", "为",
        "对", "中", "上", "下", "这", "那", "其", "被", "把", "将", "也",
        "都", "就", "而", "并", "还", "等", "什么", "怎么", "如何", "哪些",
        "哪个", "多少", "是否", "可以", "需要", "关于", "进行", "问题",
        "the", "and", "or", "of", "to", "in", "on", "for", "with", "is",
        "are", "was", "were", "what", "how", "which", "who", "why",
        "document", "query", "score", "term", "documents",
    }
)


def _profile_type_terms(profiles: list[DomainProfile]) -> frozenset[str]:
    """从画像声明的实体类型派生泛化词集合（类型名 + 标签）。

    例如宠物医疗画像会派生 ``disease`` / ``疾病`` / ``drug`` / ``药物``；
    这些是类型概念，不是具体实体，作为 seed 容易命中 hub。
    """
    terms: set[str] = set()
    for profile in profiles:
        for item in profile.entity_types:
            terms.add(item.name.casefold())
            if item.label:
                # “疾病/综合征”“药物/活性成分”这类标签按分隔符拆开，
                # 让“疾病”“药物”这类泛化词也被拦截。
                terms.update(
                    part.casefold()
                    for part in re.split(r"[/、,，\s]+", item.label)
                    if part
                )
    return frozenset(terms)


@dataclass(slots=True)
class SeedExtraction:
    """一次 seed 提取结果。"""

    entities: list[str] = field(default_factory=list)
    source: str = "none"
    # 解析出的原始候选（已去噪），用于日志/排障
    candidates: list[str] = field(default_factory=list)
    # 实际包含 explain 的 chunk 数
    explain_chunks: int = 0


def normalize_blacklist(terms: Iterable[str]) -> frozenset[str]:
    """归一化实体黑名单：去空白、转小写、丢弃空值，供 seed 快速判断。"""
    result: set[str] = set()
    for item in terms:
        text = " ".join(str(item or "").split()).strip()
        if text:
            result.add(text.casefold())
    return frozenset(result)


def is_entity_blocked(
    term: str,
    profiles: list[DomainProfile],
    *,
    type_terms: frozenset[str] | None = None,
    extra_blacklist: frozenset[str] = frozenset(),
    max_chars: int = 60,
) -> bool:
    """判断实体是否应被黑名单/噪声/泛化词过滤。

    这是图 seed 与查询规划共用的唯一判定：内置版式噪声、语言停用词、
    画像 ``noise_terms``、画像实体类型词（疾病/药物…）以及运维追加的
    ``graph_noise_terms`` 都在这里拦截，避免“入库过滤掉、检索又拿来当
    seed”的漂移。
    """
    value = (term or "").strip()
    if not value or len(value) > max_chars:
        return True
    if len(value) < 2 and value.isascii():
        return True
    key = value.casefold()
    if key in extra_blacklist or key in _STOPWORDS:
        return True
    if is_noise_term(value):
        return True
    resolved_types = (
        type_terms if type_terms is not None else _profile_type_terms(profiles)
    )
    if key in resolved_types:
        return True
    return any(profile.is_noise_term(value) for profile in profiles)


def _has_cjk(text: str) -> bool:
    """判断文本是否包含中日韩统一表意文字。"""
    return any("㐀" <= char <= "鿿" for char in text or "")


def expand_bilingual_entities(
    entities: list[str],
    profiles: list[DomainProfile],
    *,
    limit: int = 12,
    extra_blacklist: frozenset[str] = frozenset(),
) -> list[str]:
    """为画像已知实体补一个另一种语言的别名，让中英文图节点都参与召回。

    只使用画像 ``aliases`` 里已有的写法，不做机器翻译；每个实体最多补
    一个另一语言写法，避免 seed 膨胀。规划器已同时给出中英文时自然去重；
    补入的写法仍要经过同一套黑名单/噪声过滤。
    """
    result = list(entities)
    original = list(entities)
    keys = {item.casefold() for item in result}
    expanded: set[str] = set()
    for profile in profiles:
        for item in profile.aliases:
            entry_key = item.name.casefold()
            if entry_key in expanded:
                continue
            surfaces = [item.name, *item.aliases]
            surface_keys = {value.casefold() for value in surfaces if value}
            matched = [
                value for value in original if value.casefold() in surface_keys
            ]
            if not matched:
                continue
            wanted_cjk = not any(_has_cjk(value) for value in matched)
            for surface in surfaces:
                value = " ".join(str(surface or "").split()).strip()
                key = value.casefold()
                if not value or key in keys:
                    continue
                if _has_cjk(value) != wanted_cjk:
                    continue
                if is_entity_blocked(
                    value,
                    profiles,
                    extra_blacklist=extra_blacklist,
                ):
                    continue
                keys.add(key)
                result.append(value)
                expanded.add(entry_key)
                if len(result) >= max(1, limit):
                    return result
                break
    return result


def canonicalize_entities(
    entities: list[str],
    profiles: list[DomainProfile],
    *,
    limit: int = _DEFAULT_LIMIT,
    extra_blacklist: frozenset[str] = frozenset(),
) -> list[str]:
    """规划器实体归一：先黑名单/噪声过滤，再去重、按画像标准名归一。"""
    result: list[str] = []
    seen: set[str] = set()
    type_terms = _profile_type_terms(profiles)
    for item in entities:
        text = " ".join(str(item or "").split()).strip(" ,，、;；。")
        if is_entity_blocked(
            text,
            profiles,
            type_terms=type_terms,
            extra_blacklist=extra_blacklist,
        ):
            continue
        canonical = _canonicalize(text, profiles)
        if is_entity_blocked(
            canonical,
            profiles,
            type_terms=type_terms,
            extra_blacklist=extra_blacklist,
        ):
            continue
        key = canonical.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(canonical)
        if len(result) >= max(1, limit):
            break
    return result


def extract_seeds_from_bm25(
        chunks: list[RetrievedChunk],
        *,
        query: str,
        profiles: list[DomainProfile],
        limit: int = _DEFAULT_LIMIT,
        extra_blacklist: frozenset[str] = frozenset(),
) -> SeedExtraction:
    """从 BM25 explainScore 中提取 seed 实体。

    ``chunks`` 按 BM25 排名传入；只读取前若干个 chunk 的
    ``metadata['bm25_explain']``。候选必须出现在原 query 中，再经过黑名单、
    版式/数值过滤和画像标准名归一。
    """
    query_text = query or ""
    query_fold = query_text.casefold()
    candidates: list[str] = []
    explain_chunks = 0
    for chunk in chunks[:_EXPLAIN_CHUNKS]:
        explain = str(chunk.metadata.get("bm25_explain") or "").strip()
        if not explain:
            continue
        explain_chunks += 1
        candidates.extend(_parse_explain_terms(explain))

    # 频率优先、首次出现次之；同一 term 在多个 chunk 中出现更可信。
    counts = Counter(candidates)
    order: dict[str, int] = {}
    for index, term in enumerate(candidates):
        order.setdefault(term, index)
    ranked = sorted(
        counts,
        key=lambda term: (-counts[term], order.get(term, 0)),
    )

    result: list[str] = []
    seen: set[str] = set()
    # 领域类型词由画像派生；与画像黑名单一起过滤“疾病/药物”这类泛化词。
    type_terms = _profile_type_terms(profiles)
    for term in ranked:
        if not _candidate_ok(
            term,
            query_fold=query_fold,
            profiles=profiles,
            type_terms=type_terms,
            extra_blacklist=extra_blacklist,
        ):
            continue
        canonical = _canonicalize(term, profiles)
        key = canonical.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(canonical)
        if len(result) >= max(1, limit):
            break
    return SeedExtraction(
        entities=result,
        source="bm25_explain" if result else "none",
        candidates=ranked[: max(1, limit) * 3],
        explain_chunks=explain_chunks,
    )


def _canonicalize(term: str, profiles: list[DomainProfile]) -> str:
    """画像辅助归一：先精确匹配标准名/别名，再取最长别名命中。

    这里只做“同一个实体的标准名统一”，不把画像当成实体来源。
    """
    for profile in profiles:
        value = profile.canonicalize(term)
        if value:
            return value
    best_name = term
    best_len = 0
    for profile in profiles:
        for match in profile.match(term):
            if len(match.surface) > best_len:
                best_name = match.name
                best_len = len(match.surface)
    return best_name


def _parse_explain_terms(text: str) -> list[str]:
    """从 Weaviate explainScore 文本中宽松提取候选词。"""
    terms: list[str] = []
    terms.extend(match.group(1) for match in _BM25F_TERM_RE.finditer(text))
    for regex in (_PAREN_RE, _QUOTED_RE, _ASSIGN_RE, _LINE_TERM_RE):
        terms.extend(match.group(1) for match in regex.finditer(text))
    # 兜底：按空白/标点切分；后续用“必须出现在 query 中”过滤噪声。
    terms.extend(_TOKEN_SPLIT_RE.split(text))
    cleaned: list[str] = []
    for term in terms:
        value = " ".join(term.split()).strip(" ,，、;；。:：()（）[]【】")
        if value:
            cleaned.append(value)
    return cleaned


def _candidate_ok(
        term: str,
        *,
        query_fold: str,
        profiles: list[DomainProfile],
        type_terms: frozenset[str] = frozenset(),
        extra_blacklist: frozenset[str] = frozenset(),
) -> bool:
    """候选词过滤：黑名单/噪声/泛化词，且必须出现在 query 或可被画像归一。"""
    if is_entity_blocked(
        term,
        profiles,
        type_terms=type_terms,
        extra_blacklist=extra_blacklist,
        max_chars=40,
    ):
        return False
    if term.casefold() in query_fold:
        return True
    # explainScore 可能给出标准名/分词变体（如 query 是“犬细小”，explain 是
    # “犬细小病毒病”）；画像只在这里做辅助识别，不主动生成候选。
    return any(
        profile.canonicalize(term) or profile.match(term)
        for profile in profiles
    )
