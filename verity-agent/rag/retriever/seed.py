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
from dataclasses import dataclass, field

from config.domain import DomainProfile
from rag.shared.models import RetrievedChunk
from rag.shared.noise import is_noise_term

__all__ = [
    "SeedExtraction",
    "canonicalize_entities",
    "extract_seeds_from_bm25",
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
_STOPWORDS = frozenset(
    {
        "的", "了", "是", "在", "和", "与", "及", "或", "有", "无", "为",
        "对", "中", "上", "下", "这", "那", "其", "被", "把", "将", "也",
        "都", "就", "而", "并", "还", "等", "什么", "怎么", "如何", "哪些",
        "哪个", "多少", "是否", "可以", "需要", "关于", "进行", "问题",
        "the", "and", "or", "of", "to", "in", "on", "for", "with", "is",
        "are", "was", "were", "what", "how", "which", "who", "why",
        "document", "query", "score", "term", "documents",
        "treatment", "diagnosis", "symptom", "symptoms", "therapy",
        "care", "disease", "vaccine", "vaccination", "medicine", "drug",
        # 泛化域名词：它们是类型/概念，不是具体实体，作为 seed 容易命中 hub。
        "病毒", "细菌", "真菌", "寄生虫", "细胞", "组织", "疾病", "症状",
        "药物", "药品", "疫苗", "检查", "诊断", "预防", "感染", "病例",
        "患者", "宠物", "动物",
    }
)


@dataclass(slots=True)
class SeedExtraction:
    """一次 seed 提取结果。"""

    entities: list[str] = field(default_factory=list)
    source: str = "none"
    # 解析出的原始候选（已去噪），用于日志/排障
    candidates: list[str] = field(default_factory=list)
    # 实际包含 explain 的 chunk 数
    explain_chunks: int = 0


def canonicalize_entities(
        entities: list[str],
        profiles: list[DomainProfile],
        *,
        limit: int = _DEFAULT_LIMIT,
) -> list[str]:
    """规划器实体归一：去空、去重，按画像标准名归一；画像只做辅助。"""
    result: list[str] = []
    seen: set[str] = set()
    for item in entities:
        text = " ".join(str(item or "").split()).strip(" ,，、;；。")
        if not text:
            continue
        canonical = _canonicalize(text, profiles)
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
    for term in ranked:
        if not _candidate_ok(term, query_fold=query_fold, profiles=profiles):
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
) -> bool:
    """候选词过滤：长度、版式、停用词、画像黑名单、必须出现在 query 中。"""
    value = term.strip()
    if not value or len(value) > 40:
        return False
    # 中文允许单字（如“犬”“猫”），英文至少两个字符。
    if len(value) < 2 and value.isascii():
        return False
    if is_noise_term(value):
        return False
    if value.casefold() in _STOPWORDS:
        return False
    if any(profile.is_blacklisted(value) for profile in profiles):
        return False
    if value.casefold() in query_fold:
        return True
    # explainScore 可能给出标准名/分词变体（如 query 是“犬细小”，explain 是
    # “犬细小病毒病”）；画像只在这里做辅助识别，不主动生成候选。
    return any(
        profile.canonicalize(value) or profile.match(value)
        for profile in profiles
    )
