"""图富化：从 chunk 抽取关键词/关系。

v1 使用规则抽取（jieba 分词 + 词性/长度/停用词过滤），不调用 LLM：

- 只写“可检索的名词性概念”，不写 mentions/related_to 这类弱关系；
- 关系抽取默认留空，通过 ``relation_extractor`` 预留 LLM/领域规则接入；
- 所有关系统一经过白名单、数量上限和去重，再交给 ``GraphIndexStore``。

设计对应《RAG 模块数据结构设计》§7.3。
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable
from typing import Protocol

from config.rag import IndexerConfig
from rag.shared.models import Chunk, GraphRelation, ChunkGraphExtraction

__all__ = [
    "GraphExtractor",
    "RelationExtractor",
    "RuleGraphExtractor",
    "extract_keywords",
    "filter_relations",
]

# 只保留名词性词性：普通名词/专名/动名词/动名词短语/形容词性名词/英文
_KEYWORD_POS = frozenset({"n", "nr", "ns", "nt", "nz", "vn", "an", "eng"})

# 中文/英文停用词；只放高频且无检索价值的词，避免维护大词表。
_STOPWORDS = frozenset(
    {
        "这个",
        "那个",
        "这些",
        "那些",
        "可以",
        "需要",
        "使用",
        "通过",
        "进行",
        "以及",
        "或者",
        "但是",
        "因为",
        "所以",
        "如果",
        "然后",
        "我们",
        "你们",
        "他们",
        "自己",
        "一个",
        "一些",
        "什么",
        "怎么",
        "如何",
        "其中",
        "对于",
        "关于",
        "the",
        "and",
        "for",
        "with",
        "this",
        "that",
        "from",
        "into",
        "are",
        "was",
        "were",
        "will",
        "can",
        "not",
        "you",
        "your",
        "our",
        "its",
        "has",
        "have",
        "had",
        "but",
        "or",
        "of",
        "to",
        "in",
        "on",
        "is",
        "as",
        "by",
        "be",
        "it",
    }
)

# 英文技术词（API、RAG、BGE-M3、C++ 等）；jieba 不一定能完整切出。
_ASCII_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+#.\-]{1,31}")
_NUMERIC_RE = re.compile(r"^[\d\W_]+$", re.UNICODE)
# 8 位以上纯十六进制串（uuid 片段/内容哈希）不作为关键词。
_HASH_RE = re.compile(r"^[0-9a-f]{8,}$", re.IGNORECASE)


class GraphExtractor(Protocol):
    """图抽取协议；IndexingPipeline 只依赖这一层。"""

    async def extract(self, chunk: Chunk) -> ChunkGraphExtraction:
        """从单个 chunk 抽取关键词与关系。"""
        ...


# 可注入的关系抽取器：返回 (subject, relation, object, confidence) 列表。
# 默认不注入，规则模式只抽关键词；后续接 LLM 时实现同一签名即可。
RelationExtractor = Callable[
    [Chunk], Awaitable[Iterable[GraphRelation]]
]


def _normalize_keyword(value: str) -> str:
    """关键词归一化：去空白、英文小写、去掉包裹标点。"""
    text = (value or "").strip().strip(".,;:!?()[]{}<>\"'`“”‘’《》")
    if not text:
        return ""
    if re.fullmatch(r"[A-Za-z0-9_+#.\-]+", text):
        return text.lower()
    return text


def _is_valid_keyword(value: str) -> bool:
    """过滤空值、纯数字/符号、单字、停用词和明显噪音。"""
    if not value or value in _STOPWORDS:
        return False
    if _NUMERIC_RE.match(value):
        return False
    # URL/邮箱/长哈希不是可检索概念。
    if "@" in value or "://" in value or _HASH_RE.match(value):
        return False
    # 中文单字信息量低；英文/技术词允许 2 个字符（如 AI、DB）。
    if len(value) < 2:
        return False
    return True


def extract_keywords(text: str, *, top_k: int = 10) -> list[str]:
    """从文本抽取关键词，按词频、长度和首次出现顺序稳定排序。

    策略：
    1. jieba 精确模式分词，保留名词性词性；
    2. 额外用正则捞英文/技术术语，避免分词器把 ``BGE-M3`` 切碎；
    3. 去重（大小写不敏感）后按词频降序，频次相同按长度降序、首次出现升序，
       保证同一输入每次结果一致。
    """
    if not text or top_k <= 0:
        return []

    import jieba.posseg as pseg

    counts: Counter[str] = Counter()
    first_seen: dict[str, int] = {}
    display: dict[str, str] = {}

    # 先捞完整英文/技术词，供后面过滤被切碎的片段（BGE / M3 → BGE-M3）。
    ascii_terms = {
        _normalize_keyword(match.group(0)).lower()
        for match in _ASCII_TERM_RE.finditer(text)
    }

    def add(raw: str, *, from_ascii: bool = False) -> None:
        value = _normalize_keyword(raw)
        if not _is_valid_keyword(value):
            return
        key = value.lower()
        # jieba 可能把 "BGE-M3" 拆成 bge/m3；已有完整术语时丢掉子串。
        if (
            not from_ascii
            and re.fullmatch(r"[a-z0-9_+#.\-]+", key)
            and any(key != term and key in term for term in ascii_terms)
        ):
            return
        if key not in display:
            display[key] = value
            first_seen[key] = len(first_seen)
        counts[key] += 1

    for term in ascii_terms:
        add(term, from_ascii=True)
    for token in pseg.cut(text):
        if token.flag in _KEYWORD_POS:
            add(token.word)

    ordered = sorted(
        counts,
        key=lambda key: (
            -counts[key],
            -len(key),
            first_seen[key],
        ),
    )
    return [display[key] for key in ordered[:top_k]]


def filter_relations(
    relations: Iterable[GraphRelation],
    config: IndexerConfig,
) -> list[GraphRelation]:
    """按白名单、置信度、数量上限过滤关系；保持首次出现顺序。"""
    whitelist = {item.lower() for item in config.graph_relation_whitelist}
    seen: set[tuple[str, str, str]] = set()
    result: list[GraphRelation] = []
    for relation in relations:
        subject = _normalize_keyword(relation.subject)
        obj = _normalize_keyword(relation.object)
        name = (relation.relation or "").strip().lower()
        if not subject or not obj or not name:
            continue
        if name not in whitelist:
            continue
        confidence = (
            relation.confidence
            if relation.confidence is not None
            else 0.0
        )
        if confidence < config.graph_min_confidence:
            continue
        key = (subject.lower(), name, obj.lower())
        if key in seen:
            continue
        seen.add(key)
        result.append(
            GraphRelation(
                subject=subject,
                relation=name,
                object=obj,
                confidence=confidence,
            )
        )
        if len(result) >= config.graph_max_relations_per_chunk:
            break
    return result


class RuleGraphExtractor:
    """规则图抽取器：关键词靠 jieba，关系靠可选注入（默认空）。"""

    def __init__(
        self,
        config: IndexerConfig,
        *,
        relation_extractor: RelationExtractor | None = None,
    ) -> None:
        """保存配置与可选关系抽取器。"""
        self.config = config
        self.relation_extractor = relation_extractor

    async def extract(self, chunk: Chunk) -> ChunkGraphExtraction:
        """抽取关键词；关系为空或由注入的抽取器提供。"""
        if self.config.graph_extraction_mode == "off":
            return ChunkGraphExtraction(chunk_id=chunk.chunk_id)

        # 标题路径参与关键词抽取：表格/代码正文很短时，标题能补足语义。
        heading = " ".join(chunk.heading_path) if chunk.heading_path else (
            chunk.heading or ""
        )
        text = f"{heading}\n{chunk.content}" if heading else chunk.content
        keywords = extract_keywords(
            text,
            top_k=self.config.graph_max_keywords_per_chunk,
        )
        relations: list[GraphRelation] = []
        if self.relation_extractor is not None:
            extracted = await self.relation_extractor(chunk)
            relations = filter_relations(extracted, self.config)
        return ChunkGraphExtraction(
            chunk_id=chunk.chunk_id,
            keywords=keywords,
            relations=relations,
        )
