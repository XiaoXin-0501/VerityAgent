"""图富化：从 chunk 抽取关键词/关系。

v1 提供三类实现：

- ``RuleGraphExtractor``：英文/技术术语正则，不认识任何领域知识；
- ``DomainGraphExtractor``（``domain/extractor.py``）：领域词典命中 +
  英文/技术术语补充；
- ``LLMGraphExtractor``（``llm_graph.py``）：领域提示词驱动，可降级到词典。

中文分词引入的碎片和泛化词噪音较大，图抽取不再使用 jieba；中文领域概念
由 LLM 或领域词典负责。本模块只保留与领域无关的公共能力：技术术语正则、
关系过滤、白名单合并，以及“图抽取只看标题 + 正文”的准入规则。

设计对应《RAG 模块数据结构设计》§7.3。
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable
from typing import Protocol

from config.rag import IndexerConfig
from rag.shared.enums import BlockType
from rag.shared.models import Chunk, GraphRelation, ChunkGraphExtraction

__all__ = [
    "GENERIC_KEYWORD_BLACKLIST",
    "GENERIC_STOPWORDS",
    "GRAPH_TEXT_BLOCK_TYPES",
    "GraphExtractor",
    "RelationExtractor",
    "RuleGraphExtractor",
    "extract_keywords",
    "filter_relations",
    "graph_nearest_heading",
    "graph_source_text",
    "merge_relation_whitelist",
    "normalize_blocked_terms",
]

# 只有纯正文/列表参与图关键词与关系抽取；表格、代码、图注、公式等
# 语义密度低，抽取结果噪声大，且会污染概念/关系图。
GRAPH_TEXT_BLOCK_TYPES = frozenset({BlockType.PARAGRAPH, BlockType.LIST})

# 通用停用词（内置词表）；领域停用词在
# DomainProfile.stopwords，由领域抽取器合并使用。
GENERIC_STOPWORDS = frozenset(
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

# 通用关键词黑名单（版式/文献信息/泛化词）；命中后不进入图节点。
# 领域画像可追加 ``keyword_blacklist``，部署可追加
# ``RAG__INDEXER__GRAPH_KEYWORD_BLACKLIST``。
GENERIC_KEYWORD_BLACKLIST = frozenset(
    {
        "摘要",
        "前言",
        "引言",
        "结论",
        "讨论",
        "方法",
        "材料",
        "结果",
        "参考文献",
        "附录",
        "致谢",
        "目录",
        "章节",
        "本文",
        "本节",
        "本章",
        "全文",
        "如下",
        "上述",
        "以下",
        "例如",
        "比如",
        "如图",
        "如表",
        "上图",
        "下图",
        "上表",
        "下表",
        "表格",
        "图表",
        "公式",
        "注释",
        "备注",
        "来源",
        "版权",
        "作者",
        "单位",
        "通讯作者",
        "基金项目",
        "收稿日期",
        "修回日期",
        "病例",
        "患者",
        "doi",
        "http",
        "https",
        "www",
        "copyright",
        "abstract",
        "introduction",
        "results",
        "methods",
        "discussion",
        "materials",
        "background",
        "keywords",
        "conclusion",
        "references",
        "appendix",
        "figure",
        "table",
        "section",
        "chapter",
        "funding",
        "acknowledgements",
        "et",
        "al",
        "ibid",
    }
)

# 图/表编号、章节编号、页码等版式标签不是领域概念。
_LABEL_RE = re.compile(
    r"^(?:图|表|figure|table|fig\.?|tab\.?)\s*\d+(?:[-.]\d+)*$",
    re.IGNORECASE,
)

# 英文技术词（API、RAG、BGE-M3、C++ 等）；用保守正则避免中文分词碎片。
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


def normalize_blocked_terms(values: Iterable[str]) -> frozenset[str]:
    """归一化黑名单/停用词：去空白、小写、丢弃空值。"""
    return frozenset(
        _normalize_keyword(item).lower()
        for item in values
        if item and _normalize_keyword(item)
    )


def _normalize_keyword(value: str) -> str:
    """关键词归一化：去空白、英文小写、去掉包裹标点。"""
    text = (value or "").strip().strip(".,;:!?()[]{}<>\"'`“”‘’《》")
    if not text:
        return ""
    if re.fullmatch(r"[A-Za-z0-9_+#.\-]+", text):
        return text.lower()
    return text


def _is_valid_keyword(
    value: str,
    *,
    stopwords: Iterable[str] = (),
    blacklist: Iterable[str] = (),
) -> bool:
    """过滤空值、纯数字/符号、单字、停用词、黑名单和明显噪音。

    ``stopwords`` / ``blacklist`` 传入已归一化的集合（通用词 + 领域词），
    由调用方提供；两者分开是为了让“无检索价值”和“明确禁止入图”的语义
    在配置层可区分。
    """
    key = (value or "").strip().lower()
    if not key or key in stopwords or key in blacklist:
        return False
    if _NUMERIC_RE.match(value) or _LABEL_RE.match(value):
        return False
    # URL/邮箱/长哈希不是可检索概念。
    if "@" in value or "://" in value or _HASH_RE.match(value):
        return False
    # 中文单字信息量低；英文/技术词允许 2 个字符（如 AI、DB）。
    if len(value) < 2:
        return False
    return True


def graph_nearest_heading(chunk: Chunk) -> str:
    """返回离当前 chunk 最近的标题，不带祖先路径。

    图抽取只需要“本节标题”作为语义锚点；带完整 ``heading_path`` 会让同一
    章节下每个 chunk 都重复抽取祖先标题里的实体，造成关键词/关系重复。
    优先取 ``heading_path[-1]``；没有路径时再兼容 chunker 拼好的
    ``"一级 > 二级"`` 展示串，只取最后一级。
    """
    path = [item.strip() for item in chunk.heading_path if item and item.strip()]
    if path:
        return path[-1]
    heading = (chunk.heading or "").strip()
    if not heading:
        return ""
    return heading.split(" > ")[-1].strip()


def graph_source_text(chunk: Chunk) -> str | None:
    """返回用于图抽取的文本：最近一级标题 + 正文。

    表格/代码/图注/公式等低语义块返回 ``None``，由各抽取器直接产出空结果；
    这样图里的概念、关系与 ChunkRef 只来自正文，不会把表头、代码符号或
    公式变量误当成领域实体。
    """
    if chunk.block_type not in GRAPH_TEXT_BLOCK_TYPES:
        return None
    heading = graph_nearest_heading(chunk)
    body = (chunk.content or "").strip()
    if heading and body:
        return f"{heading}\n{body}"
    return heading or body


def extract_keywords(
    text: str,
    *,
    top_k: int = 10,
    stopwords: Iterable[str] | None = None,
    blacklist: Iterable[str] = (),
) -> list[str]:
    """抽取英文/技术术语关键词，按词频、长度和首次出现顺序稳定排序。

    图节点质量优先：不再使用 jieba 中文分词（碎片/泛化词噪音大），中文领域
    概念交给 LLM 或领域词典。这里只保留保守的 ASCII 技术词正则；``stopwords``
    为空时使用模块内置 ``GENERIC_STOPWORDS``，``blacklist`` 为明确禁止入图的词。
    """
    if not text or top_k <= 0:
        return []

    if stopwords is None:
        stopwords = GENERIC_STOPWORDS
    extra_stopwords = frozenset(stopwords)
    blocked = frozenset(blacklist)
    counts: Counter[str] = Counter()
    first_seen: dict[str, int] = {}
    display: dict[str, str] = {}

    for match in _ASCII_TERM_RE.finditer(text):
        value = _normalize_keyword(match.group(0))
        if not _is_valid_keyword(
            value, stopwords=extra_stopwords, blacklist=blocked
        ):
            continue
        key = value.lower()
        if key not in display:
            display[key] = value
            first_seen[key] = len(first_seen)
        counts[key] += 1

    ordered = sorted(
        counts,
        key=lambda key: (
            -counts[key],
            -len(key),
            first_seen[key],
        ),
    )
    return [display[key] for key in ordered[:top_k]]


def merge_relation_whitelist(
    config: IndexerConfig,
    domain_relations: Iterable[str] = (),
) -> set[str]:
    """合并配置白名单与领域画像白名单（统一小写）。

    配置负责运维侧追加，领域画像负责声明“这个领域有哪些强关系”；
    两者取并集，避免切换领域后配置未同步导致关系全部被过滤。
    """
    allowed = {item.lower() for item in config.graph_relation_whitelist}
    allowed.update(item.lower() for item in domain_relations if item)
    return allowed


def filter_relations(
    relations: Iterable[GraphRelation],
    config: IndexerConfig,
    *,
    allowed_relations: Iterable[str] | None = None,
) -> list[GraphRelation]:
    """按白名单、置信度、数量上限过滤关系；保持首次出现顺序。

    ``allowed_relations`` 为空时使用 ``config.graph_relation_whitelist``；
    领域抽取器会传入“配置 ∪ 领域画像”的合并白名单。
    """
    if allowed_relations is None:
        whitelist = {item.lower() for item in config.graph_relation_whitelist}
    else:
        whitelist = {item.lower() for item in allowed_relations}
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
    """规则图抽取器：关键词靠英文/技术术语正则，关系靠可选注入（默认空）。"""

    def __init__(
        self,
        config: IndexerConfig,
        *,
        relation_extractor: RelationExtractor | None = None,
    ) -> None:
        """保存配置与可选关系抽取器。"""
        self.config = config
        self.relation_extractor = relation_extractor
        self.blacklist = GENERIC_KEYWORD_BLACKLIST | normalize_blocked_terms(
            config.graph_keyword_blacklist
        )

    async def extract(self, chunk: Chunk) -> ChunkGraphExtraction:
        """抽取关键词；关系为空或由注入的抽取器提供。

        只处理标题 + 正文；表格/代码等低语义块返回空结果，不建图。
        """
        if self.config.graph_extraction_mode == "off":
            return ChunkGraphExtraction(chunk_id=chunk.chunk_id)

        text = graph_source_text(chunk)
        if not text:
            return ChunkGraphExtraction(chunk_id=chunk.chunk_id)
        keywords = extract_keywords(
            text,
            top_k=self.config.graph_max_keywords_per_chunk,
            stopwords=GENERIC_STOPWORDS,
            blacklist=self.blacklist,
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
