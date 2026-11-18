"""图抽取公共基础：协议、文本边界、实体名/关系过滤。

图抽取只走 LLM：领域画像提供类型、补充词白名单、黑名单与关系白名单，
但**不直接扫描词典产出节点**。本模块只保留与具体模型无关的校验逻辑。

设计对应《RAG 模块数据结构设计》§7.3。
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Protocol

from config.rag import IndexerConfig
from rag.shared.enums import BlockType
from rag.shared.noise import is_noise_term
from rag.shared.models import Chunk, GraphRelation, ChunkGraphExtraction

__all__ = [
    "GENERIC_KEYWORD_BLACKLIST",
    "GRAPH_TEXT_BLOCK_TYPES",
    "GraphExtractor",
    "filter_relations",
    "graph_nearest_heading",
    "graph_source_text",
    "is_valid_keyword",
    "normalize_blocked_terms",
    "normalize_keyword",
]

# 只有纯正文/列表参与图实体与关系抽取；表格、代码、图注、公式等语义密度
# 低，抽取结果噪声大，且会污染概念/关系图。
GRAPH_TEXT_BLOCK_TYPES = frozenset({BlockType.PARAGRAPH, BlockType.LIST})

# 通用关键词黑名单（版式/文献信息/泛化词）；命中后不进入图节点。
# 领域画像可追加 ``blacklist``，部署可追加
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

class GraphExtractor(Protocol):
    """图抽取协议；IndexingPipeline 只依赖这一层。"""

    async def extract(self, chunk: Chunk) -> ChunkGraphExtraction:
        """从单个 chunk 抽取实体与关系。"""
        ...

    async def extract_batch(
        self,
        chunks: list[Chunk],
        *,
        max_concurrency: int | None = None,
    ) -> dict[str, ChunkGraphExtraction]:
        """批量抽取；返回 chunk_id → 抽取结果，顺序不作为语义。"""
        ...


def normalize_blocked_terms(values: Iterable[str]) -> frozenset[str]:
    """归一化黑名单：去空白、小写、丢弃空值。"""
    return frozenset(
        normalize_keyword(item).lower()
        for item in values
        if item and normalize_keyword(item)
    )


def normalize_keyword(value: str) -> str:
    """关键词归一化：去空白、英文小写、去掉包裹标点。"""
    text = (value or "").strip().strip(".,;:!?()[]{}<>\"'`“”‘’《》")
    if not text:
        return ""
    if re.fullmatch(r"[A-Za-z0-9_+#.\-]+", text):
        return text.lower()
    return text


def is_valid_keyword(
    value: str,
    *,
    blacklist: Iterable[str] = (),
) -> bool:
    """过滤空值、纯数字/符号、单字、黑名单和明显噪音。"""
    key = (value or "").strip().lower()
    if not key or key in blacklist:
        return False
    # 纯数值/版式标签/URL/邮箱/长哈希不是可检索概念。
    if is_noise_term(value):
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

    表格/代码/图注/公式等低语义块返回 ``None``，由抽取器直接产出空结果；
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


def filter_relations(
    relations: Iterable[GraphRelation],
    config: IndexerConfig,
    *,
    allowed_relations: Iterable[str],
) -> list[GraphRelation]:
    """按领域关系白名单、置信度、数量上限过滤；保持首次出现顺序。

    ``allowed_relations`` 由调用方传入领域画像声明的关系集合；关系抽取
    不再接受部署级通用白名单，避免把 ``implements`` 等软件关系写进领域图。
    """
    whitelist = {item.lower() for item in allowed_relations}
    seen: set[tuple[str, str, str]] = set()
    result: list[GraphRelation] = []
    for relation in relations:
        subject = normalize_keyword(relation.subject)
        obj = normalize_keyword(relation.object)
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
                subject_type=relation.subject_type,
                object_type=relation.object_type,
            )
        )
        if len(result) >= config.graph_max_relations_per_chunk:
            break
    return result
