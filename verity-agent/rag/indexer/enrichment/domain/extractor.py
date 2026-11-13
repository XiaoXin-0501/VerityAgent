"""领域词典图抽取器。

关键词来源分三层，按优先级合并：

1. **领域实体命中**：``DomainProfile`` 中最长优先的词典匹配；实体类型权重、
   词频、是否出现在标题路径都会参与打分；
2. **通用名词补充**：jieba 词性过滤，仅补充领域词典没有覆盖的概念；
   可用 ``graph_include_generic_terms`` / ``profile.include_generic_terms`` 关闭；
3. **可选关系抽取**：仍通过注入的 ``RelationExtractor`` 扩展（默认不抽），
   返回值统一经领域关系白名单过滤。

该抽取器不调用 LLM，适合无模型或强调确定性的场景；需要语义泛化时用
``LLMGraphExtractor`` / ``HybridGraphExtractor``。
"""

from __future__ import annotations

import logging

from config.rag import IndexerConfig
from rag.shared.models import Chunk, ChunkGraphExtraction, GraphRelation

from ..graph import (
    RelationExtractor,
    _normalize_keyword,
    extract_keywords,
    filter_relations,
    merge_relation_whitelist,
)
from ...policy import IndexerPolicy, get_indexer_policy
from .models import DomainMatch, DomainProfile

__all__ = [
    "DomainGraphExtractor",
    "canonicalize_relations",
    "merge_keywords",
    "merged_allowed_relations",
]

logger = logging.getLogger(__name__)


class DomainGraphExtractor:
    """词典 + 分词关键词抽取器；一个实例绑定一个领域画像。"""

    def __init__(
        self,
        config: IndexerConfig,
        profile: DomainProfile,
        *,
        relation_extractor: RelationExtractor | None = None,
        policy: IndexerPolicy | None = None,
    ) -> None:
        """保存配置/画像/策略，并构建私有 jieba 分词器。

        使用 ``jieba.Tokenizer`` 而不是全局 jieba，避免不同领域/实例之间
        互相污染用户词典。
        """
        self.config = config
        self.profile = profile
        self.relation_extractor = relation_extractor
        self.policy = policy or get_indexer_policy()
        # 通用停用词与领域停用词合并，避免通用词表只在一处生效。
        self._stopwords = frozenset(self.policy.graph.generic_stopwords) | (
            profile.stopword_keys
        )
        self._postokenizer = _build_postokenizer(profile)

    def extract_keywords(self, text: str, *, heading: str = "") -> list[str]:
        """按“领域实体优先、通用名词补充”的规则返回关键词。"""
        if self.config.graph_extraction_mode == "off" or not text:
            return []

        top_k = self.config.graph_max_keywords_per_chunk
        matches = self.profile.match(text)
        domain_terms = self._score_domain_terms(matches, heading)

        include_generic = (
            self.config.graph_include_generic_terms
            and self.profile.include_generic_terms
        )
        if not include_generic:
            return [name for name, _score in domain_terms][:top_k]

        # 通用词多取一些候选，过滤掉被测到的领域实体覆盖的片段后再补位。
        candidates = extract_keywords(
            text,
            top_k=top_k * self.profile.generic_candidate_factor,
            stopwords=self._stopwords,
            postokenizer=self._postokenizer,
        )
        matched_keys = {_normalize_keyword(item.surface).lower() for item in matches}
        generic = [
            term
            for term in candidates
            if self._is_generic_term(term, matched_keys)
        ]
        return merge_keywords(
            [name for name, _score in domain_terms],
            generic,
            top_k=top_k,
        )

    async def extract(self, chunk: Chunk) -> ChunkGraphExtraction:
        """从单个 chunk 抽取关键词；关系默认留空或由注入器提供。"""
        if self.config.graph_extraction_mode == "off":
            return ChunkGraphExtraction(chunk_id=chunk.chunk_id)

        text = _chunk_text(chunk)
        heading = (
            " ".join(chunk.heading_path)
            if chunk.heading_path
            else (chunk.heading or "")
        )
        keywords = self.extract_keywords(text, heading=heading)
        relations: list[GraphRelation] = []
        if self.relation_extractor is not None:
            try:
                extracted = await self.relation_extractor(chunk)
            except Exception as exc:  # noqa: BLE001 - 关系抽取失败不应阻断入库
                logger.warning(
                    "领域关系抽取失败，仅保留关键词: chunk_id=%s err=%s",
                    chunk.chunk_id,
                    exc,
                )
                extracted = []
            relations = filter_relations(
                canonicalize_relations(list(extracted), self.profile),
                self.config,
                allowed_relations=merged_allowed_relations(
                    self.config, self.profile
                ),
            )
        return ChunkGraphExtraction(
            chunk_id=chunk.chunk_id,
            keywords=keywords,
            relations=relations,
            confidence=1.0 if keywords else None,
        )

    # ——————————————————— 内部打分 ———————————————————

    def _score_domain_terms(
        self, matches: list[DomainMatch], heading: str = ""
    ) -> list[tuple[str, float]]:
        """领域实体打分：类型权重 + 词频 + 标题路径加成。

        先按分数降序、同分按首次出现顺序，保证同一输入结果确定。
        """
        heading_key = _normalize_keyword(heading).lower()
        stats: dict[str, dict[str, object]] = {}
        for index, match in enumerate(matches):
            entry = stats.setdefault(
                match.name,
                {
                    "weight": match.weight,
                    "count": 0,
                    "first": index,
                    "in_heading": False,
                },
            )
            entry["count"] = int(entry["count"]) + 1
            # 标题里的实体通常是本节主题，给一个固定加成。
            if heading_key and match.surface.lower() in heading_key:
                entry["in_heading"] = True

        scored: list[tuple[str, float]] = []
        for name, entry in stats.items():
            if name.lower() in self._stopwords or self.profile.is_stopword(name):
                continue
            weight = float(entry["weight"])
            count = int(entry["count"])
            score = weight
            if count > 1:
                score += min(count - 1, 3) * 0.2
            if entry["in_heading"]:
                score += 0.3
            scored.append((name, score))
        scored.sort(key=lambda item: (-item[1], _first_index(stats, item[0])))
        return scored

    def _is_generic_term(self, term: str, matched_keys: set[str]) -> bool:
        """过滤与领域实体重复/被覆盖的通用词。

        - 本身是领域词（如未命中的别名）→ 不重复补；
        - 是某个已命中实体的子串（如“糖尿病”之于“2型糖尿病”）→ 噪声；
        - 领域停用词 → 丢弃。
        """
        key = _normalize_keyword(term).lower()
        if not key or key in matched_keys:
            return False
        if self.profile.canonicalize(term) is not None:
            return False
        if self.profile.is_stopword(term):
            return False
        return not any(key != item and key in item for item in matched_keys)


def _chunk_text(chunk: Chunk) -> str:
    """把标题路径与正文拼成抽取文本；标题为空时只用正文。"""
    heading = (
        " ".join(chunk.heading_path)
        if chunk.heading_path
        else (chunk.heading or "")
    )
    return f"{heading}\n{chunk.content}" if heading else chunk.content


def _build_postokenizer(profile: DomainProfile):
    """构建带领域用户词典的私有 POS 分词器。

    领域词加进分词器后，通用关键词不会被切成“糖尿/病”这类碎片；
    词频给高值，保证分词器优先保留完整词。
    """
    import jieba
    import jieba.posseg as pseg

    tokenizer = jieba.Tokenizer()
    for entity in profile.entities:
        tokenizer.add_word(entity.name, freq=100_000)
        for alias in entity.aliases:
            tokenizer.add_word(alias, freq=100_000)
    return pseg.POSTokenizer(tokenizer)


def _first_index(stats: dict[str, dict[str, object]], name: str) -> int:
    """取实体首次出现位置，用于同分排序。"""
    return int(stats[name]["first"])


def merge_keywords(
    primary: list[str],
    secondary: list[str],
    *,
    top_k: int,
) -> list[str]:
    """按顺序合并两组关键词并去重（大小写不敏感），primary 优先。"""
    result: list[str] = []
    seen: set[str] = set()
    for term in [*primary, *secondary]:
        key = _normalize_keyword(term).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(term)
        if len(result) >= top_k:
            break
    return result


def canonicalize_relations(
    relations: list[GraphRelation],
    profile: DomainProfile,
) -> list[GraphRelation]:
    """把关系两端归一为标准实体名；词典外保留原词。"""
    result: list[GraphRelation] = []
    for relation in relations:
        subject = profile.canonicalize(relation.subject) or _normalize_keyword(
            relation.subject
        )
        obj = profile.canonicalize(relation.object) or _normalize_keyword(
            relation.object
        )
        if not subject or not obj:
            continue
        result.append(
            relation.model_copy(update={"subject": subject, "object": obj})
        )
    return result


def merged_allowed_relations(
    config: IndexerConfig,
    profile: DomainProfile,
) -> set[str]:
    """领域关系白名单 ∪ 配置白名单。

    领域画像负责“这个领域有哪些强关系”，配置负责在运维侧临时追加；
    两者合并可以避免切换领域后把已声明关系全部过滤掉。
    """
    return merge_relation_whitelist(config, profile.relation_names)
