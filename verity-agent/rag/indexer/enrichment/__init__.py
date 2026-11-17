"""入库富化：图实体/关系抽取。

表格/代码的语义描述由 ``chunkers/describer.py`` 在切片阶段生成；
本包只负责图抽取，且只处理标题 + 正文（``PARAGRAPH/LIST``），
表格/代码等低语义块不建图。入口是 ``build_graph_extractor``：
领域画像只做提示/归一，实体与关系全部由 LLM 抽取。
"""

from .domain import (
    BUILTIN_DOMAIN_DIR,
    DomainEntity,
    DomainEntityType,
    DomainMatch,
    DomainProfile,
    DomainPrompt,
    DomainRelationType,
    load_domain_profile,
    resolve_domain_profile_path,
)
from .graph import (
    GENERIC_KEYWORD_BLACKLIST,
    GRAPH_TEXT_BLOCK_TYPES,
    GraphExtractor,
    LLMGraphExtractor,
    build_graph_extractor,
    filter_relations,
    graph_nearest_heading,
    graph_source_text,
    is_valid_keyword,
    normalize_blocked_terms,
    normalize_keyword,
)

__all__ = [
    "BUILTIN_DOMAIN_DIR",
    "DomainEntity",
    "DomainEntityType",
    "DomainMatch",
    "DomainProfile",
    "DomainPrompt",
    "DomainRelationType",
    "GENERIC_KEYWORD_BLACKLIST",
    "GRAPH_TEXT_BLOCK_TYPES",
    "GraphExtractor",
    "LLMGraphExtractor",
    "build_graph_extractor",
    "filter_relations",
    "graph_nearest_heading",
    "graph_source_text",
    "is_valid_keyword",
    "load_domain_profile",
    "normalize_blocked_terms",
    "normalize_keyword",
    "resolve_domain_profile_path",
]
