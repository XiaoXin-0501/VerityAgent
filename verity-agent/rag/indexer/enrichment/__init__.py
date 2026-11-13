"""入库富化：图关键词/关系抽取（表格/代码摘要后续接入）。

入口是 ``build_graph_extractor``：按配置选择规则/领域词典/LLM/混合实现。
"""

from .domain import (
    BUILTIN_DOMAIN_DIR,
    DomainEntity,
    DomainEntityType,
    DomainGraphExtractor,
    DomainMatch,
    DomainProfile,
    DomainPrompt,
    DomainRelationType,
    canonicalize_relations,
    load_domain_profile,
    merge_keywords,
    merged_allowed_relations,
    resolve_domain_profile_path,
)
from .factory import GRAPH_EXTRACTION_MODES, build_graph_extractor
from .graph import (
    GraphExtractor,
    RelationExtractor,
    RuleGraphExtractor,
    extract_keywords,
    filter_relations,
    merge_relation_whitelist,
)
from .llm_graph import HybridGraphExtractor, LLMGraphExtractor

__all__ = [
    "BUILTIN_DOMAIN_DIR",
    "GRAPH_EXTRACTION_MODES",
    "DomainEntity",
    "DomainEntityType",
    "DomainGraphExtractor",
    "DomainMatch",
    "DomainProfile",
    "DomainPrompt",
    "DomainRelationType",
    "GraphExtractor",
    "HybridGraphExtractor",
    "LLMGraphExtractor",
    "RelationExtractor",
    "RuleGraphExtractor",
    "build_graph_extractor",
    "canonicalize_relations",
    "extract_keywords",
    "filter_relations",
    "load_domain_profile",
    "merge_keywords",
    "merge_relation_whitelist",
    "merged_allowed_relations",
    "resolve_domain_profile_path",
]
