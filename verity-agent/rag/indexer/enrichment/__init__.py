"""入库富化：图关键词/关系抽取。

表格/代码的语义描述由 ``chunkers/describer.py`` 在切片阶段生成；
本包只负责图抽取，且只处理标题 + 正文（`PARAGRAPH/LIST`），
表格/代码等低语义块不建图。入口是 ``build_graph_extractor``：按配置选择
规则/领域词典/LLM/混合实现。
"""

from .domain import (
    BUILTIN_DOMAIN_DIR,
    DomainEntity,
    DomainEntityType,
    DomainGraphExtractor,
    DomainMatch,
    DomainProfile,
    DomainPrompt,
    DomainRelationExtractor,
    DomainRelationRule,
    DomainRelationType,
    canonicalize_relations,
    load_domain_profile,
    merge_keywords,
    merged_allowed_relations,
    resolve_domain_profile_path,
)
from .factory import GRAPH_EXTRACTION_MODES, build_graph_extractor
from .graph import (
    GRAPH_TEXT_BLOCK_TYPES,
    GraphExtractor,
    RelationExtractor,
    RuleGraphExtractor,
    extract_keywords,
    filter_relations,
    graph_nearest_heading,
    graph_source_text,
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
    "DomainRelationExtractor",
    "DomainRelationRule",
    "DomainProfile",
    "DomainPrompt",
    "DomainRelationType",
    "GraphExtractor",
    "GRAPH_TEXT_BLOCK_TYPES",
    "HybridGraphExtractor",
    "LLMGraphExtractor",
    "RelationExtractor",
    "RuleGraphExtractor",
    "build_graph_extractor",
    "canonicalize_relations",
    "extract_keywords",
    "filter_relations",
    "graph_nearest_heading",
    "graph_source_text",
    "load_domain_profile",
    "merge_keywords",
    "merge_relation_whitelist",
    "merged_allowed_relations",
    "resolve_domain_profile_path",
]
