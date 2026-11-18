"""入库富化：图实体/关系抽取。

领域画像已迁到 ``config.domain``（模型/加载/校验），本包只保留图抽取实现。
表格/代码的语义描述由 ``chunkers/describer.py`` 在切片阶段生成；本包只处理
标题 + 正文（``PARAGRAPH/LIST``），表格/代码等低语义块不建图。入口是
``build_graph_extractor``：领域画像只做提示/归一，实体与关系全部由 LLM 抽取。
"""

from .base import (
    GENERIC_KEYWORD_BLACKLIST,
    GRAPH_TEXT_BLOCK_TYPES,
    GraphExtractor,
    filter_relations,
    graph_nearest_heading,
    graph_source_text,
    is_valid_keyword,
    normalize_blocked_terms,
    normalize_keyword,
)
from .extractor import LLMGraphExtractor
from .factory import build_graph_extractor

__all__ = [
    "GENERIC_KEYWORD_BLACKLIST",
    "GRAPH_TEXT_BLOCK_TYPES",
    "GraphExtractor",
    "LLMGraphExtractor",
    "build_graph_extractor",
    "filter_relations",
    "graph_nearest_heading",
    "graph_source_text",
    "is_valid_keyword",
    "normalize_blocked_terms",
    "normalize_keyword",
]
