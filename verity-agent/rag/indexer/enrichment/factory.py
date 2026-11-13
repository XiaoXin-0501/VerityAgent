"""图抽取器工厂：按配置选择通用规则 / 领域词典 / LLM / 混合。

工厂是“配置 → 实现”的唯一入口：

- ``rule``：不加载领域画像，兼容旧行为；
- ``lexicon``（默认）：加载领域画像，词典为主、jieba 补充；
- ``llm``：领域提示词 + 模型，失败降级到词典；
- ``hybrid``：词典与 LLM 合并；
- ``off``：返回不做事的抽取器，由流水线跳过建图阶段。

没有注入 chat model 时，``llm/hybrid`` 会显式降级到词典并打 warning，
避免“配置写了 LLM 但实际静默走规则”的不可见行为。
"""

from __future__ import annotations

import logging
from typing import Any

from config.rag import IndexerConfig
from exception import RagError, RagErrorCode
from prompt import PromptRegistry

from .domain.extractor import DomainGraphExtractor
from .domain.loader import load_domain_profile
from .domain.models import DomainProfile
from .graph import GraphExtractor, RelationExtractor, RuleGraphExtractor
from .llm_graph import HybridGraphExtractor, LLMGraphExtractor
from ..policy import IndexerPolicy

__all__ = ["GRAPH_EXTRACTION_MODES", "build_graph_extractor"]

logger = logging.getLogger(__name__)

# 支持的抽取模式；未知值直接 fail-fast，避免拼写错误导致静默降级。
GRAPH_EXTRACTION_MODES = frozenset({"off", "rule", "lexicon", "llm", "hybrid"})


def build_graph_extractor(
    config: IndexerConfig,
    *,
    profile: DomainProfile | None = None,
    chat_model: Any | None = None,
    relation_extractor: RelationExtractor | None = None,
    prompts: PromptRegistry | None = None,
    indexer_policy: IndexerPolicy | None = None,
) -> GraphExtractor:
    """按 ``config.graph_extraction_mode`` 构造抽取器。

    ``profile`` / ``chat_model`` 允许调用方预先加载/复用，避免每个流水线
    重复读 JSON、重复创建模型句柄；``prompts`` 允许注入外部覆盖目录构建的
    Prompt 注册表；``indexer_policy`` 提供通用停用词等索引策略。
    """
    mode = (config.graph_extraction_mode or "lexicon").strip().lower()
    if mode not in GRAPH_EXTRACTION_MODES:
        raise RagError(
            RagErrorCode.INVALID_CONFIG,
            f"不支持的图抽取模式: {config.graph_extraction_mode}",
            details={"mode": config.graph_extraction_mode},
        )
    if mode in {"off", "rule"}:
        return RuleGraphExtractor(
            config,
            relation_extractor=relation_extractor,
            policy=indexer_policy,
        )

    domain = profile or load_domain_profile(
        config.domain_profile,
        path=config.domain_profile_path,
        directory=config.domain_profile_dir,
    )
    lexicon = DomainGraphExtractor(
        config,
        domain,
        relation_extractor=relation_extractor,
        policy=indexer_policy,
    )
    if mode == "lexicon":
        return lexicon

    if chat_model is None:
        logger.warning(
            "graph_extraction_mode=%s 但未注入 chat model，"
            "降级为领域词典抽取（profile=%s）",
            mode,
            domain.name,
        )
        return lexicon

    llm = LLMGraphExtractor(
        config,
        domain,
        chat_model,
        # llm 模式自己持有词典兜底；hybrid 由外层合并，不需要重复兜底。
        fallback=lexicon if mode == "llm" else None,
        prompts=prompts,
        policy=indexer_policy,
    )
    if mode == "llm":
        return llm
    return HybridGraphExtractor(config, domain, llm, lexicon)
