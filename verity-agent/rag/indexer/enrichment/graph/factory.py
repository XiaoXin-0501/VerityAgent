"""图抽取器工厂：领域画像 + LLM，唯一实现。

图抽取不再提供规则/词典/混合模式：

- 领域画像只提供类型、补充词白名单、黑名单、关系白名单和提示词；
- 实体与关系必须由 LLM 从正文抽取；
- 未注入 chat model 属于配置错误，直接 fail-fast，避免静默降级。
"""

from __future__ import annotations

from typing import Any

from config.rag import IndexerConfig
from exception import RagError, RagErrorCode
from prompt import PromptRegistry
from rag.shared.events import RagEventSink

from ..domain.loader import load_domain_profile
from ..domain.models import DomainProfile
from .base import GraphExtractor
from .extractor import LLMGraphExtractor

__all__ = ["build_graph_extractor"]


def build_graph_extractor(
    config: IndexerConfig,
    *,
    profile: DomainProfile | None = None,
    chat_model: Any | None = None,
    prompts: PromptRegistry | None = None,
    cache: Any | None = None,
    event_sink: RagEventSink | None = None,
) -> GraphExtractor:
    """构造 LLM 图抽取器。

    ``profile`` / ``chat_model`` 允许调用方预先加载/复用，避免每个流水线
    重复读 JSON、重复创建模型句柄；``prompts`` 允许注入外部覆盖目录构建的
    Prompt 注册表。
    """
    if chat_model is None:
        raise RagError(
            RagErrorCode.INVALID_CONFIG,
            "图抽取必须注入 LLM chat model；领域词典不直接产出图节点",
            details={"domain_profile": config.domain_profile},
        )
    domain = profile or load_domain_profile(
        config.domain_profile,
        path=config.domain_profile_path,
        directory=config.domain_profile_dir,
    )
    return LLMGraphExtractor(
        config,
        domain,
        chat_model,
        prompts=prompts,
        cache=cache,
        event_sink=event_sink,
    )
