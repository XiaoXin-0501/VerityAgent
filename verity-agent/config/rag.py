"""RAG 模块配置（启动层只映射，不重复模块默认值）。

字段与《RAG 模块数据结构设计》保持一致；这里先提供可启动的 v1 默认值，
解析/切片/检索的完整策略在实现 Indexer/Retriever 时继续细化。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["RagConfig", "IndexerConfig", "RetrieverConfig"]


class IndexerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # 用于入库的 embedding 应用模型 id
    embedding_model_id: str | None = None
    # 向量维度；必须与 Weaviate collection 对齐
    embedding_dimensions: int = Field(default=1024, gt=0)
    # embedding 版本；换模型/维度时递增，触发整 KB 重灌
    embedding_version: str = "v1"
    # 目标 chunk token 大小
    chunk_size_tokens: int = Field(default=512, ge=64)
    # 相邻 chunk 重叠 token
    chunk_overlap_tokens: int = Field(default=64, ge=0)
    # 切片策略版本；写入 DocChunk.chunking_version
    chunking_version: str = "v1"
    # 低于该字符数的块直接丢弃
    min_chunk_chars: int = Field(default=30, ge=0)
    # 超过该字符数的块截断/拆分
    max_chunk_chars: int = Field(default=8_000, ge=1)
    # 是否启用入库降噪质量闸门
    quality_gate_enabled: bool = True
    # v1 入库始终建图；检索扩展由 RetrieverConfig 控制（D-55/D-56）
    graph_build_enabled: bool = True
    # 入库任务最大重试次数；超过视为 dead
    index_max_attempts: int = Field(default=5, ge=1)
    # 入库 worker 租约时长（秒）
    index_lease_seconds: int = Field(default=300, ge=30)
    # 入库失败退避基数（秒）
    index_retry_backoff_s: float = Field(default=30.0, gt=0)


class RetrieverConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # 混合召回候选数
    top_k: int = Field(default=20, ge=1)
    # rerank 后保留数
    rerank_top_k: int = Field(default=8, ge=1)
    # 混合检索中向量/BM25 权重（0=纯 BM25，1=纯向量）
    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    # 低于该分数的 chunk 丢弃
    min_score: float = 0.0
    # 是否启用 rerank
    rerank_enabled: bool = True
    # rerank 应用模型 id（引用 PROVIDERS__RERANK__*__MODELS[].id）
    rerank_model_id: str | None = None
    # 是否启用 query 改写（D-81，由配置类控制）
    query_rewrite_enabled: bool = False
    # query 改写最多生成几个变体
    query_rewrite_max_variants: int = Field(default=2, ge=1)
    # query 改写应用模型 id；为空时使用 LLM 默认模型
    query_rewrite_model_id: str | None = None
    # 是否启用图检索扩展；默认关闭（D-55）
    graph_expansion_enabled: bool = False
    # 图扩展最大跳数；v1 只允许 1
    graph_max_hops: int = Field(default=1, ge=1)
    # 检索结果缓存 TTL（秒）；0 表示不缓存
    cache_ttl_s: int = Field(default=300, ge=0)


class RagConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # 入库配置
    indexer: IndexerConfig = Field(default_factory=IndexerConfig)
    # 检索配置
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    # 文档描述生成失败时保持 NULL，进入人工队列（P1-17）
    # 描述有效的最小字符数；低于视为提取失败
    description_min_chars: int = Field(default=40, ge=0)
    # 描述最大字符数；超过截断
    description_max_chars: int = Field(default=500, ge=1)
