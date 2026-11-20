"""配置与 setting 的统一入口。

所有配置类集中在 ``config`` 包内，按领域拆分文件；业务模块不再各自维护
配置类，也不再保留兼容性的 ``setting.py``。

- ``config.app``：App / Server / Auth
- ``config.infrastructure``：PG / Redis / MinIO / Weaviate / Neo4j
- ``config.provider``：Provider 连接配置与 ProviderKind / ChatApiStyle
- ``config.model``：独立模型清单 ModelConfig（id/provider/model/kind/能力/价格/defaults）
- ``config.llm``：LLM 应用层配置（model_ids/default_model_id/fallback）
- ``config.prompt``：Prompt 模板加载/覆盖/渲染策略
- ``config.message``：消息限制
- ``config.agent``：Agent 运行限制与流式
- ``config.rag``：Indexer / Retriever
- ``config.hooks``：Hooks
- ``config.tools``：工具选择/路由/批量/入参保护
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import Field, model_validator

from .agent import (
    AgentConfig,
    AgentPolicy,
    AgentStreamConfig,
    QueryPlanConfig,
    ReflectionConfig,
)
from .app import AppConfig, AuthConfig, LoggingConfig, ServerConfig
from .base import SOURCE_ROOT, EnvSettings
from .hooks import (
    AuditHookConfig,
    HooksConfig,
    LoggingHookConfig,
    MetricsHookConfig,
    RedactionConfig,
    TokenUsageHookConfig,
    TracingHookConfig,
)
from .infrastructure import (
    MinioBuckets,
    MinioConfig,
    Neo4jConfig,
    PGConfig,
    RedisConfig,
    WeaviateConfig,
)
from .llm import LLMConfig, StructuredOutputConfig, TokenizerConfig
from .memory import MemoryConfig
from .message import MessageConfig
from .model import ModelConfig
from .prompt import PromptConfig
from .provider import (
    ChatApiStyle,
    ChatProviderConfig,
    EmbeddingProviderConfig,
    ProviderConfig,
    ProviderKind,
    RerankProviderConfig,
)
from .rag import (
    DedupPolicyConfig,
    ImagePdfConfig,
    IndexerConfig,
    OcrConfig,
    QualityPolicyConfig,
    RagConfig,
    RetrieverConfig,
)
from .tools import ToolRoutingConfig, ToolRoutingMode, ToolsConfig

__all__ = [
    "Settings",
    "settings",
    # app
    "AppConfig",
    "ServerConfig",
    "AuthConfig",
    "LoggingConfig",
    # infrastructure
    "PGConfig",
    "RedisConfig",
    "MinioConfig",
    "MinioBuckets",
    "WeaviateConfig",
    "Neo4jConfig",
    # provider
    "ProviderKind",
    "ChatApiStyle",
    "ProviderConfig",
    "ChatProviderConfig",
    "EmbeddingProviderConfig",
    "RerankProviderConfig",
    # llm
    "LLMConfig",
    "StructuredOutputConfig",
    "TokenizerConfig",
    # memory
    "MemoryConfig",
    # prompt
    "PromptConfig",
    # message
    "MessageConfig",
    "ModelConfig",
    # agent
    "AgentConfig",
    "AgentPolicy",
    "AgentStreamConfig",
    "QueryPlanConfig",
    "ReflectionConfig",
    # rag
    "RagConfig",
    "OcrConfig",
    "ImagePdfConfig",
    "IndexerConfig",
    "RetrieverConfig",
    "QualityPolicyConfig",
    "DedupPolicyConfig",
    # hooks
    "HooksConfig",
    "LoggingHookConfig",
    "TokenUsageHookConfig",
    "MetricsHookConfig",
    "TracingHookConfig",
    "AuditHookConfig",
    "RedactionConfig",
    # tools
    "ToolsConfig",
    "ToolRoutingConfig",
    "ToolRoutingMode",
]


class Settings(EnvSettings):
    """全局配置聚合；字段名决定环境变量前缀（如 pg → PG__*）。"""

    # 应用元信息 / 运行环境
    app: AppConfig = Field(default_factory=AppConfig)
    # ASGI 服务启动参数
    server: ServerConfig = Field(default_factory=ServerConfig)
    # JWT 认证
    auth: AuthConfig = Field(default_factory=AuthConfig)

    # PostgreSQL
    pg: PGConfig = Field(default_factory=PGConfig)
    # Redis
    redis: RedisConfig = Field(default_factory=RedisConfig)
    # MinIO 对象存储
    minio: MinioConfig = Field(default_factory=MinioConfig)
    # Weaviate 向量库
    weaviate: WeaviateConfig = Field(default_factory=WeaviateConfig)
    # Neo4j 图数据库
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)

    # LLM 应用层默认策略
    llm: LLMConfig = Field(default_factory=LLMConfig)
    # 长期记忆抽取 worker
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    # Prompt 模板目录/覆盖/严格渲染
    # 字段名用 prompts，避免与 shell 常见的 PROMPT 环境变量冲突
    prompts: PromptConfig = Field(default_factory=PromptConfig)
    # 模型目录文件；为空时使用下方 MODELS（JSON 数组）
    models_file: str = ""
    # 模型清单：provider + model + capabilities/pricing/defaults；
    # 文件存在时以文件为准，MODELS 作为 fallback
    models: list[ModelConfig] = Field(default_factory=list)
    # 消息协议与附件上限
    message: MessageConfig = Field(default_factory=MessageConfig)
    # Agent 运行限制与流式
    agent: AgentConfig = Field(default_factory=AgentConfig)
    # RAG 入库/检索
    rag: RagConfig = Field(default_factory=RagConfig)
    # Hooks 观察者
    hooks: HooksConfig = Field(default_factory=HooksConfig)
    # 工具选择、路由、批量与入参保护
    tools: ToolsConfig = Field(default_factory=ToolsConfig)

    @model_validator(mode="after")
    def _load_models_file(self) -> "Settings":
        """加载 models_file 指向的 JSON/JSONC 模型清单。"""
        if not self.models_file:
            return self
        path = Path(self.models_file)
        if not path.is_absolute():
            path = SOURCE_ROOT / path
        if not path.exists():
            raise ValueError(f"模型配置文件不存在: {path}")
        try:
            text = path.read_text(encoding="utf-8")
            if path.suffix.lower() == ".jsonc":
                from .jsonc import loads as jsonc_loads

                payload = jsonc_loads(text)
            else:
                payload = json.loads(text)
        except (OSError, ValueError) as exc:
            raise ValueError(f"模型配置文件不合法: {path}: {exc}") from exc
        entries = payload.get("models") if isinstance(payload, dict) else payload
        if not isinstance(entries, list):
            raise ValueError(f"模型配置文件必须包含 models 数组: {path}")
        self.models = [ModelConfig.model_validate(item) for item in entries]
        return self

    @model_validator(mode="after")
    def _check_model_refs(self) -> "Settings":
        """校验模型 id 唯一，并检查 LLM/RAG 引用的模型存在。"""
        seen: set[str] = set()
        for model in self.models:
            if model.id in seen:
                raise ValueError(f"模型 id 重复: {model.id}")
            seen.add(model.id)
        # 没有配置 MODELS 时允许 RAG/LLM 保留占位 id；一旦配了模型，就校验引用。
        if not seen:
            return self
        refs: list[str] = []
        if self.llm.default_model_id:
            refs.append(self.llm.default_model_id)
        refs.extend(self.llm.model_ids)
        refs.extend(self.llm.fallback_model_ids)
        if self.rag.indexer.embedding_model_id:
            refs.append(self.rag.indexer.embedding_model_id)
        for model_id in (
            self.rag.indexer.description_model_id,
            self.rag.indexer.table_summary_model_id,
            self.rag.indexer.code_summary_model_id,
            self.rag.indexer.graph_model_id,
        ):
            if model_id:
                refs.append(model_id)
        if self.rag.retriever.rerank_model_id:
            refs.append(self.rag.retriever.rerank_model_id)
        if self.rag.retriever.chunk_conflict_model_id:
            refs.append(self.rag.retriever.chunk_conflict_model_id)
        if self.agent.query_plan.model_id:
            refs.append(self.agent.query_plan.model_id)
        if self.agent.reflection.model_id:
            refs.append(self.agent.reflection.model_id)
        if self.memory.extraction_model_id:
            refs.append(self.memory.extraction_model_id)
        if self.memory.compression_model_id:
            refs.append(self.memory.compression_model_id)
        unknown = sorted({item for item in refs if item not in seen})
        if unknown:
            raise ValueError(f"配置引用了未声明的模型 id: {unknown}")
        if (
            self.llm.default_model_id
            and self.llm.model_ids
            and self.llm.default_model_id not in self.llm.model_ids
        ):
            raise ValueError("LLM default_model_id 必须包含在 model_ids 中")
        if self.llm.model_ids:
            outside = sorted(set(self.llm.fallback_model_ids) - set(self.llm.model_ids))
            if outside:
                raise ValueError(f"LLM fallback 模型不在 model_ids 中: {outside}")
        return self


# 全局单例：启动时读一次 .env；测试可显式构造 Settings(_env_file=...)
settings = Settings()
