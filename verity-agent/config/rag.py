"""RAG 配置（Indexer / Retriever / 质量与去重）。

所有默认值在这里维护；RAG 模块只消费配置对象，不重复写死。
"""

from __future__ import annotations

from pydantic import Field

from .base import ConfigModel

__all__ = [
    "RagConfig",
    "IndexerConfig",
    "RetrieverConfig",
    "QualityPolicyConfig",
    "DedupPolicyConfig",
    "OcrConfig",
    "ImagePdfConfig",
]


class ImagePdfConfig(ConfigModel):
    """纯图片/扫描 PDF 的版式解析配置。

    默认使用 MinerU pipeline（非 LLM）；不可用时降级到 RapidOCR。
    LLM 相关开关默认关闭，为后续后处理预留。
    """

    # 是否启用图片 PDF 版式解析
    enabled: bool = True
    # auto / mineru / rapid / simple / none
    engine: str = "auto"
    # 页面渲染 DPI
    dpi: int = Field(default=300, ge=120, le=600)
    # 超长页判定：页面高宽比超过该值则切 tile
    long_page_ratio: float = Field(default=8.0, ge=2.0)
    # tile 高度/重叠（像素，渲染后）
    tile_height: int = Field(default=1400, ge=400)
    tile_overlap: int = Field(default=140, ge=0)
    # 单文档最多处理页数；超出部分记 warning
    max_pages: int = Field(default=100, ge=1)
    # 单页最多 tile 数，防止极端长图
    max_tiles_per_page: int = Field(default=24, ge=1)
    # OCR 置信度阈值；低于仍保留，但标记 low_confidence
    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    # MinerU CLI 命令与后端
    mineru_command: str = "mineru"
    # 可选：连接独立的 MinerU API 服务，避免主进程加载重型依赖
    mineru_api_url: str | None = None
    mineru_backend: str = "pipeline"
    mineru_method: str = "ocr"
    mineru_language: str = "ch"
    # 是否启用表格/公式识别
    mineru_table_enabled: bool = True
    mineru_formula_enabled: bool = False
    mineru_timeout_s: int = Field(default=900, ge=30)
    # 模型源：modelscope / huggingface / auto
    mineru_model_source: str = "modelscope"
    # 预留 LLM 后处理；默认关闭
    llm_enhance_enabled: bool = False
    llm_model_id: str | None = None
    llm_processors: list[str] = Field(default_factory=list)


class OcrConfig(ConfigModel):
    """扫描件/图片 PDF 的 OCR 配置。

    默认 ``auto``：优先 PaddleOCR，其次 Tesseract；两者都不可用时，
    文本层 PDF 仍可正常解析，只有真正需要 OCR 的页面才会报错。
    """

    # 是否允许对无文本层页面执行 OCR
    enabled: bool = True
    # auto / paddle / tesseract / none
    provider: str = "auto"
    # Tesseract 语言包；需要安装 chi_sim + eng
    language: str = "chi_sim+eng"
    # PaddleOCR 语言：ch / en / ...
    paddle_lang: str = "ch"
    # 页面渲染 DPI；越高越慢
    dpi: int = Field(default=200, ge=72, le=600)
    # 低于该置信度的 OCR 文本仍保留，但标记 LOW_OCR
    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    # 单文档最多 OCR 页数，防止超大扫描件无限耗时
    max_pages: int = Field(default=100, ge=1)


class QualityPolicyConfig(ConfigModel):
    """入库质量闸门。"""

    # 有文本页占比低于此值 → 拒绝或 OCR
    min_text_coverage: float = Field(default=0.3, ge=0.0, le=1.0)
    # OCR 文本占比过高 → 标记
    max_ocr_ratio: float = Field(default=0.9, ge=0.0, le=1.0)
    # 语言检测最低置信度
    min_language_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    # 单页最小有效 token
    min_page_tokens: int = Field(default=20, ge=0)
    # 块最小 token
    min_block_tokens: int = Field(default=10, ge=0)
    # chunk 最小 token
    min_chunk_tokens: int = Field(default=50, ge=0)
    # 页眉页脚/导航占比上限
    max_boilerplate_ratio: float = Field(default=0.3, ge=0.0, le=1.0)
    # 近重复阈值
    max_duplicate_similarity: float = Field(default=0.95, ge=0.0, le=1.0)
    # 表格置信度下限
    min_table_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    # OCR 置信度下限
    min_ocr_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    # 是否允许代码解析 fallback
    code_parse_fallback_allowed: bool = True
    # 是否丢弃低质量 chunk；默认保留并标记
    drop_low_quality_chunks: bool = False


class DedupPolicyConfig(ConfigModel):
    """文档/块/chunk 去重策略。"""

    # 是否按 sha256 做文档级精确去重
    document_sha256: bool = True
    # 是否做文档级近重复检测
    near_duplicate: bool = True
    # 块级内容 hash 去重
    block_hash: bool = True
    # chunk 级内容 hash 去重
    chunk_hash: bool = True
    # 近重复阈值
    similarity_threshold: float = Field(default=0.95, ge=0.0, le=1.0)


class IndexerConfig(ConfigModel):
    """入库配置；字段与《RAG 模块数据结构设计》§12.1 对齐。"""

    # 启用的文档类型；v1 默认只启用这五种，空列表表示全部已注册类型
    enabled_doc_types: list[str] = Field(
        default_factory=lambda: [
            "pdf",
            "docx",
            "markdown",
            "text",
            "spreadsheet",
        ]
    )
    # 默认 profile 名（解析/切片组合）
    default_profile: str = "default"
    # 默认切片 profile 名
    chunking_profile: str = "default"

    # 用于入库的 embedding 应用模型 id
    embedding_model_id: str | None = None
    # 向量维度；必须与 Weaviate collection 对齐
    embedding_dimensions: int = Field(default=1024, gt=0)
    # embedding 版本；换模型/维度时递增
    embedding_version: str = "v1"
    # 单批 embedding 文本数
    embedding_batch_size: int = Field(default=32, ge=1)

    # 文档描述生成模型 id
    description_model_id: str | None = None
    # 是否启用文档描述提取
    description_enabled: bool = True
    # 非可靠“内容提取”候选是否交给 LLM；文件名/强规则/空提取不调用
    description_llm_enabled: bool = False
    # 描述最小/最大字符数；可靠规则命中必须落在此范围
    description_min_chars: int = Field(default=40, ge=0)
    description_max_chars: int = Field(default=500, ge=1)
    # 无可信描述时是否允许使用可读文件名作为描述
    description_use_filename: bool = True
    # 扫描描述规则时只看前 N 个内容块
    description_max_early_blocks: int = Field(default=20, ge=1)
    # 扫描件/图片 PDF 的 OCR 配置
    ocr: OcrConfig = Field(default_factory=OcrConfig)
    # 纯图片/扫描 PDF 的版式解析（MinerU/RapidOCR）
    image_pdf: ImagePdfConfig = Field(default_factory=ImagePdfConfig)
    # 表格 LLM 摘要模型 id
    table_summary_model_id: str | None = None
    # 代码 LLM 摘要模型 id
    code_summary_model_id: str | None = None
    # 图抽取 LLM 模型 id
    graph_model_id: str | None = None
    # 是否启用大表/歧义表的 LLM 摘要
    table_summary_llm_enabled: bool = False
    # 是否启用代码 LLM 摘要；默认只用签名/docstring
    code_summary_enabled: bool = False
    # 入库是否建图；v1 默认 true（D-56）
    graph_build_enabled: bool = True
    # 图关键词抽取模式：rule / off（llm 预留）
    graph_extraction_mode: str = "rule"
    # 每个 chunk 最多写入多少关键词
    graph_max_keywords_per_chunk: int = Field(default=10, ge=1)
    # 关系抽取最低置信度；规则抽取暂不使用
    graph_min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    # 每个 chunk 最多写入多少关系
    graph_max_relations_per_chunk: int = Field(default=20, ge=1)
    # 关系白名单；非白名单关系不写入图
    graph_relation_whitelist: list[str] = Field(
        default_factory=lambda: [
            "is_a",
            "part_of",
            "depends_on",
            "implements",
            "supports",
            "requires",
            "causes",
            "contrasts",
        ]
    )

    # 相邻的代码/表格描述最大 token 数；超过则视为独立正文
    special_context_max_tokens: int = Field(default=80, ge=0)
    # 没有明确提示词（如“如下/上述”）时，仍可吸附的短文本上限
    special_context_short_tokens: int = Field(default=40, ge=0)
    # 是否把紧贴代码/表格的短文本并入特殊块
    special_context_enabled: bool = True

    # 目标 chunk token 大小
    chunk_size_tokens: int = Field(default=300, ge=64)
    # chunk 硬上限 token
    max_chunk_tokens: int = Field(default=600, ge=64)
    # chunk 最小 token；低于合并/丢弃
    min_chunk_tokens: int = Field(default=40, ge=0)
    # 相邻文本 chunk 重叠 token
    chunk_overlap_tokens: int = Field(default=80, ge=0)
    # 大表按行组切分时每组的数据行数（不含重复表头）
    table_row_group_size: int = Field(default=20, ge=1)
    # 切片策略版本；影响确定性 chunk_id
    chunking_version: str = "v1"
    # 确定性 chunk_id 的命名空间前缀
    chunk_id_namespace: str = "verity:chunk"
    # 低于该字符数的块丢弃
    min_chunk_chars: int = Field(default=30, ge=0)
    # 超过该字符数的块截断/拆分
    max_chunk_chars: int = Field(default=8_000, ge=1)

    # 是否启用入库降噪质量闸门
    quality_gate_enabled: bool = True
    # 质量阈值
    quality: QualityPolicyConfig = Field(default_factory=QualityPolicyConfig)
    # 去重策略
    dedup: DedupPolicyConfig = Field(default_factory=DedupPolicyConfig)

    # 单 worker 并发处理文档数
    max_concurrency: int = Field(default=2, ge=1)
    # 入库最大重试次数
    index_max_attempts: int = Field(default=5, ge=1)
    # 入库 worker 租约时长（秒）
    index_lease_seconds: int = Field(default=300, ge=30)
    # 入库失败退避基数（秒）
    index_retry_backoff_s: float = Field(default=30.0, gt=0)


class RetrieverConfig(ConfigModel):
    """检索配置；字段与《RAG 模块数据结构设计》§12.2 对齐。"""

    # 最终返回 chunk 数
    top_k: int = Field(default=8, ge=1)
    # rerank 后保留数
    rerank_top_k: int = Field(default=8, ge=1)
    # 混合检索向量/BM25 权重（0=纯 BM25，1=纯向量）
    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    # 最终最低分
    min_score: float = 0.0
    # 最低质量分；为空时使用 0
    min_quality_score: float | None = None
    # 上下文 token 预算；None 表示不限制
    token_budget: int | None = Field(default=None, gt=0)

    # 是否启用 rerank
    rerank_enabled: bool = True
    # rerank 应用模型 id
    rerank_model_id: str | None = None
    # rerank 候选数
    rerank_top_n: int = Field(default=20, ge=1)
    # rerank 批大小
    rerank_batch_size: int = Field(default=16, ge=1)
    # rerank 最低分
    rerank_min_score: float | None = None

    # 是否启用 query 改写（D-81）
    query_rewrite_enabled: bool = False
    # query 改写应用模型 id；为空使用 LLM 默认模型
    query_rewrite_model_id: str | None = None
    # 最多生成几个变体
    query_rewrite_max_variants: int = Field(default=2, ge=1)
    # query 改写超时（秒）
    query_rewrite_timeout_s: float = Field(default=20.0, gt=0)
    # 过短 query 不改写
    query_rewrite_min_query_tokens: int = Field(default=3, ge=1)

    # 是否启用图检索扩展；默认关闭（D-55）
    graph_expansion_enabled: bool = False
    # 图扩展最大跳数；v1 只允许 1
    graph_max_hops: int = Field(default=1, ge=1)
    # 允许扩展的关系白名单
    graph_allowed_relations: list[str] = Field(
        default_factory=lambda: [
            "is_a",
            "part_of",
            "depends_on",
            "implements",
            "supports",
            "requires",
            "causes",
            "contrasts",
        ]
    )
    # 关系最小证据数
    graph_min_evidence: int = Field(default=2, ge=1)
    # 最多下一跳实体数
    graph_max_next_hops: int = Field(default=5, ge=1)
    # hub 节点是否可作为中间跳
    graph_hub_as_intermediate: bool = False

    # 是否启用邻居扩展
    neighbors_enabled: bool = True
    # 命中 chunk 前取几个邻居
    neighbor_before: int = Field(default=1, ge=0)
    # 命中 chunk 后取几个邻居
    neighbor_after: int = Field(default=1, ge=0)
    # 邻居总 token 预算
    neighbor_max_tokens: int = Field(default=1000, ge=0)
    # 邻居是否限定同文档
    neighbor_same_doc: bool = True
    # 邻居是否限定同 block_type
    neighbor_same_block_type: bool = True

    # 检索结果缓存 TTL（秒）；0 表示不缓存
    cache_ttl_s: int = Field(default=300, ge=0)
    # 文档状态缓存 TTL（秒）
    doc_status_cache_ttl_s: int = Field(default=60, ge=0)
    # 文档 meta 缓存 TTL（秒）
    doc_meta_cache_ttl_s: int = Field(default=300, ge=0)


class RagConfig(ConfigModel):
    """RAG 总配置。"""

    # 入库配置
    indexer: IndexerConfig = Field(default_factory=IndexerConfig)
    # 检索配置
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    # 文档描述有效最小字符数；低于视为提取失败
    description_min_chars: int = Field(default=40, ge=0)
    # 文档描述最大字符数；超过截断
    description_max_chars: int = Field(default=500, ge=1)
    # 单条 query 最大字符数
    max_query_chars: int = Field(default=2000, gt=0)
