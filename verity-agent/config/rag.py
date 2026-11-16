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
    # 相邻 tile 重叠像素；避免长图在 tile 边界切断行
    tile_overlap: int = Field(default=140, ge=0)
    # 单文档最多处理页数；超出部分记 warning
    max_pages: int = Field(default=100, ge=1)
    # 单页最多 tile 数，防止极端长图
    max_tiles_per_page: int = Field(default=24, ge=1)
    # OCR 置信度阈值；低于仍保留，但标记 low_confidence
    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    # MinerU 命令行；mini 安装可用 `mineru`，自定义环境可改绝对路径
    mineru_command: str = "mineru"
    # 可选：连接独立 MinerU API 服务，避免主进程加载重型依赖
    mineru_api_url: str | None = None
    # MinerU pipeline 后端（pipeline / vlm 等，取决于安装版本）
    mineru_backend: str = "pipeline"
    # MinerU 解析方法（ocr / txt）
    mineru_method: str = "ocr"
    # MinerU OCR 语言（ch / en / ...）
    mineru_language: str = "ch"
    # 是否启用 MinerU 表格识别
    mineru_table_enabled: bool = True
    # 是否启用 MinerU 公式识别
    mineru_formula_enabled: bool = False
    # MinerU 单文档超时（秒）
    mineru_timeout_s: int = Field(default=900, ge=30)
    # MinerU 模型源：modelscope / huggingface / auto
    mineru_model_source: str = "modelscope"
    # 预留 LLM 后处理；默认关闭
    llm_enhance_enabled: bool = False
    # LLM 后处理模型 id；llm_enhance_enabled=true 时使用
    llm_model_id: str | None = None
    # LLM 后处理要执行的 processor 名称；空表示全部
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
    # chunk 最小 token
    min_chunk_tokens: int = Field(default=50, ge=0)
    # 表格置信度下限
    min_table_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    # OCR 置信度下限
    min_ocr_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    # 是否丢弃低质量 chunk；默认保留并标记
    drop_low_quality_chunks: bool = False


class DedupPolicyConfig(ConfigModel):
    """chunk 内容去重策略；文档/块级去重留给上传/解析入口。"""

    # chunk 级内容 hash 去重
    chunk_hash: bool = True


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

    # 用于入库的 embedding 应用模型 id
    embedding_model_id: str | None = None
    # 向量维度；必须与 Weaviate collection 对齐
    embedding_dimensions: int = Field(default=1024, gt=0)
    # embedding 版本；换模型/维度时递增
    embedding_version: str = "v1"
    # 单批 embedding 文本数；服务会再按模型 max_batch_size 收敛
    embedding_batch_size: int = Field(default=20, ge=1)
    # 是否缓存 chunk/query 向量；命中后重试/重灌不再调用 embedding 模型
    embedding_cache_enabled: bool = True
    # 向量缓存 TTL（秒）；默认 30 天，模型/维度变化通过 key 版本失效
    embedding_cache_ttl_s: int = Field(default=2_592_000, ge=0)

    # 文档描述生成模型 id
    description_model_id: str | None = None
    # 是否启用文档描述提取
    description_enabled: bool = True
    # 非可靠“内容提取”候选或“标题+摘要/前言”过长时交给 LLM；
    # 需要同时注入 description chat model（bootstrap 按 description_model_id 装配）。
    description_llm_enabled: bool = True
    # 描述最小/最大字符数；标题+摘要/前言超过上限时优先 LLM 压缩
    description_min_chars: int = Field(default=40, ge=0)
    # 文档描述最大字符数；标题+摘要超过时 LLM 压缩/按句截断
    description_max_chars: int = Field(default=500, ge=1)
    # 无可信描述时是否允许使用可读文件名作为描述
    description_use_filename: bool = True
    # 扫描描述规则的早期文本块数；目录/作者信息可能占几十块，需覆盖摘要页
    description_max_early_blocks: int = Field(default=80, ge=1)
    # 是否缓存 LLM 生成的 chunk/文档描述，避免重试重复消耗 token
    description_cache_enabled: bool = True
    # 描述缓存 TTL（秒）；模型/Prompt 版本变化通过 key 失效
    description_cache_ttl_s: int = Field(default=2_592_000, ge=0)
    # 扫描件/图片 PDF 的 OCR 配置
    ocr: OcrConfig = Field(default_factory=OcrConfig)
    # 纯图片/扫描 PDF 的版式解析（MinerU/RapidOCR）
    image_pdf: ImagePdfConfig = Field(default_factory=ImagePdfConfig)
    # 表格 LLM 摘要模型 id
    table_summary_model_id: str | None = None
    # 代码 LLM 摘要模型 id
    code_summary_model_id: str | None = None
    # 图抽取 LLM 模型 id（mode=llm/hybrid 时使用）
    graph_model_id: str | None = None
    # 是否启用大表/歧义表的 LLM 摘要
    table_summary_llm_enabled: bool = False
    # 是否启用代码 LLM 摘要；默认只用签名/docstring
    code_summary_enabled: bool = False
    # 特殊块（表格/代码）向量化策略：只向量化 heading + 描述，避免低语义正文
    semantic_embedding_enabled: bool = True
    # heading 少于该 token 数视为“太少”，无描述时才触发 LLM 兜底
    semantic_heading_min_tokens: int = Field(default=4, ge=0)
    # 表格/Excel 送入 LLM 的预览行数
    semantic_preview_rows: int = Field(default=3, ge=1)
    # 生成的描述最大字符数
    semantic_description_max_chars: int = Field(default=200, ge=20)
    # 代码送入 LLM 的最大字符数
    semantic_code_max_input_chars: int = Field(default=2000, ge=100)
    # 描述生成最大输出 token
    semantic_description_max_output_tokens: int = Field(default=160, ge=32)
    # 描述生成温度
    semantic_description_temperature: float = Field(
        default=0.0, ge=0.0, le=2.0
    )
    # 入库是否建图；v1 默认 true（D-56）
    graph_build_enabled: bool = True
    # 垂直领域画像名；一个部署只接入一个领域（默认宠物医疗参考画像）
    domain_profile: str = "pet_medical"
    # 自定义领域画像目录；为空时使用内置 domains/ 目录
    domain_profile_dir: str | None = None
    # 自定义领域画像文件；设置后优先于 domain_profile/domain_profile_dir
    domain_profile_path: str | None = None
    # 图抽取模式：
    # rule=纯通用规则；lexicon=领域词典 + 通用词；llm=LLM（失败降级词典）；
    # hybrid=LLM + 词典合并；off=关闭
    graph_extraction_mode: str = "lexicon"
    # 领域词之外是否补充通用名词关键词
    graph_include_generic_terms: bool = True
    # 每个 chunk 最多写入多少关键词
    graph_max_keywords_per_chunk: int = Field(default=10, ge=1)
    # 关系抽取最低置信度
    graph_min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    # 每个 chunk 最多写入多少关系
    graph_max_relations_per_chunk: int = Field(default=20, ge=1)
    # 关系白名单；与领域画像声明的关系取并集
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
    # LLM 图抽取：过短 chunk 不调用模型
    graph_llm_min_chunk_chars: int = Field(default=40, ge=0)
    # LLM 图抽取：单 chunk 最多送入模型的字符数
    graph_llm_max_input_chars: int = Field(default=4000, ge=200)
    # LLM 图抽取温度；抽取任务建议低温
    graph_llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    # LLM 图抽取最大输出 token
    graph_llm_max_output_tokens: int = Field(default=800, ge=64)
    # LLM 未返回 confidence 时的兜底置信度
    graph_llm_default_confidence: float = Field(default=0.8, ge=0.0, le=1.0)

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

    # 是否启用入库降噪质量闸门
    quality_gate_enabled: bool = True
    # 质量阈值
    quality: QualityPolicyConfig = Field(default_factory=QualityPolicyConfig)
    # 去重策略
    dedup: DedupPolicyConfig = Field(default_factory=DedupPolicyConfig)

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
    # RAG 上下文最大 token；调用方未传 token_budget 时的默认预算
    max_context_tokens: int = Field(default=3000, gt=0)
    # 权限粒度：KB_ONLY / KB_AND_DOC
    permission_mode: str = "KB_AND_DOC"
    # 最终 top-k 确定后是否按 doc_id 批量补文档描述（D-103）
    include_doc_meta: bool = True
    # 预留：是否注册/暴露 resolve_chunk_conflict 工具（D-88）
    chunk_conflict_tool_enabled: bool = False

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

    # 预留：是否启用 query 改写（D-81，当前未接入 LLM 改写）
    query_rewrite_enabled: bool = False
    # query 改写应用模型 id；为空使用 LLM 默认模型
    query_rewrite_model_id: str | None = None
    # 最多生成几个变体
    query_rewrite_max_variants: int = Field(default=2, ge=1)
    # query 改写超时（秒）
    query_rewrite_timeout_s: float = Field(default=20.0, gt=0)
    # 过短 query 不改写
    query_rewrite_min_query_tokens: int = Field(default=3, ge=1)

    # 是否启用规则意图识别（lookup/compare/summarize/howto）
    intent_rule_enabled: bool = True
    # 规则无法判断时是否调用 LLM 兜底分类
    intent_llm_enabled: bool = True
    # 意图分类应用模型 id；为空使用默认 chat 模型
    intent_model_id: str | None = None
    # 意图分类 LLM 超时（秒）；超时保留规则结果
    intent_timeout_s: float = Field(default=5.0, gt=0)

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

    # 是否启用检索结果缓存
    cache_enabled: bool = True
    # 检索结果缓存 TTL（秒）；0 表示不缓存
    cache_ttl_s: int = Field(default=300, ge=0)
    # query embedding 缓存 TTL（秒）；0 表示不缓存查询向量
    query_embedding_cache_ttl_s: int = Field(default=2_592_000, ge=0)
    # 文档状态缓存 TTL（秒）；按 doc_id 批量缓存 completed 状态
    doc_status_cache_ttl_s: int = Field(default=60, ge=0)
    # 文档描述/meta 缓存 TTL（秒）；按 doc_id + 索引版本缓存
    doc_meta_cache_ttl_s: int = Field(default=300, ge=0)


class RagConfig(ConfigModel):
    """RAG 总配置。"""

    # 入库配置
    indexer: IndexerConfig = Field(default_factory=IndexerConfig)
    # 检索配置
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    # 单条 query 最大字符数；超过截断并写 warning
    max_query_chars: int = Field(default=2000, gt=0)
    # 新建知识库未显式指定权限码时使用的默认值
    default_kb_permission_code: str = Field(
        default="kb:read", min_length=1, max_length=100
    )
