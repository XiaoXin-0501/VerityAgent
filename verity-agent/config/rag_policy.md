# RAG 索引策略配置

`rag_policy.json` 与 `IndexerConfig` 一起放在 `config` 包，集中管理索引阶段
会随领域/部署调整的**策略数据**，不放进业务代码：

- `quality.penalties`：`QualityFlag → 扣分`；
- `description.reliable_headings`：可直接作为文档描述的“文档级”标题词表；
- `association.lead_in_hints / follow_up_hints`：代码/表格与相邻描述文本的关联提示词；
- `graph.generic_stopwords`：通用关键词停用词；
- `staging.stage_progress`：`IndexingStage → 进度`，只用于 job view / PG 投影。

由 `rag/indexer/policy.py` 加载并校验，加载期 fail-fast；运行期只读，
修改后需重启进程。默认读取本文件；可通过
`IndexerConfig.indexer_policy_path`（env：`RAG__INDEXER__INDEXER_POLICY_PATH`）
指向外部策略文件，相对路径按 `verity-agent` 解析。

## quality.penalties

```text
quality_score = clamp(1.0 - Σ penalty(flag), 0.0, 1.0)
```

- 多个标记叠加扣分，不取最大值；
- 每个值必须在 `0~1`；
- 未声明的 `QualityFlag` 按 0 分处理，向前兼容新增枚举；
- 只做检索降权，不决定是否丢弃（丢弃由 QualityGate / DedupPolicy 决定）。

| 标记 | 默认 | 说明 |
|---|---:|---|
| `LOW_OCR` | 0.25 | OCR 置信度低于 `quality.min_ocr_confidence` |
| `LOW_TABLE` | 0.2 | 表格识别置信度低于 `quality.min_table_confidence` |
| `LOW_TEXT_COVERAGE` | 0.3 | 文本层覆盖率低（扫描件/图片为主） |
| `DUPLICATE` | 0.5 | 重复/近重复；默认已丢弃，保留时降权 |
| `TRUNCATED` | 0.1 | 内容被真正截断（非正常切片） |
| `BOILERPLATE` | 0.5 | 页眉/页脚/导航/广告；默认已丢弃 |
| `CODE_PARSE_FALLBACK` | 0.1 | 代码 AST 失败，退化为文本切分 |
| `EMBEDDING_TRUNCATED` | 0.05 | 仅 embedding 输入被截断，正文完整 |

> 当前实际写入 chunk flag 的只有 `LOW_OCR / LOW_TABLE / CODE_PARSE_FALLBACK`；
> 其余标记留给后续 processor/降级策略保留内容时使用。

## description.reliable_headings

- 只有这些“文档级”标题才允许直接取后续正文作为文档描述；
- 加载时统一归一化（去空白/常见标点、英文小写），因此写 `Introduction`
  或 `introduction` 等价；
- “企业简介/课程简介”等仍走候选逻辑，不直接采信；
- 不要放空字符串。

## association.lead_in_hints / follow_up_hints

- 前导说明：紧贴代码/表格、通常以“如下/例如/下表”等词结尾的短文本；
- 后续说明：紧贴代码/表格、通常以“上述/该/注意”等词开头的短文本；
- 命中后会把该短文本吸附到特殊块，作为其上下文一并切片；
- 没有命中提示词时，仍有“短文本紧邻”的弱关联兜底
  （`special_context_short_tokens`）。

## graph.generic_stopwords

- 通用关键词停用词（中英文），不进入图关键词；
- 领域停用词仍由 `DomainProfile.stopwords` 单独维护；
- 两者在 `DomainGraphExtractor` 中合并使用。

## staging.stage_progress

- 合法 key 必须是 `IndexingStage` 枚举值，value 在 `0~1`；
- 缺失阶段由流水线保留当前进度；
- 该映射不参与业务判断，只影响任务进度展示；
- 删除流程阶段（`DELETING_*`）也在此维护。

## 覆盖方式

- 直接修改本文件（适合当前仓库）；
- 设置 `IndexerConfig.indexer_policy_path` 指向外部文件（相对路径按
  `verity-agent` 解析）；
- 代码中显式传入 `IndexerPolicy` 给 `build_default_indexing_pipeline(...,
  indexer_policy=...)`，或 `load_indexer_policy(path)` 后注入；
- `load_indexer_policy(path)` 把外部文件当作**完整策略**加载：未写的 section
  使用模型默认值（空扣分/空词表/空进度），不会自动与内置合并；
- 单测覆盖：`tests/test_rag_indexer_policy.py`。
