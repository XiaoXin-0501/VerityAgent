# tests 目录说明

本目录包含三类内容：**可独立运行的辅助脚本**、RAG 解析切片的**输入语料与产物**，以及 pytest 自动化测试。业务配置（模型、数据库连接等）统一取自根目录 `.env` 与 `config`，本目录只放测试开关和临时产物。

## 目录结构

```text
tests/
├── .env.test              # 真实依赖测试开关（默认全关，避免意外产生入库 / 删除 / LLM 成本）
├── parse_and_chunk.py     # 脚本：解析文档并切片，导出可检查产物（不入库、默认不调 LLM）
├── analyze_chunks.py      # 脚本：统计 result-data 中的切片质量，输出报告
├── reset_state.py         # 脚本：重置本地 / 测试环境到刚初始化状态（清业务数据，保留表结构）
├── test-data/             # 输入语料：宠物医疗 Markdown / PDF（只读）
├── result-data/           # 脚本产物：每样本一个子目录 + 汇总与分析报告
└── __pycache__/           # 编译缓存，可随时删除
```

## 独立脚本

三个脚本都会自行把 `verity-agent/` 源码根加入 `sys.path`，读取根目录 `.env` 配置，**无需把项目安装成包**，可直接在仓库根目录单独运行：

```sh
uv run python tests/<脚本名>.py
# 或使用虚拟环境解释器
.venv/Scripts/python tests/<脚本名>.py
```

### `parse_and_chunk.py` —— 解析与切片（不入库）

把 `test-data/` 下的文档走一遍「解析 → 质量闸门 → 切片 → 去重」管线，导出可逐块检查的产物；不写 PG / 向量库 / 图库，默认不触发 LLM 描述。

```sh
# 处理 test-data 下全部支持的格式（PDF / DOCX / Markdown / TXT / XLSX）
uv run python tests/parse_and_chunk.py

# 只处理指定文件 / 目录；只列文件不解析；限制数量
uv run python tests/parse_and_chunk.py tests/test-data/某文档.pdf
uv run python tests/parse_and_chunk.py --list
uv run python tests/parse_and_chunk.py --limit 3

# 关闭版式解析 / OCR；关闭质量闸门 / 去重；指定输入环境变量
uv run python tests/parse_and_chunk.py --image-engine none --ocr-provider none
uv run python tests/parse_and_chunk.py --no-quality-gate --no-dedup
RAG_TEST_PATH=tests/test-data/某文档.md uv run python tests/parse_and_chunk.py
```

| 参数 | 说明 |
|---|---|
| `paths`（位置参数） | 输入文件或目录；默认 `test-data/`，也可用 `RAG_TEST_PATH` 指定 |
| `--out` | 产物根目录，默认 `tests/result-data` |
| `--list` | 只列出待处理文件，不解析 |
| `--limit N` | 最多处理前 N 个文件，`0` 为全部 |
| `--image-engine` | 图片 / 扫描 PDF 版式引擎：`auto / mineru / rapid / simple / none` |
| `--ocr-provider` | OCR：`auto / rapid / paddle / tesseract / none` |
| `--no-quality-gate` | 关闭质量闸门，保留低质量 / 过短块 |
| `--no-dedup` | 关闭 chunk 哈希去重 |
| `--workers` | 并发文档数；OCR / MinerU 建议保持 `1` |

每个文档输出到 `result-data/<样本名>/`：`manifest.json`（统计 / profile / 描述 / 耗时 / sha256）、`parse.json`（文档投影）、`blocks.jsonl`（内容块）、`chunks.jsonl`（切片）、`chunks.md`（可读切片）；顶层汇总 `result-data/parse_chunk_index.json`。

### `analyze_chunks.py` —— 切片质量分析

前置：先跑完 `parse_and_chunk.py`，确保 `result-data/parse_chunk_index.json` 存在。纯本地读产物做统计，不连外部服务。

```sh
uv run python tests/analyze_chunks.py
uv run python tests/analyze_chunks.py --result-dir tests/result-data --min-tokens 40 --max-tokens 600
```

输出 `result-data/chunk_analysis.md`（可读报告）与 `chunk_analysis.json`（机读明细）；统计 chunk 数 / token 总量、过短块、超长块、token 分桶与块类型分布，阈值缺省时取汇总文件中的切片配置。

### `reset_state.py` —— 环境重置

把本地 / 测试环境清空并恢复到刚初始化状态，便于重复跑入库与检索验证。**运行前先停止后端**，避免 worker 继续写入。

```sh
uv run python tests/reset_state.py
```

清空范围：

- **PostgreSQL**：用户、权限、知识库、文档、会话、消息、摘要、长期记忆、消息队列、会话文件；清空后重建默认账号 `xiaoxin`（admin）与 `xiaobai`（只读用户）。
- **Weaviate**：删除并重建 `DocChunk` / `UserFact` collection。
- **Neo4j**：删除全部节点与关系。
- **MinIO**：清空知识库 bucket 与会话 bucket 对象。
- **Redis**：清空当前 namespace 下的缓存与 run 状态。

安全约束：`APP__ENV=prod` 时拒绝执行；只清业务数据，不删表结构、bucket 或数据库。

## 测试数据与产物

- `test-data/`：只读输入语料（Markdown 疾病 / 兽药分册 + 中英文 PDF 病例报告、规范与指南）。
- `result-data/`：脚本生成的产物，可整体删除后重跑，不影响源码与线上数据。

## pytest 自动化测试

配置来自根目录 `pyproject.toml`（`testpaths=["tests"]`、`pythonpath=["verity-agent"]`，`integration` 标记）。

```sh
# 离线用例：不连外部服务、不产生 LLM 成本
uv run pytest

# 指定文件 / 用例
uv run pytest tests/test_xxx.py -k keyword

# 集成用例：需真实基础设施，且在命令行显式打开 RAG_REAL_* 开关
uv run pytest -m integration -s
```

- 离线用例在内存 / mock 态验证逻辑；集成用例需要 PG / Redis / Weaviate / Neo4j / MinIO 与模型密钥，可能产生真实写入与 token 成本。
- 真实依赖开关集中在 `.env.test`，默认全部关闭：`RAG_REAL_INGEST`（入库）、`RAG_REAL_RETRIEVAL`（召回）、`RAG_REAL_CLEANUP`（清理）、`RAG_REAL_IMAGE_PDF`（图片 PDF）；端到端 / 长对话等可选开关见该文件注释。

> 当前工作树中 `test_*.py` 源码未随目录保留，仅 `__pycache__/` 下留有历史编译缓存；恢复源码后按业务域（认证、Agent、记忆、工具、RAG、Worker 等）组织即可。

## 备注

- `result-data/`、`__pycache__/` 属运行时产物，已在 `.gitignore` 忽略范围内。
- 本目录不存放连接信息与密钥；结构化策略在 `config/` 下的 JSON 配置。
