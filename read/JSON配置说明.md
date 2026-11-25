# verity-agent JSON 配置说明

> 本文档统一说明 `verity-agent` 模块下所有 JSON 配置的结构、字段、默认值、
> 约束和加载方式。
>
> **格式约定**：本文档覆盖的所有 `.json` 配置均按**标准 JSON（UTF-8）**维护，
> 文档示例不使用 JSONC 注释或尾逗号；字段说明统一放在本文档中。配置在启动期
> 加载并校验，修改后需要重启进程（Prompt 注册表等对象是运行期只读的）。

## 0. 文件总览

| 文件 | 作用 | 加载/校验入口 | 允许热更新 |
|---|---|---|---|
| `agent/tools/manifest.json` | 工具声明表：工具名、描述、权限、暴露条件、实现工厂、横切策略差异 | `agent/tools/manifest.py`、`agent/tools/bootstrap.py` | 否 |
| `config/tools/policies.default.json` | 所有工具横切能力的全局默认值（超时/重试/缓存/熔断/限流/幂等/结果上限/并发） | `config/tools/models.py` | 否 |
| `config/models.json` | 模型目录：provider、模型名、连接信息、能力、价格、默认生成参数、thinking 配置 | `config/__init__.py`、`provider/model_specs.py` | 否 |
| `config/domains/*.json` | 垂直领域画像：实体类型、关系、别名、白黑名单、噪声词、抽取提示词 | `config/domain.py` | 否（换画像需重新入库图） |
| `prompt/templates/manifest.json` | Prompt 模板元数据：文件路径、版本、用途、变量声明 | `prompt/registry.py` | 否 |

补充说明：

- **配置边界**：env 只放运行环境、连接信息、密钥与部署差异；业务默认值在
  `config/*.py`；结构化配置在 JSON（本文件覆盖的 manifest/模型/画像/策略）；
- `models.json` 默认路径是 `config/models.json`（`Settings.models_file`）；
  换文件时才用 `MODELS_FILE` 覆盖；不再支持在 env 内联 `MODELS` 数组；
- `domains/` 可以由 `RAG__INDEXER__DOMAIN_PROFILE_DIR` /
  `RAG__INDEXER__DOMAIN_PROFILE_PATH` 指向自定义目录或文件；
- `prompt/templates/` 可以由 `PROMPTS__DIRECTORY` 指向外部覆盖目录；
- 除明确标注为自由对象的 `metadata`、`provider_options` 等字段外，配置模型
  默认 `extra="forbid"`：**未知字段会直接导致启动失败**，不会静默忽略；
  `capabilities` 虽然以对象写入，但转换时会按强类型能力模型校验，未知键同样
  会失败。

---

## 1. `agent/tools/manifest.json`：工具声明

### 1.1 顶层结构

```json
{
  "version": 1,
  "tools": []
}
```

| 字段 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `version` | int | 否 | `1` | manifest 结构版本；当前只允许 `1`。 |
| `tools` | object[] | 否 | `[]` | 工具声明数组；每个元素见 §1.2。工具名必须全局唯一。 |

### 1.2 单个工具声明字段

| 字段 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `name` | string | 是 | — | LLM 调用名，1~64 字符，匹配 `^[a-zA-Z0-9_-]{1,64}$`；同一 manifest 内唯一。 |
| `description` | string | 是 | — | 工具用途、适用场景、返回内容和边界；直接下发给 LLM。 |
| `group` | string\|null | 否 | `null` | 工具分组标签，如 `kb`、`memory`；用于管理/路由扩展。没有注册 `ToolGroup` 时只是标签，不影响执行。 |
| `required_permission` | string\|null | 否 | `null` | 静态能力码，如 `tool:kb:search`。`null` 表示登录用户可用。启动期按权限定义过滤注册，请求期再按用户实际权限过滤。 |
| `exposure` | object | 否 | `{"kind":"always"}` | 单次请求的暴露条件；见 §1.3。注册表仍保留工具，只改变下发给 LLM 的 `ToolSelection`。 |
| `side_effect` | string | 否 | `"READ"` | 副作用等级：`READ` / `WRITE` / `DESTRUCTIVE`。v1 只自动执行 `READ`；写工具还需要确认/对账能力，否则会被流水线拒绝。 |
| `tags` | string[] | 否 | `[]` | 检索/管理标签；当前只随 `ToolSpec` 保存，不参与运行期过滤。 |
| `required_services` | string[] | 否 | `[]` | 装配期必需的服务 key；每个值必须是 `ToolServiceKey` 的成员：`rag_retriever`、`chunk_conflict_service`、`conversation_messages`。新增服务需要先在 `agent/tools/services.py` 增加枚举成员，否则启动期校验失败。缺失时 bootstrap 输出具体服务名并跳过注册。 |
| `implementation` | string | 是 | — | 工厂路径，格式 `模块路径:工厂类名`，例如 `agent.tools.builtin.search_kb:SearchKbFactory`。启动期动态 import；工厂必须提供 `build(declaration, ctx)`，返回 `RegisteredTool` 或 `None`。 |
| `metadata` | object | 否 | `{}` | 内部扩展字段，不发给 LLM。已识别的键见 §1.5。 |

工具参数 schema 不在 manifest 中声明：工厂类通过 `argument_model` 提供 Pydantic
参数模型，注册表据此生成 LLM JSON Schema；`strict`、`version`、确认/对账等
`ToolSpec` 字段由代码默认值或工厂决定。

### 1.3 `exposure`：请求期暴露条件

`exposure` 是 `kind + 参数` 的可扩展对象。启动期会校验 `kind` 和已知参数，
未知 `kind` 直接失败；请求期由 `agent/tools/exposure.py` 的解释器注册表执行。
暴露条件只影响本次请求的 `ToolSelection`，不修改全局注册表。

| `kind` | 参数 | 说明 |
|---|---|---|
| `always` | 无 | 无条件暴露。 |
| `rag_required` | 无 | 查询规划判定本次需要 RAG 时暴露；只看规划，不看实际召回结果。 |
| `rag_retrieved` | `min_chunks`（int，`>=1`，默认 `1`） | 本次请求实际获取的 chunk 数达到阈值时暴露。计数包含 INJECTED 模式注入的 chunk 和 `search_kb` 返回的 chunk，并在工具调用后累计刷新。冲突工具当前配置为 `min_chunks=2`。 |
| `after_compressions` | `min_compressions`（int，`>=0`，默认 `0`） | 当前会话摘要成功压缩次数达到阈值后暴露。 |
| `all_of` | `conditions`（非空数组，元素是暴露条件对象） | 所有子条件同时满足才暴露。 |
| `any_of` | `conditions`（非空数组，元素是暴露条件对象） | 任一子条件满足就暴露。 |

组合条件示例：

```json
{
  "kind": "all_of",
  "conditions": [
    { "kind": "rag_required" },
    { "kind": "after_compressions", "min_compressions": 2 }
  ]
}
```

注意：

- `rag_retrieved` 的计数是“实际召回”而非“计划召回”；
- 执行期流水线还会复核当前 `exposed_tool_names`。即使模型编造了一个当前未
  暴露的工具调用，也会返回 `TOOL_DISABLED`，不会绕过暴露条件；
- 条件对象允许带扩展参数，但解释器只读取上表列出的参数；不要依赖未定义的键。

### 1.4 单工具横切策略覆盖

工具声明可以覆盖 §2 中的横切能力。**只写与全局默认值不同的字段**，装配时与
`policies.default.json` 做递归深合并；省略的字段继承全局默认值。

可覆盖的顶层字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `timeout_s` | number | 单次执行超时秒数，必须 `>0`。 |
| `retry` | object | 重试策略；字段同 §2.1 `retry`。 |
| `cache` | object | 缓存策略；字段同 §2.1 `cache`。 |
| `circuit_breaker` | object | 熔断策略；字段同 §2.1 `circuit_breaker`。 |
| `rate_limit` | object | 限流策略；字段同 §2.1 `rate_limit`。 |
| `idempotency` | object | 幂等策略；字段同 §2.1 `idempotency`。 |
| `result_limit` | object | 结果上限；字段同 §2.1 `result_limit`。 |
| `max_concurrency` | int\|null | 同工具/同用户并发上限，必须 `>=1`；`null` 表示不额外限制。当前批量并发主要由 Agent/Tools 的批次配置控制，该字段为后续单工具并发限制保留。 |

示例（只打开限流、关闭缓存，其余继承默认值）：

```json
{
  "name": "search_kb",
  "cache": { "enabled": false },
  "rate_limit": { "enabled": true }
}
```

### 1.5 `metadata` 已识别键

| 键 | 类型 | 说明 |
|---|---|---|
| `schema_only` | bool | 为 `true` 时允许工具不提供 `argument_model`，直接使用手写 `parameters` schema。 |
| `cache_permission_scoped` | bool | 为 `true` 时缓存 key 额外包含当前用户权限摘要，适合结果受权限过滤影响的工具。 |
| `allow_partial_resources` | bool | 为 `true` 时资源级权限校验只要有一个资源可用就放行，由执行器自行过滤未授权资源。 |

其余键会原样保存在 `ToolSpec.metadata`，供具体工具或扩展逻辑读取，不发给 LLM。

### 1.6 当前内置工具

| 工具 | 暴露条件 | 必需服务 | 关键覆盖 |
|---|---|---|---|
| `search_kb` | `rag_required` | `rag_retriever`（`kb_catalog` 可选，用于省略 `kb_ids` 时解析可读库） | 关闭工具缓存；打开熔断与限流 |
| `recall_round` | `after_compressions(min_compressions=2)` | 无 | 关闭缓存；运行配置可覆盖压缩阈值 |
| `get_tool_result` | `always` | 无（默认构造 `ConversationMessageRepository`） | 关闭缓存；按 `message_id` 回放当前会话的历史工具结果 |
| `resolve_chunk_conflict` | `rag_retrieved(min_chunks=2)` | `chunk_conflict_service` | 继承全局横切默认值 |

新增工具时只需要：

1. 在实现模块提供工厂类（通常继承 `SimpleToolFactory`）；
2. 在 `manifest.json` 增加声明，并把 `implementation` 指向
   `module:FactoryClass`；
3. 如果工厂需要外部服务，在 `required_services` 声明服务 key。

不需要修改 `manifest.py`、`bootstrap.py` 或工具聚合入口。

---

## 2. `config/tools/policies.default.json`：工具横切默认值

### 2.1 字段说明

该文件中的每个字段都是**所有工具共享的默认策略**。单工具只在 manifest 中写
差异字段，装配时递归深合并。

#### `timeout_s`

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `timeout_s` | number | `30` | 单次工具执行超时（秒），必须 `>0`。 |

#### `retry`

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `retry.max_attempts` | int | `2` | 总尝试次数（含首次）；`1` 表示不重试，必须 `>=1`；写工具应在 manifest 覆盖为 1。 |
| `retry.backoff_ms` | int | `200` | 首次退避毫秒数，必须 `>=0`。 |
| `retry.backoff_multiplier` | number | `2.0` | 指数退避倍数，必须 `>=1.0`。 |
| `retry.jitter` | bool | `true` | 是否为退避增加随机抖动。 |
| `retry.retry_on` | string[] | `["TIMEOUT","UPSTREAM_ERROR","RATE_LIMITED"]` | 只对这些错误码重试；其他错误直接返回。可用错误码见 §2.3。 |

#### `cache`

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `cache.backend` | string | `"REDIS"` | 缓存后端；v1 只支持 `REDIS`。 |
| `cache.enabled` | bool | `false` | 是否启用工具结果缓存。非 `READ` 工具不缓存。 |
| `cache.ttl_s` | int | `60` | 缓存存活秒数，必须 `>0`。 |
| `cache.key_fields` | string[]\|null | `null` | 参与缓存 key 的参数名；`null` 表示全部参数。 |
| `cache.exclude_fields` | string[] | `[]` | 不参与 key 的参数名，适合时间戳、nonce 等。 |
| `cache.cache_empty` | bool | `false` | 是否缓存空结果。 |

缓存 key 默认包含 `user_id`、工具版本和稳定参数；如果工具声明了资源权限模板
或 `metadata.cache_permission_scoped=true`，还会包含权限摘要，避免跨用户或
权限变化后命中旧结果。

#### `circuit_breaker`

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `circuit_breaker.backend` | string | `"REDIS"` | 状态后端；v1 只支持 `REDIS`。 |
| `circuit_breaker.enabled` | bool | `true` | 是否启用熔断。 |
| `circuit_breaker.failure_threshold` | int | `5` | 窗口内失败次数阈值，必须 `>=1`。 |
| `circuit_breaker.failure_rate_threshold` | number\|null | `null` | 失败率阈值，范围 `0~1`；`null` 表示只看失败次数。 |
| `circuit_breaker.window_s` | int | `60` | 统计窗口秒数，必须 `>0`。 |
| `circuit_breaker.open_cooldown_s` | int | `30` | OPEN 状态冷却多少秒后允许 HALF_OPEN 试探，必须 `>0`。 |
| `circuit_breaker.half_open_max_calls` | int | `3` | HALF_OPEN 允许的并发试探数，必须 `>=1`。 |
| `circuit_breaker.success_threshold` | int | `2` | HALF_OPEN 连续成功多少次回到 CLOSED，必须 `>=1`。 |
| `circuit_breaker.count_errors` | string[] | `["TIMEOUT","UPSTREAM_ERROR"]` | 哪些错误码计入失败；可用错误码见 §2.3。 |

#### `rate_limit`

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `rate_limit.backend` | string | `"REDIS"` | 计数后端；v1 只支持 `REDIS`。 |
| `rate_limit.enabled` | bool | `false` | 是否启用限流。 |
| `rate_limit.max_calls` | int\|null | `60` | 窗口内最大调用数，必须 `>=1`；`enabled=true` 时不能为 `null`。 |
| `rate_limit.window_s` | int | `60` | 窗口秒数，必须 `>0`。 |
| `rate_limit.scope` | string | `"TOOL"` | 限流维度：`TOOL` / `USER` / `USER_TOOL` / `GLOBAL`；当前实现按工具维度计数，其他值预留给后续扩展。 |

#### `idempotency`

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `idempotency.backend` | string | `"REDIS"` | 幂等记录后端；v1 只支持 `REDIS`。 |
| `idempotency.mode` | string | `"NONE"` | 幂等模式：`NONE` / `OPTIONAL` / `REQUIRED`。 |
| `idempotency.ttl_s` | int | `86400` | 幂等记录保留秒数，必须 `>0`。 |
| `idempotency.key_fields` | string[]\|null | `null` | 参与幂等 key 的业务参数；`null` 表示全部参数。 |
| `idempotency.exclude_fields` | string[] | `["timestamp","now","nonce","request_id"]` | 不参与 key 的非确定性字段。 |
| `idempotency.store_result` | bool | `true` | 是否保存结果供重复调用重放。 |
| `idempotency.lock_ttl_s` | int | `60` | `IN_PROGRESS` 锁的过期秒数，必须 `>0`。 |
| `idempotency.on_conflict` | string | `"RETURN_EXISTING"` | 冲突处理：`RETURN_EXISTING` / `REJECT` / `WAIT`。 |

#### `result_limit`

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `result_limit.max_content_tokens` | int | `4096` | 回给 LLM 的文本 token 上限，必须 `>=1`。 |
| `result_limit.max_data_tokens` | int | `16384` | 结构化 `data` 的 token 上限，必须 `>=1`。 |
| `result_limit.max_content_bytes` | int | `65536` | 文本字节硬上限，必须 `>=1`。 |
| `result_limit.max_data_bytes` | int | `65536` | 结构化数据字节硬上限，必须 `>=1`。 |
| `result_limit.max_files` | int | `10` | 返回文件数量上限，必须 `>=0`。 |
| `result_limit.max_images` | int | `10` | 返回图片数量上限，必须 `>=0`。 |
| `result_limit.overflow_enabled` | bool | `true` | 超限时是否把完整结果写入 MinIO。 |
| `result_limit.overflow_ttl_s` | int\|null | `null` | 溢出对象 TTL 秒数，必须 `>0`；`null` 表示跟随会话。 |

#### `max_concurrency`

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `max_concurrency` | int\|null | `null` | 同工具/同用户并发上限；`null` 表示不额外限制，非空时必须 `>=1`。当前批量并发主要由 Agent/Tools 的批次配置控制，该字段为后续单工具并发限制保留。 |

### 2.2 合并规则

1. 启动时读取本文件，生成 `ToolsConfig.policy`（进程级缓存）；
2. manifest 中每个工具声明自己的 `timeout_s` / `retry` / `cache` /
   `circuit_breaker` / `rate_limit` / `idempotency` / `result_limit` /
   `max_concurrency`；
3. 只对**显式出现的字段**做递归深合并，未出现的字段继承顶层默认值；
4. 合并结果再转换为强类型 `ToolPolicy`，非法类型/枚举/未知字段在启动期失败。

### 2.3 可用的工具错误码

以下值可用于 `retry_on` 和 `count_errors`：

`INVALID_ARGUMENTS`、`PERMISSION_DENIED`、`TOOL_NOT_FOUND`、`TIMEOUT`、
`CIRCUIT_OPEN`、`RATE_LIMITED`、`IDEMPOTENCY_CONFLICT`、`UPSTREAM_ERROR`、
`RESULT_TOO_LARGE`、`CANCELLED`、`CONFIRMATION_REQUIRED`、`TOOL_DISABLED`、
`INTERNAL_ERROR`。

---

## 3. `config/models.json`：模型目录

### 3.1 顶层结构

推荐格式：

```json
{
  "models": []
}
```

也接受顶层直接是模型数组。实际使用哪个文件由 `Settings.models_file` 决定，
默认 `config/models.json`；env 只允许用 `MODELS_FILE` 覆盖路径，不内联模型数组。

| 字段 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `models` | object[] | 是 | — | 模型配置数组；每个元素的 `id` 必须全局唯一。 |

### 3.2 单个模型字段

| 字段 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `id` | string | 是 | — | 应用侧模型 id，非空且唯一；LLM/RAG/记忆等配置通过它引用模型。 |
| `provider` | string | 是 | — | 厂商/适配器名。允许值按 `kind` 区分，见 §3.3。 |
| `model` | string | 是 | — | 调用厂商 API 时发送的实际模型名。 |
| `kind` | string | 否 | `"chat"` | 模型类型：`chat` / `embedding` / `rerank`。 |
| `api_style` | string\|null | 否 | `null` | chat 的 wire 协议：`openai_chat` / `openai_responses` / `anthropic_messages`；`null` 时按 provider 自动选择。embedding/rerank 不使用该字段。 |
| `base_url` | string | 否 | `""` | API 基地址；每个模型可以指向不同网关。OpenAI SDK 通常写 `.../v1`；Anthropic/Cohere SDK 会自动追加 `/v1`，不要重复。 |
| `api_key_env` | string | 否 | `""` | 存放 API key 的环境变量名；推荐使用。 |
| `api_key` | string | 否 | `""` | 直接密钥；优先于 `api_key_env`，不要提交到仓库。 |
| `timeout_s` | number | 否 | `60` | 单次请求超时秒数，必须 `>0`。 |
| `max_retries` | int | 否 | `2` | SDK transport 重试次数，`0~10`；不包含 LLM 层的模型 fallback。 |
| `headers` | object | 否 | `{}` | 额外 HTTP header；值必须是字符串，不得放密钥。 |
| `enabled` | bool | 否 | `true` | 是否启用；禁用模型仍会注册，但调用返回不可用。 |
| `capabilities` | object | 否 | 见 §3.4 | 模型能力声明；字段按 `kind` 解释。 |
| `pricing` | object\|null | 否 | `null` | 价格信息，供成本统计；字段见 §3.5。 |
| `thinking_enabled` | bool\|null | 否 | `null` | chat thinking 开关：`null` 厂商默认，`false` 显式关闭，`true` 显式开启。 |
| `thinking_options` | object | 否 | `{}` | 开启 thinking 时合并到请求体的厂商参数。 |
| `thinking_disabled_options` | object | 否 | `{}` | 关闭 thinking 时合并到请求体的厂商参数；不同厂商字段名不同。 |
| `defaults` | object | 否 | `{}` | chat 默认生成参数；字段见 §3.6。 |
| `metadata` | object | 否 | `{}` | 厂商/适配器扩展字段；已识别键见 §3.7。 |

补充：如果 `capabilities.thinking=true`、`thinking_enabled=false` 且没有配置
`thinking_disabled_options`，启动日志会给出 warning；此时客户端只能省略
thinking 参数，不能保证厂商真正关闭 thinking。

### 3.3 `provider` 与 `api_style`

| `kind` | 允许的 `provider` | 说明 |
|---|---|---|
| `chat` | `openai`、`anthropic`、`deepseek`、`vllm`、`llama.cpp`、`llamacpp` | `openai` 支持 `openai_chat` / `openai_responses`；`anthropic` 必须 `anthropic_messages`；其余兼容厂商使用 `openai_chat`。未知兼容网关可用 `metadata.variant` 选择具体方言。 |
| `embedding` | `openai`、`bge` | 走 embedding adapter；不使用 `api_style`。 |
| `rerank` | `cohere`、`bge`、`dashscope`、`qwen` | `dashscope` / `qwen` 使用百炼原生 rerank；不使用 `api_style`。 |

如果 `provider` 与 `api_style` 不匹配，启动期抛配置错误。

### 3.4 `capabilities`：按模型类型

#### chat

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `streaming` | bool | `true` | 是否支持流式输出。 |
| `tool_calling` | bool | `true` | 是否支持工具调用。 |
| `parallel_tool_calls` | bool | `true` | 是否支持并行工具调用。 |
| `vision` | bool | `false` | 是否支持图片输入。 |
| `file_input` | bool | `false` | 是否支持原生文件输入。 |
| `thinking` | bool | `false` | 是否支持 thinking/reasoning。 |
| `prefix_completion` | bool | `false` | 是否支持 assistant 前缀续写。 |
| `json_object` | bool | `false` | 是否支持 JSON object 输出。 |
| `json_schema` | bool | `false` | 是否支持 JSON Schema 输出。 |
| `prompt_caching` | bool | `false` | 是否支持 prompt caching。 |
| `max_context_tokens` | int | `8192` | 最大上下文 token，必须 `>=1`。 |
| `max_output_tokens` | int | `4096` | 最大输出 token，必须 `>=1`。 |
| `max_tools` | int\|null | `null` | 单次请求最多工具数；`null` 表示不限制，非空时必须 `>=1`。 |
| `max_images` | int\|null | `null` | 单次请求最多图片数；`null` 表示不限制，非空时必须 `>=1`。 |
| `input_modalities` | string[] | `["text"]` | 输入模态，如 `text` / `image` / `file`。 |
| `output_modalities` | string[] | `["text"]` | 输出模态。 |
| `tokenizer` | string\|null | `null` | tokenizer 族提示，供预算估算使用。 |

#### embedding

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `dimensions` | int | `1024` | 向量维度，必须 `>=1`；必须与 RAG/Weaviate schema 对齐。 |
| `max_batch_size` | int | `64` | 单批最大文本数，必须 `>=1`。 |
| `max_input_tokens` | int | `8192` | 单条文本最大 token，必须 `>=1`。 |
| `supports_dimensions` | bool | `false` | 是否支持请求自定义 `dimensions`。 |
| `supports_normalize` | bool | `true` | 是否支持归一化参数。 |
| `input_type` | string\|null | `null` | query/document 前缀类型；`null` 表示不发送。 |

#### rerank

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `max_documents` | int | `100` | 单次最多文档数，必须 `>=1`。 |
| `max_query_tokens` | int | `512` | query 最大 token，必须 `>=1`。 |
| `max_document_tokens` | int | `4096` | 单文档最大 token，必须 `>=1`。 |
| `max_batch_size` | int | `32` | 单批最大请求数，必须 `>=1`。 |
| `supports_return_documents` | bool | `false` | 是否支持返回原文文档。 |

`capabilities` 中的未知键会在转换为强类型能力模型时报错。

### 3.5 `pricing`

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `input_per_million` | number\|null | `null` | 每百万输入 token 价格。 |
| `output_per_million` | number\|null | `null` | 每百万输出 token 价格。 |
| `cached_input_per_million` | number\|null | `null` | 每百万缓存输入 token 价格；`null` 表示无单独价格。 |
| `per_request` | number\|null | `null` | 按请求计价。 |
| `per_document` | number\|null | `null` | 按文档计价，常用于 rerank。 |
| `currency` | string | `"USD"` | 币种。 |
| `effective_from` | string | 当前时间 | ISO-8601 价格生效时间。 |

### 3.6 `defaults`：chat 默认生成参数

`defaults` 是自由对象，供 chat adapter 合并到每次请求。已识别字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `temperature` | number | 采样温度。 |
| `top_p` | number | nucleus sampling。 |
| `max_output_tokens` | int | 默认输出 token 上限。 |
| `stop` | string[] | 停止词列表。 |
| `seed` | int\|null | 随机种子；`null` 表示不固定。 |
| `presence_penalty` | number | 存在惩罚。 |
| `frequency_penalty` | number | 频率惩罚。 |
| `thinking_budget` | int\|null | thinking 预算 token；`null` 表示不启用。 |
| `parallel_tool_calls` | bool\|null | 是否允许并行工具调用；`null` 交给 provider 默认。 |
| `provider_options` | object | 厂商专用参数，原样合并到请求体；请求级 `GenerationOptions` 可覆盖。 |

embedding/rerank 不使用 `defaults`。未识别的键不会自动转发，除非放在
`provider_options` 中。

### 3.7 `metadata` 已识别键

| 键 | 类型 | 说明 |
|---|---|---|
| `variant` | string | chat 兼容网关的方言提示；未识别的 OpenAI 兼容 provider 可借此选择具体实现。 |
| `endpoint_path` | string | rerank 服务路径覆盖；DashScope rerank 使用。 |
| `send_input_type` | bool | embedding/rerank 是否把 `input_type` 发送给服务。 |
| `input_type` | string | 配合 `send_input_type` 使用的输入类型值。 |
| `send_normalize` | bool | embedding 是否发送归一化参数。 |
| `documents_format` | string | Cohere rerank 文档格式；设为 `"object"` 时使用对象格式。 |
| `use_max_completion_tokens` | bool | OpenAI chat 是否使用 `max_completion_tokens` 字段。 |
| `anthropic_version` | string | Anthropic SDK 版本 header 覆盖。 |

`model_specs.py` 还会自动把以下内容写入 provider metadata，不要手工重复：
`vendor`、`thinking_enabled`、`thinking_options`、`thinking_disabled_options`、
`defaults`。

### 3.8 当前模型目录

| `id` | `kind` | 说明 |
|---|---|---|
| `chat-main` | chat | 主对话模型；默认开启 thinking。 |
| `chat-no-thinking` | chat | 显式关闭 thinking 的对话模型，适合抽取/压缩等辅助任务。 |
| `rerank-default` | rerank | 百炼 rerank 模型。 |
| `embed-default` | embedding | 百炼 embedding 模型，维度需与 Weaviate 对齐。 |

模型 id 被引用后会做启动期检查：`LLM__MODEL_IDS`、`LLM__DEFAULT_MODEL_ID`、
`LLM__FALLBACK_MODEL_IDS`、RAG 的 embedding/rerank/描述/图模型、记忆抽取/压缩
模型、Agent 查询规划/反思模型等引用不存在时直接报错。对应关系包括：

- `LLM__DEFAULT_MODEL_ID` / `LLM__MODEL_IDS` / `LLM__FALLBACK_MODEL_IDS`；
- `RAG__INDEXER__EMBEDDING_MODEL_ID`、`DESCRIPTION_MODEL_ID`、
  `TABLE_SUMMARY_MODEL_ID`、`CODE_SUMMARY_MODEL_ID`、`GRAPH_MODEL_ID`；
- `RAG__RETRIEVER__RERANK_MODEL_ID`、`CHUNK_CONFLICT_MODEL_ID`；
- `AGENT__QUERY_PLAN__MODEL_ID`、`AGENT__REFLECTION__MODEL_ID`；
- `MEMORY__EXTRACTION_MODEL_ID`、`MEMORY__COMPRESSION_MODEL_ID`。

---

## 4. `config/domains/*.json`：垂直领域画像

### 4.1 定位

领域画像是**辅助配置**，不是实体/关系全集，也不做硬穷举：

- 提供默认实体类型、关系类型、类型/关系白名单/黑名单；
- 提供实体标准名与别名，用于抽取后归一；
- 提供噪声词、角色、抽取规则和示例；
- 不直接生成图节点，不生成检索 seed；
- 图检索的下一跳不受画像关系枚举限制。

内置文件：

| 文件 | `display_name` | `master` | 说明 |
|---|---|---|---|
| `medical.json` | 医疗 | 空 | 人类医疗参考主画像，只有知识库显式绑定 `medical` 时使用。 |
| `pet_medical.json` | 宠物医疗 | 空 | 宠物医疗公共主画像。 |
| `pet_case.json` | 宠物医疗案例 | `pet_medical` | 病例/诊断/转归子画像。 |
| `pet_category.json` | 宠物类别 | `pet_medical` | 物种/品种/分类维度子画像。 |
| `pet_drug.json` | 宠物药品 | `pet_medical` | 给药途径/商品名子画像。 |

### 4.2 顶层字段

| 字段 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `name` | string | 是 | — | 画像名；非空。通常与文件名一致，加载器也会用它做路径校验。 |
| `display_name` | string | 否 | `""` | 展示名；用于领域目录和提示词。为空时退回 `name`。 |
| `version` | string | 否 | `"1"` | 画像版本，便于人工标识，也会进入图抽取缓存 key；缓存失效同时取决于内容签名，通常不需要为了改内容手工递增。 |
| `language` | string | 否 | `"zh"` | 领域语言标记。 |
| `description` | string | 否 | `""` | 领域说明；用于领域目录/提示词展示。 |
| `master` | string | 否 | `""` | 主画像名；为空表示自身是主画像。子画像加载时会递归合并主画像。不能等于自身 `name`。 |
| `master_display_name` | string | 否 | `""` | 主画像展示名；由加载器在合并时写入，通常不要手写。 |
| `entity_types` | object[] | 否 | `[]` | 默认实体类型；字段见 §4.3。合并/过滤后至少保留一个有效类型。 |
| `relations` | object[] | 否 | `[]` | 默认关系类型；字段见 §4.4。 |
| `type_whitelist` | object[] | 否 | `[]` | 类型白名单；按 `name` 补充/覆盖 `entity_types`，不是实体名白名单。 |
| `relation_whitelist` | object[] | 否 | `[]` | 关系白名单；按 `name` 补充/覆盖 `relations`。 |
| `type_blacklist` | string[] | 否 | `[]` | 类型黑名单；按类型 `name` 从有效集合移除，并在抽取结果中过滤。 |
| `relation_blacklist` | string[] | 否 | `[]` | 关系黑名单；按关系 `name` 过滤。 |
| `aliases` | object[] | 否 | `[]` | 实体标准名/别名映射；字段见 §4.5。 |
| `noise_terms` | string[] | 否 | `[]` | 实体名噪声词；只做实体名过滤，不进入 Prompt。 |
| `prompt` | object | 否 | `{}` | 领域提示词片段；字段见 §4.6。 |

### 4.3 `entity_types[]`

| 字段 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `name` | string | 是 | — | 类型标识，如 `disease`；同一有效集合内不可重复。 |
| `label` | string | 否 | `""` | 展示标签，如 `疾病/综合征`；不能与其他类型的 name/label 冲突。 |
| `description` | string | 否 | `""` | 类型说明，进入图抽取 Prompt。 |
| `examples` | string[] | 否 | `[]` | 粒度示例，进入 Prompt；**不是**抽取白名单。 |

有效类型集合 = `entity_types` + `type_whitelist` - `type_blacklist`。白名单按
`name` 合并，同名项补充/覆盖；黑名单只做移除和抽取后过滤。LLM 仍可输出清单外
类型，是否入库由 RAG 的 `graph_allow_unknown_types` 等开关决定。

### 4.4 `relations[]` 与 `relation_whitelist[]`

| 字段 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `name` | string | 是 | — | 关系标识，如 `has_symptom`；同一有效集合内不可重复。 |
| `label` | string | 否 | `""` | 展示标签，如 `疾病→症状`。 |
| `description` | string | 否 | `""` | 关系说明，进入 Prompt。 |

有效关系集合 = `relations` + `relation_whitelist` - `relation_blacklist`。
关系黑名单只做过滤；图检索下一跳不按画像关系枚举限制。

### 4.5 `aliases[]`

| 字段 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `name` | string | 是 | — | 实体标准名；会以“已知标准名”进入图抽取 Prompt。 |
| `type` | string | 是 | — | 实体类型；必须是有效 `entity_types` 中的 `name`。 |
| `aliases` | string[] | 否 | `[]` | 别名/俗称/缩写；只用于抽取后归一，不进入 Prompt。 |
| `description` | string | 否 | `""` | 可选说明。 |

校验规则：

- 标准名和每个别名归一后长度至少 2；空值失败；
- 同一个别名不能属于两个不同实体；
- 标准名/别名不能同时出现在 `noise_terms`；
- 标准名、别名在匹配时忽略首尾/重复空白，英文大小写不敏感；
- 匹配按最长优先，避免短词截断长词。

### 4.6 `prompt`

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `role` | string | `""` | 领域抽取器角色描述。 |
| `entity_guidance` | string | `""` | 实体抽取规则。 |
| `relation_guidance` | string | `""` | 关系抽取规则。 |
| `extra_instructions` | string[] | `[]` | 额外指令，每行一条。 |
| `entity_examples` | string[] | `[]` | 好例子，每行一个“实体名（类型）：说明”。 |
| `noise_examples` | string[] | `[]` | 噪音例子，每行一类不应成为节点的文本。 |

### 4.7 主画像与子画像合并

1. 子画像的 `master` 指向主画像；加载器优先在同一目录查找主画像，其次配置
   目录，最后内置目录；循环引用直接报错；
2. `entity_types`、`relations`、`type_whitelist`、`relation_whitelist` 按
   `name` 合并，子画像同名项覆盖主画像；
3. `type_blacklist`、`relation_blacklist`、`noise_terms` 做保序并集；
4. `aliases` 按 `name` 合并：同名条目继承主画像别名，再叠加子画像别名；
5. `prompt` 字符串字段由子画像非空覆盖；列表字段拼接去重；
6. `display_name`、`version`、`language`、`description` 等标量由子画像非空值
   覆盖；
7. `name` 和 `master` 始终以子画像为准。

加载后 `entity_types` / `relations` 已被折叠为有效集合；`type_whitelist` /
`relation_whitelist` 不再参与运行期查找。画像内容变化会改变内容签名，图抽取
缓存随之失效，无需手工改 `version`。

### 4.8 加载期校验

- 至少有一个有效实体类型；
- 类型/关系名称不可重复；
- 类型 `label` 不能与另一个类型的 name/label 冲突；
- `aliases[].type` 必须是有效类型；
- 别名不能重复、不能过短、不能属于两个实体；
- 别名不能与 `noise_terms` 冲突；
- 所有未知字段报错。

---

## 5. `prompt/templates/manifest.json`：Prompt 元数据

### 5.1 顶层结构

```json
{
  "version": 1,
  "prompts": {}
}
```

| 字段 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `version` | int | 否 | `1` | manifest 版本。 |
| `prompts` | object | 否 | `{}` | Prompt id → Prompt 条目；加载时该对象不能为空，并且最终必须覆盖全部 `PromptId`，否则启动失败。 |

Prompt id 必须匹配 `^[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)+$`，例如
`agent.system`、`rag.graph.system`。模板正文放在单独的 `.md` 文件，不写进
manifest。

### 5.2 单个 Prompt 条目字段

| 字段 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `file` | string | 是 | — | 相对模板根目录的文件路径；不能使用 `../` 越出根目录。覆盖目录新增 Prompt 时必须提供。 |
| `version` | int | 否 | `1` | 模板版本，必须 `>=1`；进入渲染结果的 `version`。 |
| `description` | string | 否 | `""` | 模板说明。 |
| `purpose` | string | 否 | `""` | 用途标签，如 `rag.graph_extraction`、`agent.chat`。 |
| `static` | bool | 否 | `false` | 是否静态模板。`static=true` 时 `variables` 必须为空，正文不能含 `${...}`。 |
| `cacheable` | bool | 否 | `true` | 是否适合 prompt caching；只是提示，不实际启用缓存。 |
| `variables` | object[] | 否 | `[]` | 变量声明；必须与正文中 `${name}` 占位符完全一致。 |

### 5.3 `variables[]`

| 字段 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `name` | string | 是 | — | 变量名；同一 Prompt 内唯一。 |
| `description` | string | 否 | `""` | 变量说明，供维护者阅读。 |
| `required` | bool | 否 | `true` | 是否必填。缺失且没有 `default` 时，严格模式渲染失败。 |
| `default` | string\|null | 否 | `null` | 默认值；设置后缺失时使用。 |

渲染规则：

- 正文使用标准库 `string.Template` 的 `${name}` 语法；
- manifest 声明的变量与正文占位符必须双向一致，多一个/少一个都在启动期报错；
- 调用方传入未声明的变量会被忽略，便于覆盖模板只使用变量子集；
- `strict=true`（默认）时缺少必填变量直接报错；非严格模式补空串；
- `None` 会转成空串，不会渲染出字符串 `"None"`；
- 正文 SHA256 进入渲染结果，供审计/缓存比对。

### 5.4 当前 Prompt 清单

`*` 表示 `required=true`；`可选` 表示 `required=false` 或提供默认值。

| Prompt id | 文件 | 版本 | static | cacheable | 用途 | 变量 |
|---|---|---|---|---|---|---|
| `rag.description.system` | `rag/description.system.md` | 1 | false | true | `rag.document_description` | `max_chars*` |
| `rag.description.user` | `rag/description.user.md` | 1 | false | false | `rag.document_description` | `source*`、`candidate*`、`context*` |
| `rag.graph.system` | `rag/graph.system.md` | 6 | false | true | `rag.graph_extraction` | `role`（可选，有默认）、`entity_types*`、`relation_types`（可选，有默认）、`entity_guidance`（可选，有默认）、`relation_guidance`（可选，有默认）、`extra_instructions`（可选，默认空）、`max_entities*`、`max_relations*`、`entity_examples`（可选，有默认）、`noise_examples`（可选，有默认）、`alias_names`（可选，有默认） |
| `rag.graph.user` | `rag/graph.user.md` | 3 | false | false | `rag.graph_extraction` | `chunks_json*` |
| `rag.chunk_conflict.system` | `rag/chunk_conflict.system.md` | 1 | true | true | `rag.chunk_conflict` | 无 |
| `rag.table.description.system` | `rag/table.description.system.md` | 1 | false | true | `rag.special_embedding` | `max_chars*` |
| `rag.table.description.user` | `rag/table.description.user.md` | 1 | false | false | `rag.special_embedding` | `doc_type`、`heading`、`filename`、`sheet`、`caption`、`context`、`headers`、`rows`（均可选，有默认） |
| `rag.code.description.system` | `rag/code.description.system.md` | 1 | false | true | `rag.special_embedding` | `max_chars*` |
| `rag.code.description.user` | `rag/code.description.user.md` | 1 | false | false | `rag.special_embedding` | `heading`、`file_path`、`symbol_path`、`signature`、`docstring`、`context`、`code`（均可选，有默认） |
| `agent.system` | `agent/system.md` | 3 | false | true | `agent.chat` | `assistant_name`（默认 `Verity`）、`domain_name`（默认 `通用`）、`domain_description`（默认空）、`current_date`（默认 `未知`） |
| `agent.query_plan.system` | `agent/query_plan.system.md` | 2 | false | true | `agent.query_plan` | `domain_entity_types`（可选，有默认） |
| `agent.reflection.system` | `agent/reflection.system.md` | 1 | true | true | `agent.reflection` | 无 |
| `memory.extraction.system` | `memory/extraction.system.md` | 5 | true | true | `memory.extraction` | 无 |
| `memory.compression.system` | `memory/compression.system.md` | 8 | true | true | `memory.compression` | 无 |

### 5.5 覆盖目录合并

外部覆盖目录也必须包含 `manifest.json`：

- 同名 Prompt id：只覆盖显式提供的字段，未提供的继承内置；
- 覆盖项显式给了 `file`，正文从覆盖目录读取；否则仍使用内置正文；
- 新增 Prompt id 必须提供 `file`；
- 最终必须覆盖 `PromptId` 中的全部 Prompt，否则启动失败；
- 覆盖目录的路径、`include_builtin`、`strict` 由 `PromptConfig` 控制，不在
  本 JSON 中配置。

---

## 6. 跨文件关系与修改检查清单

### 6.1 常见联动

| 修改 | 需要同时检查 |
|---|---|
| 新增/修改模型 id | `config/models.json` 与 `config/llm.py`（默认模型/白名单/fallback）、RAG/记忆/Agent 的模型引用；启动期会校验引用存在。 |
| 修改 embedding 维度 | `config/models.json` 的 `capabilities.dimensions`、Weaviate schema、索引版本；已有向量需要重建。 |
| 新增工具 | `agent/tools/manifest.json` 的 `implementation`、`required_services`、`exposure`；工厂模块；必要时 `ToolServiceKey`。 |
| 修改工具默认策略 | `config/tools/policies.default.json`；单工具差异仍放 manifest；确认依赖 Redis。 |
| 新增领域类型/关系 | 对应 `config/domains/*.json`；如果换领域，还要在知识库绑定画像并重新入库图。 |
| 新增 Prompt | `prompt/templates/manifest.json`、`.md` 正文、`PromptId`；变量声明与 `${...}` 必须一致。 |
| 修改 Prompt 变量 | 同步 manifest `variables` 与所有调用方的 render 参数。 |

### 6.2 通用修改流程

1. 使用标准 JSON 编辑，确认没有注释、尾逗号和未知字段；
2. 启动服务或运行配置测试，让 Pydantic 在启动期完成校验；
3. 查看启动日志中的 warning：工具依赖缺失、thinking 配置不完整、模型/画像
   引用错误都会在这里暴露；
4. 配置变更后重启进程；不要依赖运行期热加载；
5. 涉及模型维度、领域画像或工具声明的变更，检查是否需要重建索引/图数据。

### 6.3 安全约定

- 优先使用 `api_key_env`，不要把真实密钥写进 `config/models.json`；
- `headers`、`metadata` 可能进入日志，不要放密钥；
- `required_permission` 是能力码，不是资源白名单；资源权限由工具执行器/检索
  流水线按参数展开；
- 工具 `description` 会直接进入 LLM Prompt，不要写入内部路径、密钥或未脱敏的
  业务数据。
