# web-ui

VerityAgent 前端：Vue 3 + TypeScript + Vite + Pinia + Vue Router + Element Plus。

## 开发

```sh
pnpm install
pnpm dev
```

默认地址 `http://localhost:5173`，`/api` 代理到 `http://127.0.0.1:8000`。
后端需先启动：

```sh
cd ../verity-agent
uv run python -m api.server
```

## 环境变量

所有可变配置放在 `.env.development` / `.env.production`，不要在组件里写死。

| 变量 | 默认 | 说明 |
|---|---|---|
| `VITE_API_BASE_URL` | `/api/v1` | 后端 API 前缀（后端 `APP__API_PREFIX`） |
| `VITE_PROXY_TARGET` | `http://127.0.0.1:8000` | 仅开发环境 Vite 代理目标 |
| `VITE_APP_NAME` | `VerityAgent` | 品牌名/页面标题 |
| `VITE_TOKEN_STORAGE_KEY` | `verity-agent-token` | JWT localStorage key |
| `VITE_THEME_STORAGE_KEY` | `verity-agent-theme` | 主题 localStorage key |
| `VITE_REQUEST_TIMEOUT_MS` | `15000` | axios 超时 |
| `VITE_MESSAGE_PAGE_SIZE` | `50` | 消息分页大小 |
| `VITE_MAX_MESSAGE_PAGES` | `20` | 消息翻页上限 |
| `VITE_SSE_RECONNECT_DELAY_MS` | `1000` | SSE 重连间隔 |
| `VITE_SSE_MAX_RECONNECT_ATTEMPTS` | `5` | SSE 最大重连次数 |
| `VITE_SHOW_THINKING` | `true` | 是否展示思考过程 |
| `VITE_CONVERSATION_TITLE_MAX_LENGTH` | `30` | 首条消息生成会话标题长度 |
| `VITE_MAX_MESSAGE_CHARS` | `200000` | 单条消息最大字符数，对齐后端 `MessageConfig` |
| `VITE_RUN_STORAGE_PREFIX` | `verity-agent-run:` | 会话当前 run_id 的 sessionStorage 前缀；刷新后可恢复 SSE |
| `VITE_INDEX_JOB_POLL_INTERVAL_MS` | `2000` | 入库/删除任务轮询间隔 |
| `VITE_INDEX_JOB_POLL_MAX_ATTEMPTS` | `120` | 任务轮询最大次数 |
| `VITE_DEFAULT_KB_PERMISSION_CODE` | `kb:read` | 新建知识库默认权限码 |

## 目录

```text
src/
├── api/                 # DTO 与服务端接口（与 FastAPI 字段一致）
│   └── modules/         # auth / conversations / runs / knowledge
├── assets/styles/       # 设计变量、reset、全局样式
├── components/
│   ├── chat/            # 会话侧栏、消息、Markdown、工具卡片、输入框
│   ├── layout/          # 品牌标识、用户菜单
│   └── common/          # 主题切换等通用组件
├── composables/         # useTheme
├── config/              # Vite env -> 运行期配置
├── layouts/             # 应用壳
├── router/              # 路由与登录/管理员守卫
├── stores/              # user / conversations / chat
├── utils/               # 请求、SSE、Markdown、格式化、幂等 id
└── views/               # login / home(ChatView) / admin(KnowledgeBaseView)
```

## 后端对接

| 前端行为 | 后端接口 |
|---|---|
| 登录 / 恢复用户态 | `POST /api/v1/auth/login`、`GET /api/v1/auth/me` |
| 会话列表 / 新建 | `GET /api/v1/conversations`、`POST /api/v1/conversations` |
| 消息历史 | `GET /api/v1/conversations/{id}/messages`（`after_seq` 翻页） |
| 发送消息 | `POST /api/v1/conversations/{id}/messages`（`stream=true`） |
| 订阅生成 | `GET /api/v1/runs/{run_id}/events`（Bearer + `Last-Event-ID`） |
| 取消生成 | `POST /api/v1/runs/{run_id}/cancel` |
| 管理员：知识库 | `GET/POST /api/v1/knowledge-bases` |
| 管理员：文档 | `GET/POST/DELETE /api/v1/knowledge-bases/{kb_id}/documents...` |
| 管理员：入库任务 | `GET /api/v1/index-jobs/{job_id}` |

新建对话先进入前端草稿态，不调用后端；用户发送首条消息时才创建会话
（标题取首条消息前 `VITE_CONVERSATION_TITLE_MAX_LENGTH` 个字符），并立即发起 run。

知识库管理页仅 `role=admin` 可见：创建、禁用/启用、删除知识库，上传/删除文档，
轮询入库进度。删除知识库需要在弹窗中再次输入当前管理员密码，随后软删元数据
并提交其下文档的索引清理任务。
向量模型/维度由服务端 `RAG__INDEXER__*` 决定，前端不传。

> 默认种子账号 `xiaoxin` 启动时会被同步为 `admin`；已有库在 dev/test
> 启动 `init_schema` 时会更新角色。生产环境不会自动改表，可手工执行：
> `UPDATE users SET role='admin' WHERE username='xiaoxin';`

## 目录

SSE 使用 `fetch` + `ReadableStream`，因为原生 `EventSource` 不能携带
`Authorization` 头。事件 `data` 是完整 `AgentEvent` 信封，前端从
`event.data` 读取 `assistant.delta / tool.* / run.*` 等字段。
发送消息后把 `run_id` 写入 `sessionStorage`，刷新页面/切回会话时通过
`GET /runs/{run_id}` 判断是否仍在运行，再用 `Last-Event-ID` 重新 attach。

## 设计约定

- 组件样式只使用 `var(--app-*)`，颜色、间距、圆角、阴影全部在
  `src/assets/styles/variables.css` 定义；深色模式使用 `[data-theme='dark']`。
- API 返回字段使用后端 snake_case，不额外做 camelCase 转换。
- 不消费内部模型（`AgentRun / ContextPlan / MemoryView`），只使用公共 DTO。
- 当前后端没有文件上传、知识库管理、模型列表接口，前端不提供对应入口。

## 构建 / 检查

```sh
pnpm type-check
pnpm lint
pnpm build
```
