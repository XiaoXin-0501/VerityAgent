/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** API 基础路径，默认 /api/v1 */
  readonly VITE_API_BASE_URL: string
  /** 开发环境 Vite 代理目标 */
  readonly VITE_PROXY_TARGET?: string
  /** 应用名称 */
  readonly VITE_APP_NAME?: string
  /** localStorage 键名 */
  readonly VITE_TOKEN_STORAGE_KEY?: string
  readonly VITE_THEME_STORAGE_KEY?: string
  /** 请求超时（毫秒） */
  readonly VITE_REQUEST_TIMEOUT_MS?: string
  /** 消息分页大小 / 最大翻页数 */
  readonly VITE_MESSAGE_PAGE_SIZE?: string
  readonly VITE_MAX_MESSAGE_PAGES?: string
  /** SSE 断线重连 */
  readonly VITE_SSE_RECONNECT_DELAY_MS?: string
  readonly VITE_SSE_MAX_RECONNECT_ATTEMPTS?: string
  /** 是否展示模型思考过程 */
  readonly VITE_SHOW_THINKING?: string
  /** 新建会话标题最大字符数 */
  readonly VITE_CONVERSATION_TITLE_MAX_LENGTH?: string
  /** 单条消息最大字符数 */
  readonly VITE_MAX_MESSAGE_CHARS?: string
  /** 会话当前 run_id 的 sessionStorage 前缀 */
  readonly VITE_RUN_STORAGE_PREFIX?: string
  /** 知识库入库任务轮询间隔 / 最大次数 */
  readonly VITE_INDEX_JOB_POLL_INTERVAL_MS?: string
  readonly VITE_INDEX_JOB_POLL_MAX_ATTEMPTS?: string
  /** 新建知识库默认权限码 */
  readonly VITE_DEFAULT_KB_PERMISSION_CODE?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
