/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** axios 基础路径，默认 /api */
  readonly VITE_API_BASE_URL: string
  /** 开发环境 Vite 代理目标 */
  readonly VITE_PROXY_TARGET?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
