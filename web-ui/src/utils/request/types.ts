import type { AxiosRequestConfig } from 'axios'

export interface ApiErrorOptions {
  /** HTTP 状态码 */
  status?: number
  /** 后端业务错误码（当前后端主要返回 detail，保留扩展位） */
  code?: string
  /** 原始响应体 */
  data?: unknown
}

/** 请求层统一抛出的业务异常，页面可按 code / status 做差异化处理。 */
export class ApiError extends Error {
  readonly status?: number
  readonly code?: string
  readonly data?: unknown

  constructor(message: string, options: ApiErrorOptions = {}) {
    super(message)
    this.name = 'ApiError'
    this.status = options.status
    this.code = options.code
    this.data = options.data
  }
}

export type RequestConfig = AxiosRequestConfig
