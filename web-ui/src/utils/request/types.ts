import type { AxiosRequestConfig } from 'axios'

export interface ApiErrorOptions {
  /** HTTP 状态码 */
  status?: number
  /** 后端业务错误码（统一错误信封 error.code） */
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

export type RequestConfig = AxiosRequestConfig & {
  /** 为 true 时不弹全局错误提示，由调用方自行处理。 */
  silent?: boolean
}
