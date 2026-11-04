import axios, { type AxiosError } from 'axios'

import { getToken, removeToken } from '@/utils/auth'

import { ApiError, type RequestConfig } from './types'

const BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api'
const TIMEOUT = 15_000

const service = axios.create({
  baseURL: BASE_URL,
  timeout: TIMEOUT,
})

/** 统一附加 Bearer token。 */
service.interceptors.request.use((config) => {
  const token = getToken()
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

/** FastAPI 错误体：{ detail: string } 或 422 的 { detail: [{ msg }] }。 */
function extractMessage(payload: unknown): string {
  if (!payload) return ''
  if (typeof payload === 'string') return payload
  if (typeof payload !== 'object') return ''

  const detail = (payload as { detail?: unknown }).detail
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (item && typeof item === 'object' && 'msg' in item) {
          return String((item as { msg: unknown }).msg)
        }
        return String(item)
      })
      .join('；')
  }

  const message = (payload as { message?: unknown }).message
  return typeof message === 'string' ? message : ''
}

function extractCode(payload: unknown): string | undefined {
  if (payload && typeof payload === 'object' && 'code' in payload) {
    const code = (payload as { code?: unknown }).code
    if (typeof code === 'string') return code
  }
  return undefined
}

let redirecting = false

/** 401：清理 token 并回到登录页（动态导入 router，避免循环依赖）。 */
async function redirectToLogin(): Promise<void> {
  removeToken()
  if (redirecting) return
  redirecting = true
  try {
    const { default: router } = await import('@/router')
    const current = router.currentRoute.value
    // matched 为空说明首次导航尚未完成，交给路由守卫处理，避免重复跳转
    if (current.matched.length > 0 && current.name !== 'login') {
      await router.replace({
        name: 'login',
        query: current.fullPath && current.fullPath !== '/' ? { redirect: current.fullPath } : {},
      })
    }
  } finally {
    redirecting = false
  }
}

service.interceptors.response.use(
  // 后端直接返回业务 JSON，这里直接解包 data
  (response) => response.data,
  async (error: AxiosError<unknown>) => {
    const status = error.response?.status
    const payload = error.response?.data

    let message = extractMessage(payload)
    if (!message) {
      if (error.code === 'ECONNABORTED') {
        message = '请求超时，请稍后重试'
      } else if (error.response) {
        message = `请求失败（${status ?? '未知状态'}）`
      } else {
        message = '网络异常，请检查网络连接'
      }
    }

    if (status === 401) {
      await redirectToLogin()
    }
    ElMessage.error(message)

    return Promise.reject(
      new ApiError(message, { status, code: extractCode(payload), data: payload }),
    )
  },
)

/**
 * 拦截器已把 response.data 解包，实际返回就是 T。
 * axios 1.21 的 AxiosResponseResult 条件类型无法对泛型 T 收窄，这里做一次显式收口。
 */
function request<T>(config: RequestConfig): Promise<T> {
  return service.request<unknown, T>(config) as unknown as Promise<T>
}

/** 类型安全的请求出口：拦截器已解包，返回的直接是 T。 */
export const http = {
  get<T>(url: string, config?: RequestConfig): Promise<T> {
    return request<T>({ ...config, url, method: 'GET' })
  },
  post<T>(url: string, data?: unknown, config?: RequestConfig): Promise<T> {
    return request<T>({ ...config, url, method: 'POST', data })
  },
  put<T>(url: string, data?: unknown, config?: RequestConfig): Promise<T> {
    return request<T>({ ...config, url, method: 'PUT', data })
  },
  patch<T>(url: string, data?: unknown, config?: RequestConfig): Promise<T> {
    return request<T>({ ...config, url, method: 'PATCH', data })
  },
  delete<T>(url: string, config?: RequestConfig): Promise<T> {
    return request<T>({ ...config, url, method: 'DELETE' })
  },
}

export default http
