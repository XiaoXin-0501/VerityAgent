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

/** 后端统一错误信封：{ error: { code, message, details } }。 */
function extractError(payload: unknown): { message: string; code?: string } {
  if (!payload) return { message: '' }
  if (typeof payload === 'string') return { message: payload }
  if (typeof payload !== 'object') return { message: '' }

  const envelope = (payload as { error?: unknown }).error
  if (envelope && typeof envelope === 'object') {
    const error = envelope as { code?: unknown; message?: unknown }
    return {
      message: typeof error.message === 'string' ? error.message : '',
      code: typeof error.code === 'string' ? error.code : undefined,
    }
  }

  const message = (payload as { message?: unknown }).message
  return { message: typeof message === 'string' ? message : '' }
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

    const { message: backendMessage, code } = extractError(payload)
    let message = backendMessage
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

    return Promise.reject(new ApiError(message, { status, code, data: payload }))
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
