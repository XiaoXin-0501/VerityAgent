import http from '@/utils/request'

import type { LoginParams, LoginResult, MeResult } from '@/api/types'

/** 用户名 + 密码登录，返回 JWT。 */
export function loginApi(data: LoginParams): Promise<LoginResult> {
  return http.post<LoginResult>('/auth/login', data)
}

/** 获取当前登录用户与权限码。 */
export function getUserInfoApi(): Promise<MeResult> {
  return http.get<MeResult>('/auth/me')
}
