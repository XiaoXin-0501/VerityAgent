/** 认证相关接口类型（与后端 routers/auth.py 的 Pydantic 模型对应）。 */

export interface LoginParams {
  username: string
  password: string
}

export interface UserInfo {
  id: string
  username: string
  role: string
  email: string | null
}

export interface LoginResult {
  access_token: string
  token_type: string
  expires_in: number
  user: UserInfo
}

export interface MeResult {
  user: UserInfo
  permissions: string[]
}
