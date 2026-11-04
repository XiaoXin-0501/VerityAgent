/** Token 本地存储。请求层与状态层共用，避免循环依赖。 */

const TOKEN_KEY = 'verity-agent-token'

export function getToken(): string {
  return localStorage.getItem(TOKEN_KEY) ?? ''
}

export function setToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token)
}

export function removeToken(): void {
  localStorage.removeItem(TOKEN_KEY)
}
