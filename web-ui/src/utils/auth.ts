/** Token 本地存储。请求层与状态层共用，避免循环依赖。 */

import { appConfig } from '@/config'

export function getToken(): string {
  return localStorage.getItem(appConfig.tokenStorageKey) ?? ''
}

export function setToken(token: string): void {
  localStorage.setItem(appConfig.tokenStorageKey, token)
}

export function removeToken(): void {
  localStorage.removeItem(appConfig.tokenStorageKey)
}
