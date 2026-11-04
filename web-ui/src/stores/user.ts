import { getUserInfoApi, loginApi } from '@/api/modules/auth'
import type { LoginParams, UserInfo } from '@/api/types'
import { getToken, removeToken, setToken } from '@/utils/auth'

export const useUserStore = defineStore('user', () => {
  const token = ref(getToken())
  const userInfo = ref<UserInfo | null>(null)
  const permissions = ref<string[]>([])

  const isLoggedIn = computed(() => Boolean(token.value))

  /** 登录：保存 token + 拉取完整用户态（含权限码）。 */
  async function login(params: LoginParams): Promise<void> {
    const result = await loginApi(params)
    token.value = result.access_token
    setToken(result.access_token)
    userInfo.value = result.user
    await fetchUserInfo()
  }

  /** 拉取当前用户与权限码（页面刷新 / 路由守卫时使用）。 */
  async function fetchUserInfo(): Promise<void> {
    const result = await getUserInfoApi()
    userInfo.value = result.user
    permissions.value = result.permissions
  }

  function hasPermission(code: string): boolean {
    return permissions.value.includes(code)
  }

  function logout(): void {
    token.value = ''
    userInfo.value = null
    permissions.value = []
    removeToken()
  }

  return {
    token,
    userInfo,
    permissions,
    isLoggedIn,
    login,
    fetchUserInfo,
    hasPermission,
    logout,
  }
})
