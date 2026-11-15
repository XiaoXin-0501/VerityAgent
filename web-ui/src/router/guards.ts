import type { Router } from 'vue-router'

import { appConfig } from '@/config'
import { useUserStore } from '@/stores/user'
import { getToken } from '@/utils/auth'

export function setupRouterGuards(router: Router): void {
  router.beforeEach(async (to) => {
    document.title = to.meta.title
      ? `${to.meta.title} - ${appConfig.appName}`
      : appConfig.appName

    const token = getToken()

    // 公开页：已登录时不再停留在登录页
    if (to.meta.public) {
      return token && to.name === 'login' ? { name: 'home' } : true
    }

    if (!token) {
      return {
        name: 'login',
        query: to.fullPath !== '/' ? { redirect: to.fullPath } : {},
      }
    }

    // 刷新页面后恢复用户态；失败说明 token 失效
    const userStore = useUserStore()
    if (!userStore.userInfo) {
      try {
        await userStore.fetchUserInfo()
      } catch {
        userStore.logout()
        return { name: 'login', query: { redirect: to.fullPath } }
      }
    }

    if (to.meta.requiresAdmin && userStore.userInfo?.role !== 'admin') {
      ElMessage.warning('需要管理员权限')
      return { name: 'home' }
    }

    return true
  })
}
