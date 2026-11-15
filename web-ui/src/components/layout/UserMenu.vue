<template>
  <el-dropdown class="user-menu-root" trigger="click" @command="handleCommand">
    <button type="button" class="user-menu" aria-label="用户菜单">
      <span class="user-menu__avatar" aria-hidden="true">{{ userInitial }}</span>
      <span class="user-menu__text">
        <span class="user-menu__name">{{ userStore.userInfo?.username ?? '未登录' }}</span>
        <span class="user-menu__role">{{ roleLabel }}</span>
      </span>
      <el-icon class="user-menu__arrow"><ArrowDown /></el-icon>
    </button>
    <template #dropdown>
      <el-dropdown-menu>
        <el-dropdown-item disabled>
          {{ userStore.permissions.length }} 项权限
        </el-dropdown-item>
        <el-dropdown-item divided command="logout">退出登录</el-dropdown-item>
      </el-dropdown-menu>
    </template>
  </el-dropdown>
</template>

<script setup lang="ts">
import { ArrowDown } from '@element-plus/icons-vue'

import { useChatStore } from '@/stores/chat'
import { useConversationStore } from '@/stores/conversations'
import { useUserStore } from '@/stores/user'

const userStore = useUserStore()
const conversationStore = useConversationStore()
const chatStore = useChatStore()
const router = useRouter()

const roleLabel = computed(() =>
  userStore.userInfo?.role === 'admin' ? '管理员' : '普通用户',
)

const userInitial = computed(() => {
  const name = userStore.userInfo?.username?.trim()
  return name ? name.slice(0, 1).toUpperCase() : '?'
})

function handleCommand(command: string | number | object): void {
  if (command !== 'logout') return
  chatStore.dispose()
  chatStore.resetAll()
  conversationStore.reset()
  userStore.logout()
  ElMessage.success('已退出登录')
  void router.replace({ name: 'login' })
}
</script>

<style scoped>
.user-menu {
  display: flex;
  align-items: center;
  gap: var(--app-space-sm);
  width: 100%;
  padding: var(--app-space-xs) var(--app-space-sm);
  text-align: left;
  cursor: pointer;
  background: transparent;
  border: 1px solid transparent;
  border-radius: var(--app-radius-md);
  transition:
    background-color var(--app-transition-fast),
    border-color var(--app-transition-fast);
}

.user-menu:hover {
  background-color: var(--app-bg-hover);
  border-color: var(--app-border-color-light);
}

.user-menu__avatar {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex: 0 0 auto;
  width: 30px;
  height: 30px;
  font-size: var(--app-font-size-xs);
  font-weight: 600;
  color: var(--app-color-primary);
  background-color: var(--app-color-primary-soft);
  border-radius: var(--app-radius-pill);
}

.user-menu__text {
  display: flex;
  flex-direction: column;
  min-width: 0;
}

.user-menu__name {
  overflow: hidden;
  font-size: var(--app-font-size-sm);
  color: var(--app-text-primary);
  text-overflow: ellipsis;
  white-space: nowrap;
}

.user-menu__role {
  font-size: var(--app-font-size-xs);
  color: var(--app-text-muted);
}

.user-menu__arrow {
  margin-left: auto;
  color: var(--app-text-muted);
}
</style>
