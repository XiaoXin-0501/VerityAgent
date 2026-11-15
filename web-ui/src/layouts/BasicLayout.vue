<template>
  <div class="basic-layout">
    <header class="basic-layout__header">
      <div class="basic-layout__brand">
        <span class="basic-layout__mark" aria-hidden="true">V</span>
        <span class="basic-layout__title">{{ appConfig.appName }}</span>
        <el-tag class="basic-layout__env" size="small" effect="plain" type="info">
          知识库助手
        </el-tag>
      </div>

      <div class="basic-layout__actions">
        <ThemeToggle />

        <el-dropdown trigger="click" @command="handleCommand">
          <button type="button" class="basic-layout__user">
            <span class="basic-layout__avatar" aria-hidden="true">
              {{ userInitial }}
            </span>
            <span class="basic-layout__username">
              {{ userStore.userInfo?.username ?? '未登录' }}
            </span>
            <el-icon><ArrowDown /></el-icon>
          </button>
          <template #dropdown>
            <el-dropdown-menu>
              <el-dropdown-item disabled>
                角色：{{ userStore.userInfo?.role ?? '-' }}
              </el-dropdown-item>
              <el-dropdown-item disabled>
                权限：{{ userStore.permissions.length }} 项
              </el-dropdown-item>
              <el-dropdown-item divided command="logout">退出登录</el-dropdown-item>
            </el-dropdown-menu>
          </template>
        </el-dropdown>
      </div>
    </header>

    <main class="basic-layout__main">
      <RouterView />
    </main>
  </div>
</template>

<script setup lang="ts">
import { ArrowDown } from '@element-plus/icons-vue'
import { RouterView } from 'vue-router'

import ThemeToggle from '@/components/common/ThemeToggle.vue'
import { appConfig } from '@/config'
import { useChatStore } from '@/stores/chat'
import { useConversationStore } from '@/stores/conversations'
import { useUserStore } from '@/stores/user'

const userStore = useUserStore()
const conversationStore = useConversationStore()
const chatStore = useChatStore()
const router = useRouter()

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
.basic-layout {
  display: flex;
  flex-direction: column;
  height: 100vh;
  overflow: hidden;
  background-color: var(--app-bg-page);
}

.basic-layout__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex: 0 0 var(--app-layout-header-height);
  height: var(--app-layout-header-height);
  padding: 0 var(--app-space-lg);
  background-color: var(--app-bg-card);
  border-bottom: 1px solid var(--app-border-color);
}

.basic-layout__brand {
  display: flex;
  align-items: center;
  gap: var(--app-space-sm);
  min-width: 0;
}

.basic-layout__mark {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 30px;
  height: 30px;
  font-size: var(--app-font-size-base);
  font-weight: 700;
  color: var(--app-text-on-primary);
  background: linear-gradient(135deg, var(--app-color-primary), var(--app-color-success));
  border-radius: var(--app-radius-md);
}

.basic-layout__title {
  overflow: hidden;
  font-size: var(--app-font-size-lg);
  font-weight: 600;
  color: var(--app-text-primary);
  text-overflow: ellipsis;
  white-space: nowrap;
}

.basic-layout__env {
  margin-left: var(--app-space-2xs);
}

.basic-layout__actions {
  display: flex;
  align-items: center;
  gap: var(--app-space-sm);
}

.basic-layout__user {
  display: flex;
  align-items: center;
  gap: var(--app-space-sm);
  padding: var(--app-space-2xs) var(--app-space-sm);
  color: var(--app-text-regular);
  cursor: pointer;
  background: transparent;
  border: 1px solid transparent;
  border-radius: var(--app-radius-pill);
  transition: background-color var(--app-transition-fast);
}

.basic-layout__user:hover {
  background-color: var(--app-bg-hover);
}

.basic-layout__avatar {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 26px;
  height: 26px;
  font-size: var(--app-font-size-xs);
  font-weight: 600;
  color: var(--app-color-primary);
  background-color: var(--app-color-primary-soft);
  border-radius: var(--app-radius-pill);
}

.basic-layout__username {
  max-width: 140px;
  overflow: hidden;
  font-size: var(--app-font-size-sm);
  text-overflow: ellipsis;
  white-space: nowrap;
}

.basic-layout__main {
  display: flex;
  flex: 1;
  min-height: 0;
}

@media (max-width: 900px) {
  .basic-layout__header {
    padding-inline: var(--app-space-md);
  }

  .basic-layout__env {
    display: none;
  }

  .basic-layout__username {
    display: none;
  }
}
</style>
