<template>
  <div class="basic-layout">
    <header class="basic-layout__header">
      <div class="basic-layout__brand">
        <span class="basic-layout__title">VerityAgent</span>
      </div>

      <div class="basic-layout__actions">
        <el-tag v-if="userStore.permissions.length" type="info" size="small">
          {{ userStore.permissions.length }} 项权限
        </el-tag>

        <el-dropdown trigger="click" @command="handleCommand">
          <span class="basic-layout__user">
            {{ userStore.userInfo?.username ?? '未登录' }}
            <el-icon><ArrowDown /></el-icon>
          </span>
          <template #dropdown>
            <el-dropdown-menu>
              <el-dropdown-item command="logout">退出登录</el-dropdown-item>
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

import { useUserStore } from '@/stores/user'

const userStore = useUserStore()
const router = useRouter()

function handleCommand(command: string | number | object): void {
  if (command !== 'logout') return
  userStore.logout()
  ElMessage.success('已退出登录')
  void router.replace({ name: 'login' })
}
</script>

<style scoped>
.basic-layout {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  background-color: var(--app-bg-page);
}

.basic-layout__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: var(--app-layout-header-height);
  padding: 0 var(--app-space-lg);
  background-color: var(--app-bg-card);
  border-bottom: 1px solid var(--app-border-color);
}

.basic-layout__brand {
  display: flex;
  align-items: center;
  gap: var(--app-space-sm);
}

.basic-layout__title {
  font-size: var(--app-font-size-lg);
  font-weight: 600;
  color: var(--app-text-primary);
}

.basic-layout__actions {
  display: flex;
  align-items: center;
  gap: var(--app-space-md);
}

.basic-layout__user {
  display: flex;
  align-items: center;
  gap: var(--app-space-xs);
  font-size: var(--app-font-size-base);
  color: var(--app-text-regular);
  cursor: pointer;
  outline: none;
}

.basic-layout__main {
  flex: 1;
  padding: var(--app-space-lg);
}
</style>
