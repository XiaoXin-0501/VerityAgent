<template>
  <section class="chat-view">
    <h2 class="chat-view__title">你好，{{ userStore.userInfo?.username ?? '用户' }}</h2>
    <p class="chat-view__hint">
      登录与用户信息链路已打通，对话功能待接入。
      <span v-if="userStore.permissions.length">
        当前权限：{{ userStore.permissions.join('、') }}
      </span>
    </p>
  </section>
</template>

<script setup lang="ts">
import { useUserStore } from '@/stores/user'

const userStore = useUserStore()
</script>

<style scoped>
.chat-view {
  max-width: var(--app-layout-max-width);
  margin: 0 auto;
  padding: var(--app-space-xl);
  background-color: var(--app-bg-card);
  border-radius: var(--app-radius-lg);
  box-shadow: var(--app-shadow-card);
}

.chat-view__title {
  font-size: var(--app-font-size-xl);
  font-weight: 600;
  color: var(--app-text-primary);
}

.chat-view__hint {
  margin-top: var(--app-space-sm);
  font-size: var(--app-font-size-base);
  color: var(--app-text-secondary);
}
</style>
