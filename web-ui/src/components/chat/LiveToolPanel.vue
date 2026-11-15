<template>
  <div v-if="items.length" class="live-tools">
    <div class="live-tools__title">
      <el-icon><Tools /></el-icon>
      <span>工具执行</span>
    </div>
    <div v-for="item in items" :key="item.callId" class="live-tool">
      <el-icon :class="{ 'is-spinning': item.status === 'running' }">
        <Loading v-if="item.status === 'running'" />
        <CircleCheckFilled v-else-if="item.status === 'succeeded'" />
        <CircleCloseFilled v-else-if="item.status === 'failed'" />
        <WarningFilled v-else />
      </el-icon>
      <span class="live-tool__name">{{ item.name }}</span>
      <span class="live-tool__status">
        {{ statusText(item.status) }}
        <template v-if="item.errorCode"> · {{ item.errorCode }}</template>
      </span>
    </div>
  </div>
</template>

<script setup lang="ts">
import {
  CircleCheckFilled,
  CircleCloseFilled,
  Loading,
  Tools,
  WarningFilled,
} from '@element-plus/icons-vue'

import type { LiveToolCall } from '@/stores/chat'

const props = defineProps<{ tools: Record<string, LiveToolCall> }>()

const items = computed(() => Object.values(props.tools))

function statusText(status: string): string {
  const labels: Record<string, string> = {
    running: '执行中',
    succeeded: '已完成',
    failed: '失败',
    cancelled: '已取消',
  }
  return labels[status] ?? status
}
</script>

<style scoped>
.live-tools {
  display: flex;
  flex-direction: column;
  gap: var(--app-space-xs);
  width: 100%;
  max-width: var(--app-chat-max-width);
  padding: var(--app-space-sm) var(--app-space-md);
  margin: 0 auto var(--app-space-sm);
  font-size: var(--app-font-size-sm);
  background-color: var(--app-bg-tool);
  border: 1px dashed var(--app-border-color);
  border-radius: var(--app-radius-md);
}

.live-tools__title {
  display: flex;
  align-items: center;
  gap: var(--app-space-xs);
  font-weight: 500;
  color: var(--app-text-secondary);
}

.live-tool {
  display: flex;
  align-items: center;
  gap: var(--app-space-xs);
  color: var(--app-text-regular);
}

.live-tool__name {
  font-family: var(--app-font-family-mono);
  color: var(--app-text-primary);
}

.live-tool__status {
  margin-left: auto;
  font-size: var(--app-font-size-xs);
  color: var(--app-text-muted);
}

.is-spinning {
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}
</style>
