<template>
  <div v-if="calls.length" class="tool-calls">
    <div v-for="call in calls" :key="call.call_id" class="tool-call">
      <div class="tool-call__head">
        <el-icon class="tool-call__icon"><Tools /></el-icon>
        <span class="tool-call__name">{{ call.name }}</span>
        <el-tag size="small" :type="tagType(call.status)" effect="plain">
          {{ statusText(call.status) }}
        </el-tag>
        <span class="tool-call__id">{{ call.call_id }}</span>
      </div>
      <details v-if="argumentsText(call)" class="tool-call__details">
        <summary>参数</summary>
        <pre>{{ argumentsText(call) }}</pre>
      </details>
    </div>
  </div>
</template>

<script setup lang="ts">
import { Tools } from '@element-plus/icons-vue'

import type { ToolCall } from '@/api/types'

defineProps<{ calls: ToolCall[] }>()

function normalized(status: string | undefined): string {
  return (status ?? '').toUpperCase()
}

function tagType(status: string | undefined): 'success' | 'warning' | 'danger' | 'info' {
  switch (normalized(status)) {
    case 'SUCCEEDED':
      return 'success'
    case 'RUNNING':
    case 'PENDING':
      return 'warning'
    case 'FAILED':
      return 'danger'
    default:
      return 'info'
  }
}

function statusText(status: string | undefined): string {
  const labels: Record<string, string> = {
    PENDING: '等待',
    RUNNING: '执行中',
    SUCCEEDED: '成功',
    FAILED: '失败',
    CANCELLED: '已取消',
  }
  return labels[normalized(status)] ?? (status || '未知')
}

function argumentsText(call: ToolCall): string {
  if (call.arguments && Object.keys(call.arguments).length) {
    try {
      return JSON.stringify(call.arguments, null, 2)
    } catch {
      return ''
    }
  }
  return call.raw_arguments ?? ''
}
</script>

<style scoped>
.tool-calls {
  display: flex;
  flex-direction: column;
  gap: var(--app-space-sm);
  margin-top: var(--app-space-sm);
}

.tool-call {
  padding: var(--app-space-sm) var(--app-space-md);
  background-color: var(--app-bg-tool);
  border: 1px solid var(--app-border-color-light);
  border-radius: var(--app-radius-md);
}

.tool-call__head {
  display: flex;
  align-items: center;
  gap: var(--app-space-xs);
  font-size: var(--app-font-size-sm);
}

.tool-call__icon {
  color: var(--app-color-primary);
}

.tool-call__name {
  font-weight: 500;
  color: var(--app-text-primary);
}

.tool-call__id {
  margin-left: auto;
  overflow: hidden;
  font-family: var(--app-font-family-mono);
  font-size: var(--app-font-size-xs);
  color: var(--app-text-muted);
  text-overflow: ellipsis;
  white-space: nowrap;
}

.tool-call__details {
  margin-top: var(--app-space-sm);
}

.tool-call__details summary {
  font-size: var(--app-font-size-xs);
  color: var(--app-text-secondary);
  cursor: pointer;
}

.tool-call__details pre {
  max-height: 240px;
  padding: var(--app-space-sm);
  margin: var(--app-space-xs) 0 0;
  overflow: auto;
  font-family: var(--app-font-family-mono);
  font-size: var(--app-font-size-xs);
  line-height: var(--app-line-height-tight);
  color: var(--app-text-regular);
  background-color: var(--app-bg-card);
  border: 1px solid var(--app-border-color-light);
  border-radius: var(--app-radius-sm);
}
</style>
