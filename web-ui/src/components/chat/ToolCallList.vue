<template>
  <div v-if="calls.length" class="tool-calls">
    <div v-for="call in calls" :key="call.call_id" class="tool-call">
      <el-icon class="tool-call__icon"><Tools /></el-icon>
      <span class="tool-call__name">{{ toolLabel(call) }}</span>
      <el-tag size="small" :type="tagType(effectiveStatus(call))" effect="plain">
        {{ statusText(effectiveStatus(call)) }}
      </el-tag>
      <span class="tool-call__summary">{{ resultSummary(call) }}</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { Tools } from '@element-plus/icons-vue'

import type { ToolCall } from '@/api/types'

defineProps<{ calls: ToolCall[] }>()

/** 内部工具标识 → 面向用户的中性名称；绝不直接展示调用名/类名。 */
const TOOL_LABELS: Record<string, string> = {
  search_kb: '知识库检索',
  recall_round: '历史对话回忆',
  resolve_chunk_conflict: '内容冲突处理',
}

const SUCCESS_SUMMARIES: Record<string, string> = {
  search_kb: '已获得相关知识片段',
  recall_round: '已获得历史对话内容',
  resolve_chunk_conflict: '已获得冲突处理结果',
}

function effectiveStatus(call: ToolCall): string | undefined {
  return call.result_status || call.status
}

/** 内部工具标识一律映射为中性名称；未登记工具统一显示“工具”。 */
function toolLabel(call: ToolCall): string {
  const name = call.name?.trim()
  return TOOL_LABELS[name] ?? '工具'
}

function resultSummary(call: ToolCall): string {
  switch (normalized(effectiveStatus(call))) {
    case 'PENDING':
    case 'RUNNING':
      return '执行中'
    case 'FAILED':
      return '未获得结果'
    case 'CANCELLED':
      return '已取消'
    default:
      return SUCCESS_SUMMARIES[call.name] ?? '已获得结果'
  }
}

function normalized(status: string | undefined): string {
  return (status ?? '').toUpperCase()
}

function tagType(status: string | undefined): 'success' | 'warning' | 'danger' | 'info' {
  switch (normalized(status)) {
    case 'SUCCEEDED':
    case 'COMPLETED':
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
    COMPLETED: '成功',
    FAILED: '失败',
    CANCELLED: '已取消',
  }
  return labels[normalized(status)] ?? (status || '未知')
}
</script>

<style scoped>
.tool-calls {
  display: flex;
  flex-direction: column;
  gap: var(--app-space-xs);
  margin-top: var(--app-space-xs);
}

.tool-call {
  display: flex;
  align-items: center;
  gap: var(--app-space-xs);
  padding: var(--app-space-sm) var(--app-space-md);
  font-size: var(--app-font-size-sm);
  background-color: var(--app-bg-card);
  border: 1px solid var(--app-border-color-light);
  border-radius: var(--app-radius-sm);
}

.tool-call__icon {
  color: var(--app-color-primary);
}

.tool-call__name {
  font-weight: 500;
  color: var(--app-text-primary);
}

.tool-call__summary {
  margin-left: auto;
  font-size: var(--app-font-size-xs);
  color: var(--app-text-muted);
}
</style>
