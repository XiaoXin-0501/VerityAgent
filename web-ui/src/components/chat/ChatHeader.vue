<template>
  <header class="chat-header">
    <el-button
      class="chat-header__menu"
      circle
      text
      aria-label="打开会话列表"
      @click="emit('toggle-sidebar')"
    >
      <el-icon :size="18"><Menu /></el-icon>
    </el-button>

    <div class="chat-header__identity">
      <h1 class="chat-header__title">
        {{ conversation ? conversationTitle(conversation.title) : appConfig.appName }}
      </h1>
      <div class="chat-header__meta">
        <el-tag size="small" :type="statusTagType" effect="light">
          {{ statusLabel(statusKey) }}
        </el-tag>
        <span v-if="usage && usage.total_tokens" class="chat-header__usage">
          本次 {{ formatTokens(usage.total_tokens) }} tokens
        </span>
      </div>
    </div>

    <div class="chat-header__actions">
      <el-tooltip content="刷新消息" placement="bottom">
        <el-button circle text aria-label="刷新消息" @click="emit('refresh')">
          <el-icon><Refresh /></el-icon>
        </el-button>
      </el-tooltip>
      <el-button
        v-if="streaming || cancelling"
        class="chat-header__stop"
        type="danger"
        plain
        size="small"
        :loading="cancelling"
        @click="emit('cancel')"
      >
        <el-icon v-if="!cancelling"><VideoPause /></el-icon>
        <span>{{ cancelling ? '取消中' : '停止生成' }}</span>
      </el-button>
    </div>
  </header>
</template>

<script setup lang="ts">
import { Menu, Refresh, VideoPause } from '@element-plus/icons-vue'

import type { ConversationSummary, RunStatus, Usage } from '@/api/types'
import { appConfig } from '@/config'
import { conversationTitle, formatTokens, statusLabel } from '@/utils/format'

const props = defineProps<{
  conversation: ConversationSummary | null
  runStatus: RunStatus | 'idle'
  streaming: boolean
  cancelling: boolean
  usage: Usage | null
}>()

const emit = defineEmits<{
  cancel: []
  refresh: []
  'toggle-sidebar': []
}>()

const statusKey = computed(() => {
  if (props.runStatus !== 'idle') return props.runStatus
  return props.conversation?.status ?? 'idle'
})

const statusTagType = computed<'success' | 'warning' | 'danger' | 'info' | 'primary'>(() => {
  switch (statusKey.value) {
    case 'completed':
    case 'active':
      return 'success'
    case 'running':
    case 'queued':
    case 'waiting_tool':
    case 'finalizing':
    case 'cancelling':
      return 'warning'
    case 'failed':
    case 'expired':
    case 'conflict':
      return 'danger'
    case 'cancelled':
    case 'archived':
      return 'info'
    default:
      return 'primary'
  }
})
</script>

<style scoped>
.chat-header {
  display: flex;
  align-items: center;
  gap: var(--app-space-sm);
  flex: 0 0 auto;
  min-height: var(--app-layout-header-height);
  padding: var(--app-space-sm) var(--app-space-lg);
  background-color: var(--app-bg-card);
  border-bottom: 1px solid var(--app-border-color);
}

.chat-header__menu {
  display: none;
  color: var(--app-text-secondary);
}

.chat-header__identity {
  display: flex;
  flex-direction: column;
  gap: var(--app-space-2xs);
  min-width: 0;
}

.chat-header__title {
  overflow: hidden;
  font-size: var(--app-font-size-lg);
  font-weight: 600;
  color: var(--app-text-primary);
  text-overflow: ellipsis;
  white-space: nowrap;
}

.chat-header__meta {
  display: flex;
  align-items: center;
  gap: var(--app-space-sm);
  font-size: var(--app-font-size-xs);
  color: var(--app-text-muted);
}

.chat-header__usage {
  white-space: nowrap;
}

.chat-header__actions {
  display: flex;
  align-items: center;
  gap: var(--app-space-sm);
  margin-left: auto;
}

.chat-header__stop span {
  margin-left: var(--app-space-2xs);
}

@media (max-width: 900px) {
  .chat-header {
    padding-inline: var(--app-space-md);
  }

  .chat-header__menu {
    display: inline-flex;
  }

  .chat-header__usage {
    display: none;
  }
}
</style>
