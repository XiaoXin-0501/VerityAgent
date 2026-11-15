<template>
  <div class="conversation-sidebar">
    <div class="conversation-sidebar__head">
      <div class="conversation-sidebar__title-row">
        <h2 class="conversation-sidebar__title">对话</h2>
        <div class="conversation-sidebar__actions">
          <el-tooltip content="刷新会话列表" placement="bottom">
            <el-button
              circle
              text
              :loading="loading"
              aria-label="刷新会话列表"
              @click="emit('refresh')"
            >
              <el-icon><Refresh /></el-icon>
            </el-button>
          </el-tooltip>
          <el-tooltip content="新建会话" placement="bottom">
            <el-button
              class="conversation-sidebar__new"
              type="primary"
              :loading="creating"
              aria-label="新建会话"
              @click="emit('create')"
            >
              <el-icon><Plus /></el-icon>
              <span>新建</span>
            </el-button>
          </el-tooltip>
        </div>
      </div>

      <el-input
        v-model="keyword"
        class="conversation-sidebar__search"
        placeholder="搜索会话"
        clearable
        :prefix-icon="Search"
      />
    </div>

    <div class="conversation-sidebar__list" role="list">
      <template v-if="loading && !items.length">
        <div v-for="index in 4" :key="index" class="conversation-sidebar__skeleton">
          <el-skeleton :rows="1" animated />
        </div>
      </template>

      <div v-else-if="!filtered.length" class="conversation-sidebar__empty">
        <el-icon :size="26"><ChatDotRound /></el-icon>
        <p>{{ keyword ? '没有匹配的会话' : '还没有会话' }}</p>
        <el-button v-if="!keyword" text type="primary" @click="emit('create')">
          开始第一个对话
        </el-button>
      </div>

      <button
        v-for="item in filtered"
        :key="item.id"
        type="button"
        class="conversation-item"
        :class="{ 'conversation-item--active': item.id === activeId }"
        role="listitem"
        :aria-current="item.id === activeId ? 'page' : undefined"
        @click="emit('select', item.id)"
      >
        <span class="conversation-item__icon">
          <el-icon><ChatLineRound /></el-icon>
        </span>
        <span class="conversation-item__body">
          <span class="conversation-item__title">{{ conversationTitle(item.title) }}</span>
          <span class="conversation-item__meta">
            <span>{{ formatRelative(item.last_message_at) || '暂无消息' }}</span>
            <span v-if="item.turn_count">{{ item.turn_count }} 轮</span>
          </span>
        </span>
      </button>
    </div>

    <div class="conversation-sidebar__foot">
      <span>共 {{ items.length }} 个会话</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import {
  ChatDotRound,
  ChatLineRound,
  Plus,
  Refresh,
  Search,
} from '@element-plus/icons-vue'

import type { ConversationSummary } from '@/api/types'
import { conversationTitle, formatRelative } from '@/utils/format'

const props = defineProps<{
  items: ConversationSummary[]
  activeId: string | null
  loading: boolean
  creating: boolean
}>()

const emit = defineEmits<{
  select: [id: string]
  create: []
  refresh: []
}>()

const keyword = ref('')

const filtered = computed(() => {
  const query = keyword.value.trim().toLowerCase()
  if (!query) return props.items
  return props.items.filter((item) =>
    conversationTitle(item.title).toLowerCase().includes(query),
  )
})
</script>

<style scoped>
.conversation-sidebar {
  display: flex;
  flex-direction: column;
  height: 100%;
  background-color: var(--app-bg-sidebar);
  border-right: 1px solid var(--app-border-color);
}

.conversation-sidebar__head {
  padding: var(--app-space-md);
  border-bottom: 1px solid var(--app-border-color-light);
}

.conversation-sidebar__title-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--app-space-sm);
  margin-bottom: var(--app-space-sm);
}

.conversation-sidebar__title {
  font-size: var(--app-font-size-lg);
  font-weight: 600;
  color: var(--app-text-primary);
}

.conversation-sidebar__actions {
  display: flex;
  align-items: center;
  gap: var(--app-space-2xs);
}

.conversation-sidebar__new {
  height: 32px;
  padding: 0 var(--app-space-sm);
  border-radius: var(--app-radius-pill);
}

.conversation-sidebar__new span {
  margin-left: var(--app-space-2xs);
  font-size: var(--app-font-size-sm);
}

.conversation-sidebar__search :deep(.el-input__wrapper) {
  background-color: var(--app-bg-card);
  border-radius: var(--app-radius-pill);
  box-shadow: 0 0 0 1px var(--app-border-color) inset;
}

.conversation-sidebar__list {
  flex: 1;
  overflow-y: auto;
  padding: var(--app-space-sm);
}

.conversation-sidebar__skeleton {
  padding: var(--app-space-sm);
}

.conversation-sidebar__empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--app-space-xs);
  padding: var(--app-space-2xl) var(--app-space-md);
  color: var(--app-text-secondary);
  text-align: center;
}

.conversation-item {
  display: flex;
  align-items: flex-start;
  gap: var(--app-space-sm);
  width: 100%;
  padding: var(--app-space-sm) var(--app-space-sm);
  margin-bottom: var(--app-space-2xs);
  color: inherit;
  text-align: left;
  cursor: pointer;
  background: transparent;
  border: 1px solid transparent;
  border-radius: var(--app-radius-md);
  transition:
    background-color var(--app-transition-fast),
    border-color var(--app-transition-fast);
}

.conversation-item:hover {
  background-color: var(--app-bg-hover);
}

.conversation-item--active {
  background-color: var(--app-bg-active);
  border-color: var(--app-color-primary-soft);
}

.conversation-item__icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex: 0 0 auto;
  width: 30px;
  height: 30px;
  margin-top: var(--app-space-2xs);
  color: var(--app-color-primary);
  background-color: var(--app-color-primary-soft);
  border-radius: var(--app-radius-sm);
}

.conversation-item__body {
  display: flex;
  flex-direction: column;
  gap: var(--app-space-2xs);
  min-width: 0;
}

.conversation-item__title {
  overflow: hidden;
  font-size: var(--app-font-size-base);
  color: var(--app-text-primary);
  text-overflow: ellipsis;
  white-space: nowrap;
}

.conversation-item__meta {
  display: flex;
  gap: var(--app-space-sm);
  font-size: var(--app-font-size-xs);
  color: var(--app-text-muted);
}

.conversation-sidebar__foot {
  padding: var(--app-space-sm) var(--app-space-md);
  font-size: var(--app-font-size-xs);
  color: var(--app-text-muted);
  border-top: 1px solid var(--app-border-color-light);
}
</style>
