<template>
  <div class="conversation-sidebar">
    <div class="conversation-sidebar__brand">
      <BrandMark />
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
    </div>

    <div class="conversation-sidebar__head">
      <el-button
        class="sidebar-action conversation-sidebar__new"
        :loading="creating"
        aria-label="新建会话"
        @click="emit('create')"
      >
        <el-icon><Plus /></el-icon>
        <span>新建对话</span>
      </el-button>

      <el-button
        v-if="isAdmin"
        class="sidebar-action conversation-sidebar__admin"
        @click="router.push({ name: 'knowledge-bases' })"
      >
        <el-icon><Collection /></el-icon>
        <span>知识库管理</span>
      </el-button>

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

      <div
        v-for="group in groups"
        :key="group.label"
        class="conversation-group"
      >
        <div class="conversation-group__label">{{ group.label }}</div>
        <button
          v-for="item in group.items"
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
    </div>

    <div class="conversation-sidebar__foot">
      <div class="conversation-sidebar__user">
        <UserMenu />
        <ThemeToggle />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import {
  ChatDotRound,
  ChatLineRound,
  Collection,
  Plus,
  Refresh,
  Search,
} from '@element-plus/icons-vue'

import BrandMark from '@/components/layout/BrandMark.vue'
import UserMenu from '@/components/layout/UserMenu.vue'
import ThemeToggle from '@/components/common/ThemeToggle.vue'
import type { ConversationSummary } from '@/api/types'
import { useUserStore } from '@/stores/user'
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
const userStore = useUserStore()
const router = useRouter()
const isAdmin = computed(() => userStore.userInfo?.role === 'admin')

const filtered = computed(() => {
  const query = keyword.value.trim().toLowerCase()
  if (!query) return props.items
  return props.items.filter((item) =>
    conversationTitle(item.title).toLowerCase().includes(query),
  )
})

interface ConversationGroup {
  label: string
  items: ConversationSummary[]
}

/** 按最近消息时间分组，贴近主流 Agent 的历史会话排版。 */
const groups = computed<ConversationGroup[]>(() => {
  const today: ConversationSummary[] = []
  const yesterday: ConversationSummary[] = []
  const lastWeek: ConversationSummary[] = []
  const earlier: ConversationSummary[] = []
  const now = new Date()
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime()

  for (const item of filtered.value) {
    const time = item.last_message_at ? new Date(item.last_message_at).getTime() : 0
    if (!time) {
      earlier.push(item)
    } else if (time >= startOfToday) {
      today.push(item)
    } else if (time >= startOfToday - 86_400_000) {
      yesterday.push(item)
    } else if (time >= startOfToday - 7 * 86_400_000) {
      lastWeek.push(item)
    } else {
      earlier.push(item)
    }
  }

  return [
    { label: '今天', items: today },
    { label: '昨天', items: yesterday },
    { label: '近 7 天', items: lastWeek },
    { label: '更早', items: earlier },
  ].filter((group) => group.items.length > 0)
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

.conversation-sidebar__brand {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--app-space-sm);
  padding: var(--app-space-md) var(--app-space-md) var(--app-space-sm);
}

.conversation-sidebar__head {
  padding: var(--app-space-xs) var(--app-space-md) var(--app-space-sm);
}

.sidebar-action.el-button {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--app-space-xs);
  width: 100%;
  height: 38px;
  /* Element 的 `.el-button + .el-button` 会加 12px 左边距，导致两个按钮不齐。 */
  margin: 0 0 var(--app-space-sm);
  padding: 0 var(--app-space-md);
  font-size: var(--app-font-size-base);
  font-weight: 400;
  line-height: 1;
  color: var(--app-text-primary);
  background-color: var(--app-bg-card);
  border: 1px solid var(--app-border-color);
  border-radius: var(--app-radius-md);
  box-shadow: var(--app-shadow-bubble);
}

.sidebar-action.el-button:hover {
  color: var(--app-color-primary);
  background-color: var(--app-color-primary-soft);
  border-color: var(--app-color-primary-soft);
}

.sidebar-action :deep(.el-icon) {
  position: static;
  margin: 0;
  color: var(--app-color-primary);
  transform: none;
}

.sidebar-action :deep(span) {
  display: inline-flex;
  align-items: center;
  gap: var(--app-space-xs);
  margin-left: 0;
  line-height: 1;
}

.conversation-sidebar__search :deep(.el-input__wrapper) {
  background-color: var(--app-bg-card);
  border-radius: var(--app-radius-md);
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

.conversation-group + .conversation-group {
  margin-top: var(--app-space-md);
}

.conversation-group__label {
  padding: 0 var(--app-space-sm) var(--app-space-2xs);
  font-size: var(--app-font-size-xs);
  font-weight: 500;
  color: var(--app-text-muted);
  letter-spacing: 0.02em;
}

.conversation-item {
  position: relative;
  display: flex;
  align-items: flex-start;
  gap: var(--app-space-sm);
  width: 100%;
  padding: var(--app-space-sm) var(--app-space-sm);
  margin-bottom: var(--app-space-2xs);
  overflow: hidden;
  color: inherit;
  text-align: left;
  cursor: pointer;
  background: transparent;
  border: 1px solid transparent;
  border-radius: var(--app-radius-md);
  transition:
    background-color var(--app-transition-fast),
    border-color var(--app-transition-fast),
    transform var(--app-transition-fast);
}

.conversation-item:hover {
  background-color: var(--app-bg-hover);
  transform: translateX(1px);
}

.conversation-item--active {
  background-color: var(--app-bg-active);
  border-color: var(--app-color-primary-soft);
}

.conversation-item--active::before {
  position: absolute;
  top: 50%;
  left: 0;
  width: 3px;
  height: 60%;
  content: '';
  background-color: var(--app-color-primary);
  border-radius: var(--app-radius-pill);
  transform: translateY(-50%);
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
  display: flex;
  flex-direction: column;
  gap: var(--app-space-2xs);
  padding: var(--app-space-sm);
  border-top: 1px solid var(--app-border-color-light);
}

.conversation-sidebar__user {
  display: flex;
  align-items: center;
  gap: var(--app-space-2xs);
}

.conversation-sidebar__user :deep(.user-menu-root) {
  flex: 1;
  min-width: 0;
}
</style>
