<template>
  <div ref="scrollRef" class="message-list" @scroll.passive="handleScroll">
    <div class="message-list__inner">
      <el-alert
        v-if="error"
        class="message-list__alert"
        type="error"
        :title="error.message"
        :description="error.retryable ? '可以重新发送或稍后重试。' : undefined"
        show-icon
        :closable="false"
      />
      <el-alert
        v-for="warning in warnings"
        :key="`${warning.code}-${warning.message}`"
        class="message-list__alert"
        type="warning"
        :title="warning.message"
        show-icon
        :closable="false"
      />

      <div v-if="loading && !messages.length" class="message-list__loading">
        <div v-for="index in 3" :key="index" class="message-list__skeleton">
          <el-skeleton :rows="2" animated />
        </div>
      </div>

      <div v-else-if="!messages.length && !liveMessages.length && !streaming" class="message-list__empty">
        <div class="message-list__empty-icon">
          <el-icon :size="34"><MagicStick /></el-icon>
        </div>
        <h2 class="message-list__empty-title">开始新的对话</h2>
        <p class="message-list__empty-text">
          直接输入问题即可。助手会结合知识库、长期记忆和可用工具回答。
        </p>
      </div>

      <MessageItem
        v-for="message in messages"
        :key="message.id"
        :message="message"
      />

      <MessageItem
        v-for="message in liveMessages"
        :key="message.id"
        :message="message"
      />

      <LiveToolPanel v-if="liveToolCount" :tools="liveTools" />

      <div v-if="streaming" class="message-list__live">
        <span class="message-list__live-dot"></span>
        <span>{{ statusLabel(runStatus) || '生成中' }}</span>
      </div>
    </div>

    <transition name="fade">
      <el-button
        v-if="showJump"
        class="message-list__jump"
        circle
        type="primary"
        aria-label="回到底部"
        @click="scrollToBottom('smooth')"
      >
        <el-icon><ArrowDown /></el-icon>
      </el-button>
    </transition>
  </div>
</template>

<script setup lang="ts">
import { ArrowDown, MagicStick } from '@element-plus/icons-vue'

import type { AgentError, AgentWarning, ChatMessage, RunStatus } from '@/api/types'
import type { LiveAssistantMessage, LiveToolCall } from '@/stores/chat'
import { statusLabel } from '@/utils/format'
import LiveToolPanel from '@/components/chat/LiveToolPanel.vue'
import MessageItem from '@/components/chat/MessageItem.vue'

const props = defineProps<{
  messages: ChatMessage[]
  liveAssistants: LiveAssistantMessage[]
  liveTools: Record<string, LiveToolCall>
  loading: boolean
  streaming: boolean
  error: AgentError | null
  warnings: AgentWarning[]
  runStatus: RunStatus | 'idle'
}>()

const scrollRef = ref<HTMLElement | null>(null)
const atBottom = ref(true)
const showJump = ref(false)

const liveMessages = computed<ChatMessage[]>(() =>
  props.liveAssistants.map((item) => ({
    id: item.id,
    seq_no: null,
    role: 'assistant',
    status: item.status,
    content: item.content,
    thinking: item.thinking.map((text) => ({ text })),
    tool_calls: item.toolCalls,
    created_at: null,
  })),
)

const liveToolCount = computed(() => Object.keys(props.liveTools).length)

const scrollSignature = computed(
  () =>
    props.messages.length * 1_000 +
    props.liveAssistants.length * 100 +
    props.liveAssistants.reduce(
      (total, item) => total + item.content.length + item.toolCalls.length,
      0,
    ),
)

function handleScroll(): void {
  const element = scrollRef.value
  if (!element) return
  const distance = element.scrollHeight - element.scrollTop - element.clientHeight
  atBottom.value = distance < 120
  showJump.value = !atBottom.value
}

function scrollToBottom(behavior: ScrollBehavior = 'auto'): void {
  const element = scrollRef.value
  if (!element) return
  element.scrollTo({ top: element.scrollHeight, behavior })
  atBottom.value = true
  showJump.value = false
}

watch(scrollSignature, async () => {
  if (!atBottom.value) return
  await nextTick()
  scrollToBottom()
})

watch(
  () => props.loading,
  async (loading) => {
    if (loading) return
    await nextTick()
    scrollToBottom()
  },
)

onMounted(() => {
  scrollToBottom()
})
</script>

<style scoped>
.message-list {
  position: relative;
  flex: 1;
  overflow-y: auto;
  background-color: var(--app-bg-chat);
}

.message-list__inner {
  display: flex;
  flex-direction: column;
  padding: var(--app-space-lg) var(--app-space-lg) var(--app-space-xl);
}

.message-list__alert {
  width: 100%;
  max-width: var(--app-chat-max-width);
  margin: 0 auto var(--app-space-md);
}

.message-list__loading {
  display: flex;
  flex-direction: column;
  gap: var(--app-space-lg);
  max-width: var(--app-chat-max-width);
  margin: 0 auto;
}

.message-list__skeleton {
  padding: var(--app-space-md);
  background-color: var(--app-bg-card);
  border: 1px solid var(--app-border-color-light);
  border-radius: var(--app-radius-lg);
}

.message-list__empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: var(--app-space-sm);
  min-height: 46vh;
  padding: var(--app-space-2xl);
  text-align: center;
}

.message-list__empty-icon {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 72px;
  height: 72px;
  margin-bottom: var(--app-space-sm);
  color: var(--app-color-primary);
  background-color: var(--app-color-primary-soft);
  border-radius: var(--app-radius-xl);
}

.message-list__empty-title {
  font-size: var(--app-font-size-xl);
  font-weight: 600;
  color: var(--app-text-primary);
}

.message-list__empty-text {
  max-width: 440px;
  font-size: var(--app-font-size-base);
  line-height: var(--app-line-height-base);
  color: var(--app-text-secondary);
}

.message-list__live {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--app-space-xs);
  padding: var(--app-space-sm);
  font-size: var(--app-font-size-xs);
  color: var(--app-text-muted);
}

.message-list__live-dot {
  width: 7px;
  height: 7px;
  background-color: var(--app-color-success);
  border-radius: var(--app-radius-pill);
  animation: pulse 1.4s infinite ease-in-out;
}

@keyframes pulse {
  0%,
  100% {
    opacity: 0.35;
    transform: scale(0.9);
  }
  50% {
    opacity: 1;
    transform: scale(1.1);
  }
}

.message-list__jump {
  position: sticky;
  bottom: var(--app-space-lg);
  left: 50%;
  display: flex;
  margin: 0 auto;
  box-shadow: var(--app-shadow-float);
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity var(--app-transition-fast);
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

@media (max-width: 900px) {
  .message-list__inner {
    padding: var(--app-space-md) var(--app-space-md) var(--app-space-lg);
  }
}
</style>
