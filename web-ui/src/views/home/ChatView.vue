<template>
  <section class="chat-page">
    <aside class="chat-page__sidebar" :class="{ 'chat-page__sidebar--open': sidebarOpen }">
      <ConversationSidebar
        :items="conversationStore.items"
        :active-id="conversationStore.activeId"
        :loading="conversationStore.loading"
        :creating="conversationStore.creating"
        @select="handleSelect"
        @create="handleCreate"
        @refresh="handleRefresh"
      />
    </aside>

    <div class="chat-page__main">
      <ChatHeader
        :conversation="conversationStore.activeConversation"
        :run-status="chat.runStatus"
        :usage="chat.usage"
        @refresh="handleRefresh"
        @toggle-sidebar="sidebarOpen = !sidebarOpen"
      />

      <MessageList
        :messages="chat.messages"
        :live-assistants="chat.liveAssistants"
        :live-tools="chat.liveTools"
        :loading="chat.loadingMessages"
        :streaming="chat.streaming"
        :error="chat.error"
        :warnings="chat.visibleWarnings"
        :run-status="chat.runStatus"
        @suggest="draft = $event"
      />

      <ChatComposer
        v-model="draft"
        :disabled="chat.loadingMessages"
        :streaming="chat.streaming"
        :cancelling="chat.cancelling"
        :sending="sending"
        :placeholder="composerPlaceholder"
        @send="handleSend"
        @cancel="chat.cancel()"
      />
    </div>

    <div
      v-if="sidebarOpen"
      class="chat-page__overlay"
      aria-hidden="true"
      @click="sidebarOpen = false"
    ></div>
  </section>
</template>

<script setup lang="ts">
import ChatHeader from '@/components/chat/ChatHeader.vue'
import ChatComposer from '@/components/chat/ChatComposer.vue'
import ConversationSidebar from '@/components/chat/ConversationSidebar.vue'
import MessageList from '@/components/chat/MessageList.vue'
import { useChatStore } from '@/stores/chat'
import { useConversationStore } from '@/stores/conversations'

const conversationStore = useConversationStore()
const chat = useChatStore()

const draft = ref('')
const sending = ref(false)
const sidebarOpen = ref(false)

const composerPlaceholder = computed(() =>
  conversationStore.activeConversation
    ? '输入消息，Enter 发送，Shift + Enter 换行'
    : '输入问题，发送后会自动创建会话',
)

async function loadConversation(id: string): Promise<void> {
  try {
    await chat.load(id)
  } catch {
    // 错误提示由请求层统一处理
  }
}

async function handleSelect(id: string): Promise<void> {
  sidebarOpen.value = false
  conversationStore.select(id)
  await loadConversation(id)
}

function handleCreate(): void {
  sidebarOpen.value = false
  // 新建对话只进入前端草稿态；首条消息发送时才真正创建后端会话。
  conversationStore.select(null)
  chat.resetAll()
  draft.value = ''
}

async function handleRefresh(): Promise<void> {
  try {
    await conversationStore.fetchList()
    const targetId = conversationStore.activeId
    if (!targetId) {
      // 保持当前草稿态；若原会话已不可见则清空旧消息。
      if (chat.conversationId) chat.resetAll()
      return
    }
    await loadConversation(targetId)
  } catch {
    // 错误提示由请求层统一处理
  }
}

async function handleSend(): Promise<void> {
  if (sending.value || chat.streaming) return
  const text = draft.value.trim()
  if (!text) return

  sending.value = true
  try {
    const accepted = await chat.send(text)
    if (accepted) draft.value = ''
  } catch {
    // 错误提示由请求层统一处理
  } finally {
    sending.value = false
  }
}

onMounted(async () => {
  try {
    const items = await conversationStore.fetchList()
    const preferred =
      items.find((item) => item.id === conversationStore.activeId) ?? items[0]
    if (preferred) {
      conversationStore.select(preferred.id)
      await loadConversation(preferred.id)
    }
  } catch {
    // 错误提示由请求层统一处理
  }
})

onBeforeUnmount(() => {
  chat.dispose()
})
</script>

<style scoped>
.chat-page {
  position: relative;
  display: flex;
  flex: 1;
  width: 100%;
  min-height: 0;
  height: 100%;
  overflow: hidden;
  background: var(--app-chat-background);
}

.chat-page__sidebar {
  flex: 0 0 var(--app-sidebar-width);
  width: var(--app-sidebar-width);
  height: 100%;
  overflow: hidden;
}

.chat-page__main {
  display: flex;
  flex: 1;
  flex-direction: column;
  min-width: 0;
  height: 100%;
}

.chat-page__overlay {
  display: none;
}

@media (max-width: 900px) {
  .chat-page__sidebar {
    position: fixed;
    top: 0;
    bottom: 0;
    left: 0;
    z-index: 30;
    width: min(84vw, var(--app-sidebar-width));
    transform: translateX(-100%);
    transition: transform var(--app-transition-base);
    box-shadow: var(--app-shadow-float);
  }

  .chat-page__sidebar--open {
    transform: translateX(0);
  }

  .chat-page__overlay {
    position: fixed;
    inset: 0;
    z-index: 20;
    display: block;
    background-color: var(--app-bg-overlay);
  }
}
</style>
