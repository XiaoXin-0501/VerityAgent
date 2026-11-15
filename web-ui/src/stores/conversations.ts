import { createConversationApi, listConversationsApi } from '@/api/modules/conversations'
import type { ConversationSummary } from '@/api/types'

/** 会话列表与当前会话选择；消息内容由 chat store 管理。 */
export const useConversationStore = defineStore('conversations', () => {
  const items = ref<ConversationSummary[]>([])
  const activeId = ref<string | null>(null)
  const loading = ref(false)
  const creating = ref(false)
  const loaded = ref(false)

  const activeConversation = computed(
    () => items.value.find((item) => item.id === activeId.value) ?? null,
  )

  async function fetchList(): Promise<ConversationSummary[]> {
    loading.value = true
    try {
      const result = await listConversationsApi({ limit: 50 })
      items.value = result.items
      loaded.value = true
      if (activeId.value && !items.value.some((item) => item.id === activeId.value)) {
        activeId.value = null
      }
      return result.items
    } finally {
      loading.value = false
    }
  }

  async function create(title = ''): Promise<ConversationSummary> {
    creating.value = true
    try {
      const created = await createConversationApi(title)
      const summary: ConversationSummary = {
        id: created.id,
        title: created.title,
        status: created.status,
        last_message_at: created.created_at,
        turn_count: 0,
      }
      items.value = [summary, ...items.value.filter((item) => item.id !== summary.id)]
      activeId.value = summary.id
      return summary
    } finally {
      creating.value = false
    }
  }

  function select(id: string): void {
    if (activeId.value === id) return
    activeId.value = id
  }

  function touch(id: string, patch: Partial<ConversationSummary>): void {
    const index = items.value.findIndex((item) => item.id === id)
    const current = items.value[index]
    if (!current) return
    items.value[index] = { ...current, ...patch }
    // 最近活跃的会话置顶，和后端排序保持一致。
    if (patch.last_message_at) {
      const [updated] = items.value.splice(index, 1)
      if (updated) items.value.unshift(updated)
    }
  }

  function reset(): void {
    items.value = []
    activeId.value = null
    loaded.value = false
  }

  return {
    items,
    activeId,
    activeConversation,
    loading,
    creating,
    loaded,
    fetchList,
    create,
    select,
    touch,
    reset,
  }
})
