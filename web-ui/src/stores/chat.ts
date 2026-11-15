import { cancelRunApi, getRunApi } from '@/api/modules/runs'
import { listAllMessagesApi, sendMessageApi } from '@/api/modules/conversations'
import type {
  AgentError,
  AgentEvent,
  AgentRunView,
  AgentWarning,
  ChatMessage,
  MessageStatus,
  RunStatus,
  ToolCall,
  Usage,
} from '@/api/types'
import { appConfig } from '@/config'
import { streamRunEvents } from '@/utils/sse'
import { createClientId } from '@/utils/id'
import { ApiError } from '@/utils/request/types'
import { useConversationStore } from '@/stores/conversations'
import { useUserStore } from '@/stores/user'

export interface LiveAssistantMessage {
  id: string
  content: string
  thinking: string[]
  toolCalls: ToolCall[]
  status: MessageStatus
}

export interface LiveToolCall {
  callId: string
  name: string
  status: string
  errorCode: string | null
}

function emptyUsage(): Usage {
  return {
    input_tokens: 0,
    output_tokens: 0,
    total_tokens: 0,
    cached_input_tokens: 0,
    reasoning_tokens: 0,
    source: '',
  }
}

function numberValue(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0
}

function stringValue(value: unknown): string {
  return typeof value === 'string' ? value : ''
}

function addUsage(left: Usage, right: Partial<Usage> | null | undefined): Usage {
  if (!right) return left
  const input = numberValue(right.input_tokens)
  const output = numberValue(right.output_tokens)
  return {
    input_tokens: left.input_tokens + input,
    output_tokens: left.output_tokens + output,
    total_tokens: left.total_tokens + numberValue(right.total_tokens || input + output),
    cached_input_tokens: left.cached_input_tokens + numberValue(right.cached_input_tokens),
    reasoning_tokens: left.reasoning_tokens + numberValue(right.reasoning_tokens),
    source: stringValue(right.source) || left.source,
  }
}

function localError(code: string, message: string): AgentError {
  return {
    code,
    message,
    user_visible: true,
    retryable: true,
    details: {},
    cause: null,
  }
}

function toAgentError(value: unknown, fallback: string): AgentError {
  if (value && typeof value === 'object') {
    const record = value as Record<string, unknown>
    const message = stringValue(record.message) || fallback
    return {
      code: stringValue(record.code) || 'RUN_FAILED',
      message,
      user_visible: record.user_visible !== false,
      retryable: record.retryable === true,
      details:
        record.details && typeof record.details === 'object'
          ? (record.details as Record<string, unknown>)
          : {},
      cause: typeof record.cause === 'string' ? record.cause : null,
      partial_result: record.partial_result === true,
    }
  }
  return localError('RUN_FAILED', fallback)
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms)
  })
}

async function redirectToLogin(): Promise<void> {
  const { default: router } = await import('@/router')
  const current = router.currentRoute.value
  if (current.name === 'login') return
  await router.replace({
    name: 'login',
    query: current.fullPath && current.fullPath !== '/' ? { redirect: current.fullPath } : {},
  })
}

const ACTIVE_RUN_STATUSES = new Set<RunStatus>([
  'queued',
  'running',
  'waiting_tool',
  'finalizing',
])

function runStorageKey(conversationId: string): string {
  return `${appConfig.runStoragePrefix}${conversationId}`
}

function rememberRun(conversationId: string, targetRunId: string): void {
  try {
    sessionStorage.setItem(runStorageKey(conversationId), targetRunId)
  } catch {
    // 隐私模式/配额不足时不影响本次流式
  }
}

function forgetRun(conversationId: string): void {
  try {
    sessionStorage.removeItem(runStorageKey(conversationId))
  } catch {
    // 忽略存储异常
  }
}

/** 单会话聊天状态：消息、流式增量、工具卡片、run 终态。 */
export const useChatStore = defineStore('chat', () => {
  const conversationStore = useConversationStore()

  const conversationId = ref<string | null>(null)
  const messages = ref<ChatMessage[]>([])
  const loadingMessages = ref(false)
  const streaming = ref(false)
  const cancelling = ref(false)
  const runId = ref<string | null>(null)
  const runStatus = ref<RunStatus | 'idle'>('idle')
  const liveAssistants = ref<LiveAssistantMessage[]>([])
  const liveTools = ref<Record<string, LiveToolCall>>({})
  const usage = ref<Usage>(emptyUsage())
  const error = ref<AgentError | null>(null)
  const warnings = ref<AgentWarning[]>([])
  const lastEventId = ref<string | null>(null)

  let abortController: AbortController | null = null

  const visibleWarnings = computed(() => warnings.value.filter((item) => item.user_visible))
  const isBusy = computed(() => streaming.value || cancelling.value)

  function resetStreamState(): void {
    abortController?.abort()
    abortController = null
    streaming.value = false
    cancelling.value = false
    runId.value = null
    runStatus.value = 'idle'
    liveAssistants.value = []
    liveTools.value = {}
    usage.value = emptyUsage()
    error.value = null
    warnings.value = []
    lastEventId.value = null
  }

  function resetAll(): void {
    resetStreamState()
    conversationId.value = null
    messages.value = []
    loadingMessages.value = false
  }

  function resetMessages(): void {
    resetStreamState()
    messages.value = []
    loadingMessages.value = false
  }

  async function load(targetConversationId: string): Promise<void> {
    resetStreamState()
    conversationId.value = targetConversationId
    messages.value = []
    loadingMessages.value = true
    try {
      const result = await listAllMessagesApi(targetConversationId)
      if (conversationId.value === targetConversationId) {
        messages.value = result.items
        await resumeRun(targetConversationId)
      }
    } finally {
      if (conversationId.value === targetConversationId) {
        loadingMessages.value = false
      }
    }
  }

  /** 刷新/切回会话时，从 sessionStorage 恢复仍在运行的 run。 */
  async function resumeRun(targetConversationId: string): Promise<void> {
    let storedRunId: string | null = null
    try {
      storedRunId = sessionStorage.getItem(runStorageKey(targetConversationId))
    } catch {
      return
    }
    if (!storedRunId) return
    let view: AgentRunView
    try {
      view = await getRunApi(storedRunId, { silent: true })
    } catch {
      forgetRun(targetConversationId)
      return
    }
    if (
      view.conversation_id !== targetConversationId ||
      !ACTIVE_RUN_STATUSES.has(view.status)
    ) {
      forgetRun(targetConversationId)
      return
    }
    if (conversationId.value !== targetConversationId) return

    runId.value = view.run_id
    runStatus.value = view.status
    lastEventId.value = view.last_event_id
    streaming.value = true
    if (view.assistant_message_id || view.content) {
      liveAssistants.value = [
        {
          id: view.assistant_message_id ?? `resume-${view.run_id}`,
          content: view.content,
          thinking: view.thinking.map((item) => item.text),
          toolCalls: view.tool_calls,
          status: view.assistant_status ?? 'streaming',
        },
      ]
    }
    void consumeRun(view.run_id, targetConversationId)
  }

  async function refreshMessages(target = conversationId.value): Promise<void> {
    if (!target) return
    const result = await listAllMessagesApi(target)
    if (conversationId.value === target) {
      messages.value = result.items
    }
  }

  function ensureLiveAssistant(messageId: string): LiveAssistantMessage {
    const existing = liveAssistants.value.find((item) => item.id === messageId)
    if (existing) return existing
    const created: LiveAssistantMessage = {
      id: messageId,
      content: '',
      thinking: [],
      toolCalls: [],
      status: 'streaming',
    }
    liveAssistants.value = [...liveAssistants.value, created]
    return created
  }

  function patchLiveAssistant(
    messageId: string,
    patch: Partial<LiveAssistantMessage>,
  ): void {
    liveAssistants.value = liveAssistants.value.map((item) =>
      item.id === messageId ? { ...item, ...patch } : item,
    )
  }

  /** 处理单个 SSE 事件；返回 true 表示 run 已终态，停止重连。 */
  function handleEvent(event: AgentEvent): boolean {
    const data = (event.data ?? {}) as Record<string, unknown>

    switch (event.event_type) {
      case 'run.started':
      case 'turn.started':
        runStatus.value = 'running'
        return false
      case 'assistant.started': {
        const messageId = event.message_id || stringValue(data.message_id) || `live-${event.seq}`
        ensureLiveAssistant(messageId)
        return false
      }
      case 'assistant.delta': {
        const messageId =
          event.message_id || liveAssistants.value[liveAssistants.value.length - 1]?.id
        if (!messageId) return false
        const assistant = ensureLiveAssistant(messageId)
        const text = stringValue(data.text)
        if (data.channel === 'thinking') {
          patchLiveAssistant(messageId, {
            thinking: [...assistant.thinking, text],
          })
        } else {
          patchLiveAssistant(messageId, { content: assistant.content + text })
        }
        return false
      }
      case 'assistant.tool_calls': {
        const messageId =
          event.message_id || liveAssistants.value[liveAssistants.value.length - 1]?.id
        if (!messageId) return false
        ensureLiveAssistant(messageId)
        const calls = Array.isArray(data.tool_calls) ? (data.tool_calls as ToolCall[]) : []
        patchLiveAssistant(messageId, { toolCalls: calls })
        return false
      }
      case 'assistant.completed': {
        const messageId =
          event.message_id || liveAssistants.value[liveAssistants.value.length - 1]?.id
        if (!messageId) return false
        ensureLiveAssistant(messageId)
        const thinking = Array.isArray(data.thinking)
          ? (data.thinking as unknown[]).map((item) => stringValue(item)).filter(Boolean)
          : []
        const calls = Array.isArray(data.tool_calls) ? (data.tool_calls as ToolCall[]) : []
        patchLiveAssistant(messageId, {
          content: stringValue(data.content),
          thinking,
          toolCalls: calls,
          status: 'completed',
        })
        return false
      }
      case 'tool.started': {
        const callId = event.call_id || stringValue(data.call_id)
        if (!callId) return false
        liveTools.value = {
          ...liveTools.value,
          [callId]: {
            callId,
            name: stringValue(data.name) || callId,
            status: 'running',
            errorCode: null,
          },
        }
        return false
      }
      case 'tool.result': {
        const callId = event.call_id || stringValue(data.call_id)
        const current = liveTools.value[callId]
        if (!callId || !current) return false
        liveTools.value = {
          ...liveTools.value,
          [callId]: {
            ...current,
            status: stringValue(data.status).toLowerCase() || current.status,
            errorCode: stringValue(data.error_code) || null,
          },
        }
        return false
      }
      case 'usage':
        usage.value = addUsage(usage.value, data.usage as Partial<Usage> | undefined)
        return false
      case 'warning':
        warnings.value = [...warnings.value, data as unknown as AgentWarning]
        return false
      case 'turn.failed':
        error.value = toAgentError(data, '回合执行失败')
        return false
      case 'run.failed':
        error.value = toAgentError(data, '运行失败')
        runStatus.value = 'failed'
        return true
      case 'run.cancelled':
        runStatus.value = 'cancelled'
        return true
      case 'run.completed':
        runStatus.value = 'completed'
        return false
      case 'done':
        runStatus.value = (stringValue(data.status) as RunStatus) || 'completed'
        return true
      default:
        return false
    }
  }

  function handleStreamError(streamError: unknown): boolean {
    if (streamError instanceof ApiError) {
      error.value = localError(streamError.code ?? 'STREAM_ERROR', streamError.message)
      if (streamError.status === 401) {
        useUserStore().logout()
        void redirectToLogin()
      }
      // 401/403/404 重试没有意义；其余网络错误由外层计数。
      return streamError.status === 401 || streamError.status === 403 || streamError.status === 404
    }
    error.value = localError('STREAM_ERROR', '事件流连接失败，请稍后重试')
    return false
  }

  async function consumeRun(targetRunId: string, targetConversationId: string): Promise<void> {
    const controller = new AbortController()
    abortController?.abort()
    abortController = controller
    streaming.value = true
    cancelling.value = false
    runId.value = targetRunId
    if (runStatus.value === 'idle') runStatus.value = 'running'
    let terminal = false
    let attempts = 0

    try {
      while (!controller.signal.aborted) {
        try {
          for await (const event of streamRunEvents(targetRunId, {
            lastEventId: lastEventId.value,
            signal: controller.signal,
          })) {
            if (conversationId.value !== targetConversationId) return
            lastEventId.value = `${event.run_id}:${event.seq}`
            if (handleEvent(event)) {
              terminal = true
              break
            }
          }
          if (terminal) break
        } catch (streamError) {
          if (controller.signal.aborted) return
          if (handleStreamError(streamError)) break
        }

        attempts += 1
        if (attempts > appConfig.sseMaxReconnectAttempts) {
          error.value = localError('STREAM_RECONNECT_EXHAUSTED', '事件流多次重连失败')
          break
        }
        await delay(appConfig.sseReconnectDelayMs)
      }
    } finally {
      if (abortController === controller) {
        abortController = null
        streaming.value = false
        cancelling.value = false
      }
      if (conversationId.value === targetConversationId) {
        if (terminal) {
          forgetRun(targetConversationId)
          try {
            await refreshMessages(targetConversationId)
          } catch {
            // 刷新失败保留已渲染的流式内容
          }
          liveAssistants.value = []
          liveTools.value = {}
        }
      }
    }
  }

  async function send(content: string): Promise<boolean> {
    const text = content.trim()
    if (!text || streaming.value) return false

    let targetConversationId = conversationId.value
    if (!targetConversationId) {
      const title = text.slice(0, appConfig.conversationTitleMaxLength)
      const created = await conversationStore.create(title)
      targetConversationId = created.id
      conversationId.value = created.id
      messages.value = []
    }

    const clientId = createClientId()
    const now = new Date().toISOString()
    const optimistic: ChatMessage = {
      id: `pending-${clientId}`,
      seq_no: null,
      role: 'user',
      status: 'completed',
      content: text,
      thinking: [],
      tool_calls: [],
      created_at: now,
    }
    messages.value = [...messages.value, optimistic]
    error.value = null
    warnings.value = []
    usage.value = emptyUsage()
    liveAssistants.value = []
    liveTools.value = {}

    try {
      const result = await sendMessageApi(targetConversationId, {
        client_message_id: clientId,
        content: text,
        stream: true,
      })
      streaming.value = true
      runId.value = result.run_id
      runStatus.value = 'queued'
      lastEventId.value = null
      rememberRun(targetConversationId, result.run_id)

      conversationStore.touch(targetConversationId, {
        last_message_at: now,
        turn_count:
          (conversationStore.items.find((item) => item.id === targetConversationId)?.turn_count ??
            0) + 1,
      })

      void consumeRun(result.run_id, targetConversationId)
      return true
    } catch {
      messages.value = messages.value.filter((item) => item.id !== optimistic.id)
      return false
    }
  }

  async function cancel(): Promise<void> {
    const targetRunId = runId.value
    if (!targetRunId || cancelling.value) return
    cancelling.value = true
    try {
      await cancelRunApi(targetRunId)
    } catch {
      cancelling.value = false
    }
  }

  function dispose(): void {
    abortController?.abort()
    abortController = null
  }

  return {
    conversationId,
    messages,
    loadingMessages,
    streaming,
    cancelling,
    runId,
    runStatus,
    liveAssistants,
    liveTools,
    usage,
    error,
    warnings,
    visibleWarnings,
    isBusy,
    load,
    resetAll,
    resetMessages,
    refreshMessages,
    send,
    cancel,
    dispose,
  }
})
