import type { Artifact, ChatMessage, MessageStatus, ToolCall } from '@/api/types'

/** 一轮内的一个助手步骤（对应一次 LLM 调用）。 */
export interface AssistantStep {
  id: string
  content: string
  thinking: string[]
  toolCalls: ToolCall[]
  status: MessageStatus
}

/** 前端展示用的“一个助手回复”，聚合同一轮的所有步骤。 */
export interface AssistantTurn {
  id: string
  roundId: string | null
  status: MessageStatus
  /** 最终回答；中间步骤的叙述不混入最终回答。 */
  content: string
  /** 清洗后的思考文本；多步骤合并为一段，默认收起展示。 */
  thinking: string[]
  /** 中间步骤产生的可见文本（工具调用前的说明等）。 */
  processTexts: string[]
  /** 合并后的工具调用，含结果状态与结果正文。 */
  toolCalls: ToolCall[]
  artifacts: Artifact[]
  createdAt: string | null
  /** 该轮包含的助手步骤数。 */
  stepCount: number
  /** 当前仍有步骤在流式思考，思考内容尚未完整。 */
  thinkingPending: boolean
}

export type ChatDisplayItem =
  | { kind: 'user'; key: string; message: ChatMessage }
  | { kind: 'assistant'; key: string; turn: AssistantTurn }

/**
 * 把持久化消息折叠成“用户消息 + 每轮一个助手回复”。
 *
 * 后端为审计与工具协议保留了同一轮内的多条 assistant/tool 消息；它们共享
 * ``round_id``。展示层只保留最后一条有正文的 assistant 作为回答，其余
 * thinking/工具调用收进同一个可折叠过程区，避免每步都像一个系统回复。
 * 旧数据没有 ``round_id`` 时，按连续的 assistant/tool 序列兜底分组。
 */
export function buildDisplayItems(messages: ChatMessage[]): ChatDisplayItem[] {
  const items: ChatDisplayItem[] = []
  let group: ChatMessage[] = []
  let groupRoundId: string | null = null

  const flush = (): void => {
    if (!group.length) return
    const turn = buildTurn(group)
    items.push({ kind: 'assistant', key: `turn-${turn.id}`, turn })
    group = []
    groupRoundId = null
  }

  for (const message of messages) {
    if (message.role === 'user') {
      flush()
      items.push({ kind: 'user', key: `user-${message.id}`, message })
      continue
    }
    if (message.role !== 'assistant' && message.role !== 'tool') continue

    const roundId = message.round_id ?? null
    if (group.length) {
      // 新轮次开始：round_id 明确变化时切组；旧数据（null）沿用当前组。
      // 从旧数据（null）切到有新 round_id 的消息，视为下一轮。
      if (roundId !== null && roundId !== groupRoundId) {
        flush()
      }
    }
    group.push(message)
    groupRoundId = roundId ?? groupRoundId
  }
  flush()
  return items
}

/** 把实时流式步骤折叠成一个助手回复。 */
export function buildLiveTurn(
  steps: AssistantStep[],
  liveTools: Record<string, { status: string; errorCode: string | null }>,
  fallbackStatus: MessageStatus = 'streaming',
): AssistantTurn {
  const toolCalls = steps
    .flatMap((step) => step.toolCalls ?? [])
    .map((call) => {
      const live = liveTools[call.call_id]
      return displayCall(call, live?.status ?? call.result_status)
    })
  // 流式步骤的 thinking 仍在增长；只展示已形成完整句子的稳定前缀，
  // 未结束的尾巴先缓冲，避免半句话反复出现/消失。
  const stableSteps = steps.map((step) => {
    if (fallbackStatus !== 'streaming' || step.status !== 'streaming') {
      return step
    }
    return {
      ...step,
      thinking: [stableThinkingPrefix(step.thinking.join(''))],
    }
  })
  const pending =
    fallbackStatus === 'streaming' &&
    (!steps.length || steps.some((step) => step.status === 'streaming'))
  return buildTurnFromSteps(
    stableSteps,
    toolCalls,
    steps[steps.length - 1]?.status ?? fallbackStatus,
    [],
    null,
    null,
    pending,
  )
}

function buildTurn(messages: ChatMessage[]): AssistantTurn {
  const assistants = messages.filter((item) => item.role === 'assistant')
  const tools = messages.filter((item) => item.role === 'tool')
  const toolById = new Map(
    tools
      .filter((item) => item.tool_call_id)
      .map((item) => [item.tool_call_id as string, item]),
  )

  const toolCalls: ToolCall[] = []
  const seen = new Set<string>()
  for (const assistant of assistants) {
    for (const call of assistant.tool_calls ?? []) {
      if (seen.has(call.call_id)) continue
      seen.add(call.call_id)
      const result = toolById.get(call.call_id)
      // 只保留展示所需字段：不把参数、原始参数带进展示模型。
      toolCalls.push(displayCall(call, result?.status))
    }
  }
  // 历史数据可能缺少 assistant.tool_calls；用 ToolMessage 兜底生成调用记录。
  for (const tool of tools) {
    const callId = tool.tool_call_id
    if (!callId || seen.has(callId)) continue
    seen.add(callId)
    toolCalls.push(
      displayCall(
        { call_id: callId, name: tool.name || '', status: tool.status },
        tool.status,
      ),
    )
  }

  const steps: AssistantStep[] = assistants.map((item) => ({
    id: item.id,
    content: item.content,
    thinking: (item.thinking ?? []).map((thinking) => thinking.text),
    toolCalls: item.tool_calls ?? [],
    status: item.status,
  }))
  const final = assistants[assistants.length - 1]
  return buildTurnFromSteps(
    steps,
    toolCalls,
    final?.status ?? (toolCalls.some((call) => call.result_status === 'failed') ? 'failed' : 'completed'),
    final?.artifacts ?? [],
    final?.created_at ?? messages[0]?.created_at ?? null,
    messages[0]?.round_id ?? null,
    false,
  )
}

/** 构造展示用工具调用：只保留调用标识、展示映射所需名称和状态。 */
function displayCall(
  call: Pick<ToolCall, 'call_id' | 'name' | 'status'>,
  resultStatus?: string,
): ToolCall {
  return {
    call_id: call.call_id,
    name: call.name,
    status: call.status,
    result_status: resultStatus,
  }
}

// 思考原文里的内部实现细节：工具标识、参数名、schema/JSON、调用 id、
// 结果片段等。展示层按句过滤，避免把工具协议原文带给用户。
const TOOL_DETAIL_PATTERNS: RegExp[] = [
  /\b(search_kb|recall_round|resolve_chunk_conflict)\b/i,
  /\b(search|retrieve|look up)\b[^.]{0,40}\b(knowledge base|kb)\b/i,
  /\b(knowledge base|kb)\s*(id|ids|identifier)s?\b/i,
  /\b(kb_ids?|top_k|args?|params?|raw_arguments|minitems|maxitems|schema|json|uuid|validation|placeholder)\b/i,
  /\barguments?\s*[:=]/i,
  /\b(tool|function)\s*call\b/i,
  /\b(try|attempt|make|issue|invoke|execute)\b[^.]{0,25}\bcall(ing|s)?\b/i,
  /\bcall(ing|s)?\b[^.]{0,30}\b(tool|knowledge base|kb|schema|query|empty|placeholder|error|id)\b/i,
  /\b(it'?s|it is|this is)\s+required\b/i,
  /\bno default\b|\bdefault listed\b/i,
  /\bempty\s+(array|list)\b|\bauto-?bind(s|ing)?\b/i,
  /\bthat risks nothing\b|\bworth trying\b|\bno harm in trying\b/i,
  /\bids?\b/i,
  /\bcall_[a-z0-9]+/i,
  /\breq_[a-z0-9]+/i,
  /\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b/i,
  /工具(名称|调用|参数|接口|结果|需要|要求|返回|失败|不可用|执行)|调用|必填|默认值|空数组|空列表|报错|自动绑定|绑定知识库|参数校验|字段校验|重试|占位|检索结果|返回结果|结果片段/i,
]

// 模型思考里的规划/风格元信息（“回答要简短”“无需工具”）对用户没有
// 价值；这里只匹配明确的元信息，避免误伤实质医学推理。
const STYLE_META_PATTERN =
  /\b(respond|reply)\s+(in|briefly|shortly|concisely)\b|\bkeep it (short|brief)\b|\bbe brief\b|\bno need (to use|for) tools?\b|\bno tool needed\b|\boffer help\b|\bhelp the user\b|保持简短|无需工具|不需要工具|帮助用户|直接回答/i

const USER_META_PATTERN =
  /^(the user|user|用户)[^。.!?！？]{0,60}(say(s|ing)?|ask(s|ing)?|want(s|ing)?|question(s|ing)?|问|说|想|输入|发送)/i

const HIDDEN_THINKING_NOTICE = '（部分思考涉及内部工具执行细节，已隐藏）'
// 思考默认完整保留；只有异常超长时才从尾部截断，避免“无谓残缺”。
const MAX_THINKING_CHARS = 12_000

// 稳定边界：中文句号/感叹号/问号，或英文句末标点后跟空白/结束，或换行。
const STABLE_BOUNDARY_RE = /[。！？]|[.!?](?=\s|$)|\n/g

/** 只取已经形成完整句子的前缀；最后半句留到后续 delta 或完成时再展示。 */
export function stableThinkingPrefix(text: string): string {
  let end = 0
  for (const match of text.matchAll(STABLE_BOUNDARY_RE)) {
    const index = (match.index ?? 0) + match[0].length
    if (index > end) end = index
  }
  return end > 0 ? text.slice(0, end) : ''
}

/** 清洗思考文本：保留有价值的推理，过滤工具协议细节和“用户说…”复述。 */
export function sanitizeThinking(texts: string[], hasTools: boolean): string[] {
  const cleaned = texts
    .map((text) => sanitizeThinkingText(text))
    .filter(Boolean)
  const total = cleaned.reduce((sum, item) => sum + item.length, 0)
  if (total > MAX_THINKING_CHARS) {
    const joined = cleaned.join('\n')
    return [`${joined.slice(0, MAX_THINKING_CHARS)}…`]
  }
  // 工具轮的思考可能全部是参数/重试等内部细节；明确说明已隐藏，
  // 而不是让用户以为“思考只剩工具调用”。
  if (!cleaned.length && hasTools && texts.some((text) => text.trim())) {
    return [HIDDEN_THINKING_NOTICE]
  }
  // 合并成一段连续文本，避免多个步骤各自成块、看起来像多段系统消息。
  return cleaned.length ? [cleaned.join('\n')] : []
}

function sanitizeThinkingText(text: string): string {
  const sentences = text
    // 中文句号后通常没有空格，必须单独切分；英文只在“句号+空格”处切分。
    .split(/(?<=[。！？!?])\s*|(?<=\.)\s+|\n+/)
    .map((item) => stripPlanningPrefix(item.trim()))
    .filter(Boolean)
    .filter(
      (item) =>
        !TOOL_DETAIL_PATTERNS.some((pattern) => pattern.test(item)) &&
        !isUserMeta(item) &&
        !STYLE_META_PATTERN.test(item),
    )
  // 中文标点后不需要空格；英文句间保留一个空格可读性更好。
  const joiner = /[一-鿿]/.test(text) ? '' : ' '
  return sentences.join(joiner).trim()
}

/** 去掉“Hmm/让我/我应该”等规划前缀，保留其后的实质推理。 */
function stripPlanningPrefix(text: string): string {
  const stripped = text
    .replace(
      /^(hmm|okay|ok|wait|actually)[,.!]?\s*|^(let me|i should|i need to|i'll|i will|maybe|perhaps|possibly|now i)\s+/i,
      '',
    )
    .replace(/^(嗯|等等|让我|我应该|我需要|接下来我)[，,。.!]?\s*/, '')
    .trim()
  return stripped ? stripped.charAt(0).toUpperCase() + stripped.slice(1) : ''
}

function isUserMeta(text: string): boolean {
  if (USER_META_PATTERN.test(text)) return true
  // 流式刚开始时可能只有 "The user"/"用户" 片段，先隐藏等完整句子。
  return /^(the user|user|用户)/i.test(text) && text.length <= 30
}

function buildTurnFromSteps(
  steps: AssistantStep[],
  toolCalls: ToolCall[],
  status: MessageStatus,
  artifacts: Artifact[] = [],
  createdAt: string | null = null,
  roundId: string | null = null,
  thinkingPending = false,
): AssistantTurn {
  // 一轮的最后一步才是最终回答；之前的步骤内容属于过程叙述。
  const final = steps[steps.length - 1]
  const content = final?.content ?? ''
  // 工具执行轮的思考仍保留，但必须过滤工具名、参数、schema、结果片段等
  // 内部实现细节；中间步骤正文可能是在描述工具调用，仍不展示。
  const hasTools = toolCalls.length > 0
  const rawThinking = steps.flatMap((step) => step.thinking).filter(Boolean)
  return {
    id: steps[0]?.id ?? 'live-turn',
    roundId,
    status,
    content,
    thinking: sanitizeThinking(rawThinking, hasTools && !thinkingPending),
    processTexts: hasTools
      ? []
      : steps
          .slice(0, -1)
          .map((step) => step.content)
          .filter(Boolean),
    toolCalls,
    artifacts,
    createdAt,
    stepCount: steps.length,
    thinkingPending,
  }
}
