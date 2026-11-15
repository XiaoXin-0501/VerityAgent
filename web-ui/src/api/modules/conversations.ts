import type {
  ChatMessage,
  ConversationListResult,
  ConversationSummary,
  CreateConversationResult,
  MessageListResult,
  SendMessageParams,
  SendMessageResult,
} from '@/api/types'
import { appConfig } from '@/config'
import http from '@/utils/request'

/** 创建当前用户的会话。 */
export function createConversationApi(title = ''): Promise<CreateConversationResult> {
  return http.post<CreateConversationResult>('/conversations', { title })
}

/** 按最近消息时间倒序分页列出会话。 */
export function listConversationsApi(params?: {
  limit?: number
  cursor?: string | null
}): Promise<ConversationListResult> {
  return http.get<ConversationListResult>('/conversations', { params })
}

/** 发送消息；stream=true 时返回 run 句柄，事件通过 SSE 订阅。 */
export function sendMessageApi(
  conversationId: string,
  params: SendMessageParams,
): Promise<SendMessageResult> {
  return http.post<SendMessageResult>(`/conversations/${conversationId}/messages`, params)
}

/**
 * 拉取会话消息。
 *
 * 后端 cursor 是 `after_seq`，只支持向后翻页；一次对话可能超过一页，
 * 因此按页循环直到 `next_cursor=null`，并用 maxMessagePages 防止异常数据死循环。
 */
export async function listAllMessagesApi(
  conversationId: string,
  options: { pageSize?: number; maxPages?: number } = {},
): Promise<MessageListResult> {
  const pageSize = options.pageSize ?? appConfig.messagePageSize
  const maxPages = options.maxPages ?? appConfig.maxMessagePages
  const items: ChatMessage[] = []
  let cursor: string | null = null

  for (let page = 0; page < maxPages; page += 1) {
    const result: MessageListResult = await http.get<MessageListResult>(
      `/conversations/${conversationId}/messages`,
      { params: { limit: pageSize, cursor: cursor ?? undefined } },
    )
    items.push(...result.items)
    cursor = result.next_cursor
    if (!cursor) {
      return { items, next_cursor: null }
    }
  }

  return { items, next_cursor: cursor }
}

/** 本地插入/替换会话列表项时的轻量投影。 */
export function toConversationSummary(
  item: CreateConversationResult,
  previous?: ConversationSummary,
): ConversationSummary {
  return {
    id: item.id,
    title: item.title,
    status: item.status,
    last_message_at: previous?.last_message_at ?? item.created_at,
    turn_count: previous?.turn_count ?? 0,
  }
}
