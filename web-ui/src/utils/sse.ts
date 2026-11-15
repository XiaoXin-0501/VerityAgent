/** 基于 fetch 的 SSE 客户端。
 *
 * 原生 EventSource 不能携带 Authorization 头，因此这里用 fetch +
 * ReadableStream 解析 `text/event-stream`。解析器只负责“帧 -> AgentEvent”，
 * 断线重连由 chat store 控制。
 */

import type { AgentEvent } from '@/api/types'
import { appConfig } from '@/config'
import { getToken } from '@/utils/auth'
import { ApiError } from '@/utils/request/types'

export interface RunEventStreamOptions {
  lastEventId?: string | null
  signal?: AbortSignal
}

interface SseFrame {
  id: string
  event: string
  data: string
}

function parseFrame(raw: string): SseFrame | null {
  let id = ''
  let event = 'message'
  const dataLines: string[] = []

  for (const line of raw.split(/\r?\n/)) {
    if (!line || line.startsWith(':')) continue
    const separator = line.indexOf(':')
    const field = separator === -1 ? line : line.slice(0, separator)
    let value = separator === -1 ? '' : line.slice(separator + 1)
    if (value.startsWith(' ')) value = value.slice(1)

    if (field === 'id') id = value
    else if (field === 'event') event = value
    else if (field === 'data') dataLines.push(value)
  }

  if (!dataLines.length) return null
  return { id, event, data: dataLines.join('\n') }
}

async function readErrorMessage(response: Response): Promise<ApiError> {
  let message = `请求失败（${response.status}）`
  let code: string | undefined
  try {
    const payload = (await response.json()) as {
      error?: { code?: unknown; message?: unknown }
    }
    if (typeof payload.error?.message === 'string' && payload.error.message) {
      message = payload.error.message
    }
    if (typeof payload.error?.code === 'string') {
      code = payload.error.code
    }
  } catch {
    // 非 JSON 错误体时保留状态码提示
  }
  return new ApiError(message, { status: response.status, code })
}

/** 订阅 run 事件；终态 `done` 由调用方根据 event_type 判断。 */
export async function* streamRunEvents(
  runId: string,
  options: RunEventStreamOptions = {},
): AsyncGenerator<AgentEvent> {
  const headers: Record<string, string> = { Accept: 'text/event-stream' }
  const token = getToken()
  if (token) headers.Authorization = `Bearer ${token}`
  if (options.lastEventId) headers['Last-Event-ID'] = options.lastEventId

  const response = await fetch(
    `${appConfig.apiBaseUrl}/runs/${encodeURIComponent(runId)}/events`,
    { method: 'GET', headers, signal: options.signal },
  )
  if (!response.ok) {
    throw await readErrorMessage(response)
  }
  if (!response.body) {
    throw new ApiError('浏览器不支持流式响应')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })

      while (true) {
        const separator = buffer.match(/\r?\n\r?\n/)
        if (!separator || separator.index === undefined) break
        const raw = buffer.slice(0, separator.index)
        buffer = buffer.slice(separator.index + separator[0].length)

        const frame = parseFrame(raw)
        if (!frame) continue
        try {
          yield JSON.parse(frame.data) as AgentEvent
        } catch {
          // 单帧坏数据不终止整条流
          continue
        }
      }
    }
  } finally {
    reader.releaseLock()
  }
}
