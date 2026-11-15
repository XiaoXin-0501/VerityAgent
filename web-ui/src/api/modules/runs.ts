import type { AgentRunView, CancelRunResult } from '@/api/types'
import http from '@/utils/request'

/** 读取 run 状态/快照；只能读取自己会话的 run。 */
export function getRunApi(
  runId: string,
  options: { silent?: boolean } = {},
): Promise<AgentRunView> {
  return http.get<AgentRunView>(`/runs/${encodeURIComponent(runId)}`, {
    silent: options.silent,
  })
}

/** 幂等取消 run；终态 run 也返回 202。 */
export function cancelRunApi(runId: string, reason = 'user'): Promise<CancelRunResult> {
  return http.post<CancelRunResult>(`/runs/${encodeURIComponent(runId)}/cancel`, { reason })
}
