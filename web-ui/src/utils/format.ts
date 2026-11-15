/** 展示层格式化：时间、token、状态文案。 */

export function formatDateTime(value: string | null | undefined): string {
  if (!value) return ''
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return ''
  return new Intl.DateTimeFormat('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date)
}

export function formatRelative(value: string | null | undefined): string {
  if (!value) return ''
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return ''
  const diff = Date.now() - date.getTime()
  const minute = 60_000
  const hour = 60 * minute
  const day = 24 * hour
  if (diff < minute) return '刚刚'
  if (diff < hour) return `${Math.floor(diff / minute)} 分钟前`
  if (diff < day) return `${Math.floor(diff / hour)} 小时前`
  if (diff < 7 * day) return `${Math.floor(diff / day)} 天前`
  return formatDateTime(value)
}

export function formatTokens(value: number | null | undefined): string {
  if (!value || value <= 0) return '0'
  if (value < 1_000) return String(value)
  if (value < 1_000_000) return `${(value / 1_000).toFixed(1)}k`
  return `${(value / 1_000_000).toFixed(1)}M`
}

export function statusLabel(status: string): string {
  const labels: Record<string, string> = {
    idle: '空闲',
    queued: '排队中',
    running: '生成中',
    waiting_tool: '工具执行中',
    finalizing: '收尾中',
    completed: '已完成',
    failed: '失败',
    cancelled: '已取消',
    expired: '已过期',
    conflict: '会话繁忙',
    cancelling: '取消中',
    streaming: '生成中',
    partial: '部分完成',
    active: '活跃',
    archived: '已归档',
  }
  return labels[status] ?? status
}

export function conversationTitle(title: string, fallback = '未命名会话'): string {
  return title.trim() || fallback
}
