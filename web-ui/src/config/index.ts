/** 运行期前端配置：全部来自 Vite env，避免在组件/请求层写死。 */

function text(value: string | undefined, fallback: string): string {
  const trimmed = value?.trim()
  return trimmed ? trimmed : fallback
}

function positiveInt(value: string | undefined, fallback: number): number {
  const parsed = Number.parseInt(value ?? '', 10)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback
}

function nonNegativeInt(value: string | undefined, fallback: number): number {
  const parsed = Number.parseInt(value ?? '', 10)
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : fallback
}

function boolean(value: string | undefined, fallback: boolean): boolean {
  if (value === undefined || value === '') return fallback
  return value === 'true' || value === '1'
}

export const appConfig = {
  appName: text(import.meta.env.VITE_APP_NAME, 'VerityAgent'),
  apiBaseUrl: text(import.meta.env.VITE_API_BASE_URL, '/api/v1'),
  tokenStorageKey: text(import.meta.env.VITE_TOKEN_STORAGE_KEY, 'verity-agent-token'),
  themeStorageKey: text(import.meta.env.VITE_THEME_STORAGE_KEY, 'verity-agent-theme'),
  requestTimeoutMs: positiveInt(import.meta.env.VITE_REQUEST_TIMEOUT_MS, 15_000),
  messagePageSize: positiveInt(import.meta.env.VITE_MESSAGE_PAGE_SIZE, 50),
  maxMessagePages: positiveInt(import.meta.env.VITE_MAX_MESSAGE_PAGES, 20),
  sseReconnectDelayMs: nonNegativeInt(import.meta.env.VITE_SSE_RECONNECT_DELAY_MS, 1_000),
  sseMaxReconnectAttempts: nonNegativeInt(import.meta.env.VITE_SSE_MAX_RECONNECT_ATTEMPTS, 5),
  showThinking: boolean(import.meta.env.VITE_SHOW_THINKING, true),
  conversationTitleMaxLength: positiveInt(
    import.meta.env.VITE_CONVERSATION_TITLE_MAX_LENGTH,
    30,
  ),
  maxMessageChars: positiveInt(import.meta.env.VITE_MAX_MESSAGE_CHARS, 200_000),
  runStoragePrefix: text(import.meta.env.VITE_RUN_STORAGE_PREFIX, 'verity-agent-run:'),
  indexJobPollIntervalMs: positiveInt(
    import.meta.env.VITE_INDEX_JOB_POLL_INTERVAL_MS,
    2_000,
  ),
  indexJobPollMaxAttempts: positiveInt(
    import.meta.env.VITE_INDEX_JOB_POLL_MAX_ATTEMPTS,
    120,
  ),
  defaultKbPermissionCode: text(
    import.meta.env.VITE_DEFAULT_KB_PERMISSION_CODE,
    'kb:read',
  ),
} as const
