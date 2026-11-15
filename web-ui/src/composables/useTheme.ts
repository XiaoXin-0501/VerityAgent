import { appConfig } from '@/config'

export type ThemeMode = 'light' | 'dark'

const theme = ref<ThemeMode>('light')
let initialized = false

function systemPrefersDark(): boolean {
  return typeof window !== 'undefined' && window.matchMedia('(prefers-color-scheme: dark)').matches
}

function storedTheme(): ThemeMode | null {
  try {
    const value = localStorage.getItem(appConfig.themeStorageKey)
    return value === 'light' || value === 'dark' ? value : null
  } catch {
    return null
  }
}

function apply(mode: ThemeMode): void {
  theme.value = mode
  const root = document.documentElement
  root.dataset.theme = mode
  root.classList.toggle('dark', mode === 'dark')
  try {
    localStorage.setItem(appConfig.themeStorageKey, mode)
  } catch {
    // 隐私模式下仅内存生效
  }
}

/** 应用启动时调用：优先用户选择，其次系统偏好。 */
export function initTheme(): void {
  if (initialized) return
  initialized = true
  apply(storedTheme() ?? (systemPrefersDark() ? 'dark' : 'light'))
}

/** 主题状态与切换；多个组件共享同一模块级 ref。 */
export function useTheme() {
  function setTheme(mode: ThemeMode): void {
    apply(mode)
  }

  function toggle(): void {
    apply(theme.value === 'dark' ? 'light' : 'dark')
  }

  return { theme, setTheme, toggle }
}
