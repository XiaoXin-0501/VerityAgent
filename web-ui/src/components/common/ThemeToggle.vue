<template>
  <el-tooltip :content="isDark ? '切换到浅色模式' : '切换到深色模式'" placement="bottom">
    <el-button
      class="theme-toggle"
      circle
      text
      :aria-label="isDark ? '切换到浅色模式' : '切换到深色模式'"
      @click="toggle"
    >
      <el-icon :size="18">
        <Sunny v-if="isDark" />
        <Moon v-else />
      </el-icon>
    </el-button>
  </el-tooltip>
</template>

<script setup lang="ts">
import { Moon, Sunny } from '@element-plus/icons-vue'

import { useTheme } from '@/composables/useTheme'

const { theme, toggle } = useTheme()
const isDark = computed(() => theme.value === 'dark')
</script>

<style scoped>
.theme-toggle {
  color: var(--app-text-secondary);
}

.theme-toggle:hover {
  color: var(--app-text-primary);
  background-color: var(--app-bg-hover);
}
</style>
