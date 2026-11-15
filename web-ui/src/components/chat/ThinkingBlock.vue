<template>
  <details v-if="text" class="thinking-block">
    <summary class="thinking-block__summary">
      <el-icon><Compass /></el-icon>
      <span>思考过程</span>
      <span class="thinking-block__hint">点击展开</span>
    </summary>
    <div class="thinking-block__body">{{ text }}</div>
  </details>
</template>

<script setup lang="ts">
import { Compass } from '@element-plus/icons-vue'

const props = defineProps<{ items: string[] }>()

const text = computed(() => props.items.filter((item) => item.trim()).join('\n\n').trim())
</script>

<style scoped>
.thinking-block {
  padding: var(--app-space-sm) var(--app-space-md);
  margin-bottom: var(--app-space-sm);
  background-color: var(--app-bg-muted);
  border: 1px dashed var(--app-border-color);
  border-radius: var(--app-radius-md);
}

.thinking-block__summary {
  display: flex;
  align-items: center;
  gap: var(--app-space-xs);
  font-size: var(--app-font-size-sm);
  font-weight: 500;
  color: var(--app-text-secondary);
  cursor: pointer;
  list-style: none;
}

.thinking-block__summary::-webkit-details-marker {
  display: none;
}

.thinking-block__hint {
  margin-left: auto;
  font-size: var(--app-font-size-xs);
  font-weight: 400;
  color: var(--app-text-muted);
}

.thinking-block__body {
  padding-top: var(--app-space-sm);
  margin-top: var(--app-space-sm);
  font-size: var(--app-font-size-sm);
  line-height: var(--app-line-height-base);
  color: var(--app-text-regular);
  white-space: pre-wrap;
  border-top: 1px solid var(--app-border-color-light);
}
</style>
