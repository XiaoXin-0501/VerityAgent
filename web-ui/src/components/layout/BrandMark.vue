<template>
  <div class="brand-mark" :class="{ 'brand-mark--compact': compact }">
    <AppLogo :size="compact ? 'xs' : 'sm'" />
    <span v-if="!compact" class="brand-mark__name">{{ appConfig.appName }}</span>
  </div>
</template>

<script setup lang="ts">
import AppLogo from '@/components/layout/AppLogo.vue'
import { appConfig } from '@/config'

defineProps<{ compact?: boolean }>()
</script>

<style scoped>
.brand-mark {
  display: flex;
  align-items: center;
  gap: var(--app-space-sm);
  min-width: 0;
}

.brand-mark__name {
  overflow: hidden;
  font-size: var(--app-font-size-base);
  font-weight: 600;
  color: var(--app-text-primary);
  letter-spacing: 0.01em;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.brand-mark--compact :deep(.app-logo) {
  width: 26px;
  height: 26px;
}
</style>
