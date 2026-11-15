<template>
  <span class="app-logo" :class="`app-logo--${size}`" aria-hidden="true">
    <svg class="app-logo__glyph" viewBox="0 0 24 24">
      <path d="M6.2 7.5 12 17.9 17.8 7.5" />
      <circle class="app-logo__spark" cx="19.4" cy="4.6" r="1.3" />
    </svg>
  </span>
</template>

<script setup lang="ts">
/** 全站统一 Logo：品牌位、助手头像、空状态、登录页共用同一个标记。 */
withDefaults(defineProps<{ size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl' }>(), {
  size: 'sm',
})
</script>

<style scoped>
.app-logo {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex: 0 0 auto;
  color: var(--app-text-on-primary);
  background: var(--app-brand-gradient);
  border-radius: 30%;
  box-shadow: 0 4px 12px var(--app-color-primary-soft);
}

.app-logo__glyph {
  width: 62%;
  height: 62%;
}

.app-logo__glyph path {
  fill: none;
  stroke: currentColor;
  stroke-width: 2.1;
  stroke-linecap: round;
  stroke-linejoin: round;
}

.app-logo__spark {
  fill: currentColor;
}

.app-logo--xs {
  width: 24px;
  height: 24px;
  border-radius: var(--app-radius-sm);
}

.app-logo--sm {
  width: 30px;
  height: 30px;
  border-radius: var(--app-radius-md);
}

.app-logo--md {
  width: 44px;
  height: 44px;
  border-radius: var(--app-radius-lg);
}

.app-logo--lg {
  width: 64px;
  height: 64px;
  border-radius: var(--app-radius-xl);
}

.app-logo--xl {
  width: 76px;
  height: 76px;
  border-radius: var(--app-radius-xl);
}
</style>
