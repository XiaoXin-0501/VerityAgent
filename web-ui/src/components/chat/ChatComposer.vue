<template>
  <div class="composer">
    <div
      class="composer__box"
      :class="{ 'composer__box--disabled': disabled }"
      @keydown.enter.exact="handleEnter"
    >
      <el-input
        v-model="text"
        class="composer__input"
        type="textarea"
        :autosize="{ minRows: 1, maxRows: 6 }"
        :maxlength="maxChars"
        :placeholder="placeholder"
        :disabled="disabled"
        resize="none"
        aria-label="输入消息"
      />

      <div class="composer__actions">
        <span class="composer__hint">Enter 发送，Shift + Enter 换行</span>
        <span v-if="text.length > 0" class="composer__count">
          {{ text.length }} / {{ maxChars }}
        </span>

        <el-button
          v-if="streaming"
          class="composer__stop"
          size="small"
          :loading="cancelling"
          @click="emit('cancel')"
        >
          <el-icon v-if="!cancelling"><VideoPause /></el-icon>
          <span>{{ cancelling ? '取消中' : '停止' }}</span>
        </el-button>

        <el-button
          class="composer__send"
          circle
          :disabled="!canSend"
          :loading="sending"
          aria-label="发送消息"
          @click="handleSend"
        >
          <el-icon><Promotion /></el-icon>
        </el-button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { Promotion, VideoPause } from '@element-plus/icons-vue'

import { appConfig } from '@/config'

const props = defineProps<{
  modelValue: string
  disabled: boolean
  streaming: boolean
  cancelling: boolean
  sending: boolean
  placeholder: string
  maxChars?: number
}>()

const emit = defineEmits<{
  'update:modelValue': [value: string]
  send: []
  cancel: []
}>()

const maxChars = computed(() => props.maxChars ?? appConfig.maxMessageChars)

const text = computed({
  get: () => props.modelValue,
  set: (value: string) => emit('update:modelValue', value),
})

const canSend = computed(
  () =>
    !props.disabled &&
    !props.streaming &&
    !props.sending &&
    text.value.trim().length > 0 &&
    text.value.length <= maxChars.value,
)

function handleEnter(event: KeyboardEvent): void {
  // 只处理输入框内的 Enter；按钮/其他控件交给自身处理。
  if (!(event.target instanceof HTMLTextAreaElement)) return
  // 中文输入法选词时的 Enter 不触发发送。
  if (event.isComposing) return
  event.preventDefault()
  handleSend()
}

function handleSend(): void {
  if (!canSend.value) return
  emit('send')
}
</script>

<style scoped>
.composer {
  position: relative;
  flex: 0 0 auto;
  padding: var(--app-space-md) var(--app-space-xl) var(--app-space-lg);
  background: var(--app-composer-fade);
}

.composer__box {
  max-width: var(--app-chat-max-width);
  padding: var(--app-space-sm) var(--app-space-sm) var(--app-space-xs);
  margin: 0 auto;
  background-color: var(--app-bg-card);
  border: 1px solid var(--app-border-color);
  border-radius: var(--app-radius-xl);
  box-shadow: var(--app-shadow-float);
  transition:
    border-color var(--app-transition-fast),
    box-shadow var(--app-transition-fast);
}

.composer__box:focus-within {
  border-color: var(--app-color-primary);
  box-shadow: var(--app-shadow-focus);
}

.composer__box--disabled {
  opacity: 0.7;
}

.composer__input :deep(.el-textarea__inner) {
  padding: var(--app-space-xs) var(--app-space-sm);
  font-family: var(--app-font-family);
  font-size: var(--app-font-size-base);
  line-height: var(--app-line-height-base);
  color: var(--app-text-primary);
  background: transparent;
  border: 0;
  box-shadow: none;
}

.composer__input :deep(.el-textarea__inner::placeholder) {
  color: var(--app-text-placeholder);
}

.composer__actions {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: var(--app-space-sm);
  padding: var(--app-space-2xs) var(--app-space-2xs) 0;
}

.composer__hint,
.composer__count {
  font-size: var(--app-font-size-xs);
  color: var(--app-text-muted);
}

.composer__hint {
  margin-right: auto;
}

.composer__count {
  font-variant-numeric: tabular-nums;
}

.composer__stop.el-button {
  color: var(--app-text-secondary);
  background-color: transparent;
  border-color: var(--app-border-color);
}

.composer__stop.el-button:hover {
  color: var(--app-color-danger);
  background-color: var(--app-color-danger-soft);
  border-color: var(--app-color-danger-soft);
}

.composer__stop span {
  margin-left: var(--app-space-2xs);
}

.composer__send.el-button {
  flex: 0 0 auto;
  width: 36px;
  height: 36px;
  color: var(--app-color-primary);
  background-color: var(--app-color-primary-soft);
  border-color: transparent;
}

.composer__send.el-button:hover:not(.is-disabled) {
  color: var(--app-text-on-primary);
  background-color: var(--app-color-primary);
}

.composer__send.el-button.is-disabled {
  color: var(--app-text-muted);
  background-color: var(--app-bg-subtle);
}

@media (max-width: 900px) {
  .composer {
    padding: var(--app-space-sm) var(--app-space-md) var(--app-space-md);
  }

  .composer__hint {
    display: none;
  }
}
</style>
