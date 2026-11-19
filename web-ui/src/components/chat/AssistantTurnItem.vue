<template>
  <article class="message" :class="{ 'message--streaming': isStreaming }">
    <div class="message__avatar">
      <AppLogo size="xs" />
    </div>

    <div class="message__body">
      <div class="message__head">
        <span class="message__name">{{ appConfig.appName }}</span>
        <el-tag v-if="turn.status !== 'completed'" size="small" type="warning" effect="plain">
          {{ statusLabel(turn.status) }}
        </el-tag>
        <span v-if="turn.createdAt" class="message__time">
          {{ formatDateTime(turn.createdAt) }}
        </span>
      </div>

      <div class="message__bubble">
        <details v-if="hasActivity" class="activity">
          <summary class="activity__summary">
            <el-icon class="activity__icon"><MagicStick /></el-icon>
            <span>{{ activityTitle }}</span>
            <span class="activity__meta">{{ activityMeta }}</span>
          </summary>
          <div class="activity__body">
            <div
              v-if="turn.thinkingPending && !visibleThinking.length"
              class="activity__pending"
            >
              正在思考…
            </div>
            <div v-for="(text, index) in visibleThinking" :key="`thinking-${index}`" class="activity__thinking">
              {{ text }}
            </div>
            <div v-for="(text, index) in visibleProcessTexts" :key="`process-${index}`" class="activity__process">
              {{ text }}
            </div>
            <ToolCallList v-if="hasTools" :calls="turn.toolCalls" />
          </div>
        </details>

        <div v-if="turn.content" class="markdown-body" v-html="renderedContent"></div>
        <div v-else-if="isStreaming" class="message__typing">
          <span></span><span></span><span></span>
        </div>

        <div v-if="turn.artifacts.length" class="message__artifacts">
          <a
            v-for="artifact in turn.artifacts"
            :key="artifact.url"
            class="message__artifact"
            :href="artifact.url"
            target="_blank"
            rel="noopener noreferrer"
          >
            <el-icon><Document /></el-icon>
            <span>{{ artifact.filename || artifact.kind || '附件' }}</span>
          </a>
        </div>
      </div>

      <div class="message__foot">
        <span v-if="turn.createdAt" class="message__time">{{ formatDateTime(turn.createdAt) }}</span>
        <el-tooltip v-if="turn.content && !isStreaming" content="复制回答" placement="top">
          <el-button class="message__copy" text circle size="small" @click="copyContent">
            <el-icon><CopyDocument /></el-icon>
          </el-button>
        </el-tooltip>
      </div>
    </div>
  </article>
</template>

<script setup lang="ts">
import { CopyDocument, Document, MagicStick } from '@element-plus/icons-vue'

import AppLogo from '@/components/layout/AppLogo.vue'
import ToolCallList from '@/components/chat/ToolCallList.vue'
import type { AssistantTurn } from '@/utils/chatGrouping'
import { appConfig } from '@/config'
import { formatDateTime, statusLabel } from '@/utils/format'
import { renderMarkdown } from '@/utils/markdown'

const props = defineProps<{ turn: AssistantTurn }>()

const isStreaming = computed(() => props.turn.status === 'streaming')
const renderedContent = computed(() => renderMarkdown(props.turn.content))
const hasTools = computed(() => props.turn.toolCalls.length > 0)
// 思考在分组阶段已过滤工具协议细节；这里只按产品开关决定是否展示。
const visibleThinking = computed(() =>
  appConfig.showThinking ? props.turn.thinking : [],
)
const visibleProcessTexts = computed(() =>
  hasTools.value ? [] : props.turn.processTexts,
)
const hasActivity = computed(
  () =>
    props.turn.thinkingPending ||
    visibleThinking.value.length > 0 ||
    visibleProcessTexts.value.length > 0 ||
    hasTools.value,
)
const activityTitle = computed(() => {
  if (
    (visibleThinking.value.length > 0 || props.turn.thinkingPending) &&
    hasTools.value
  ) {
    return '思考与执行过程'
  }
  return hasTools.value ? '执行过程' : '思考过程'
})
const activityMeta = computed(() => {
  const parts: string[] = []
  if (props.turn.thinkingPending) parts.push('思考中')
  if (props.turn.stepCount > 1) parts.push(`${props.turn.stepCount} 步`)
  if (props.turn.toolCalls.length) parts.push(`${props.turn.toolCalls.length} 次工具调用`)
  return parts.join(' · ')
})

async function copyContent(): Promise<void> {
  try {
    await navigator.clipboard.writeText(props.turn.content)
    ElMessage.success('已复制')
  } catch {
    ElMessage.warning('当前浏览器不支持复制')
  }
}
</script>

<style scoped>
.message {
  display: flex;
  gap: var(--app-space-md);
  width: 100%;
  max-width: var(--app-chat-max-width);
  margin: 0 auto var(--app-space-xl);
}

.message__avatar {
  display: flex;
  align-items: center;
  justify-content: center;
  flex: 0 0 auto;
  width: var(--app-message-avatar-size);
  height: var(--app-message-avatar-size);
  margin-top: 2px;
}

.message__body {
  display: flex;
  flex-direction: column;
  gap: var(--app-space-xs);
  min-width: 0;
  max-width: 100%;
}

.message__head {
  display: flex;
  align-items: center;
  gap: var(--app-space-sm);
  margin-bottom: var(--app-space-2xs);
  font-size: var(--app-font-size-xs);
  color: var(--app-text-muted);
}

.message__name {
  font-size: var(--app-font-size-sm);
  font-weight: 600;
  color: var(--app-text-primary);
}

.message__time {
  font-size: var(--app-font-size-xs);
  color: var(--app-text-muted);
}

.message__bubble {
  font-size: var(--app-chat-font-size);
  line-height: var(--app-chat-line-height);
  color: var(--app-text-primary);
}

.message__typing {
  display: flex;
  align-items: center;
  gap: var(--app-space-xs);
  min-height: 24px;
}

.message__typing span {
  width: 6px;
  height: 6px;
  background-color: var(--app-text-muted);
  border-radius: var(--app-radius-pill);
  animation: typing 1.2s infinite ease-in-out;
}

.message__typing span:nth-child(2) {
  animation-delay: 0.15s;
}

.message__typing span:nth-child(3) {
  animation-delay: 0.3s;
}

@keyframes typing {
  0%,
  60%,
  100% {
    opacity: 0.35;
    transform: translateY(0);
  }
  30% {
    opacity: 1;
    transform: translateY(-3px);
  }
}

.activity {
  margin-bottom: var(--app-space-sm);
  background-color: var(--app-bg-tool);
  border: 1px solid var(--app-border-color-light);
  border-radius: var(--app-radius-md);
}

.activity__summary {
  display: flex;
  align-items: center;
  gap: var(--app-space-xs);
  padding: var(--app-space-sm) var(--app-space-md);
  font-size: var(--app-font-size-sm);
  color: var(--app-text-secondary);
  cursor: pointer;
  list-style: none;
}

.activity__summary::-webkit-details-marker {
  display: none;
}

.activity__icon {
  color: var(--app-color-primary);
}

.activity__meta {
  margin-left: auto;
  font-size: var(--app-font-size-xs);
  color: var(--app-text-muted);
}

.activity__body {
  display: flex;
  flex-direction: column;
  gap: var(--app-space-sm);
  padding: 0 var(--app-space-md) var(--app-space-md);
}

.activity__thinking,
.activity__process {
  padding-left: var(--app-space-sm);
  font-size: var(--app-font-size-sm);
  line-height: var(--app-line-height-base);
  color: var(--app-text-secondary);
  white-space: pre-wrap;
  word-break: break-word;
  border-left: 2px solid var(--app-border-color-light);
}

.activity__pending {
  font-size: var(--app-font-size-sm);
  color: var(--app-text-muted);
}

.activity__process {
  color: var(--app-text-regular);
}

.message__foot {
  display: flex;
  align-items: center;
  gap: var(--app-space-xs);
  min-height: 22px;
  padding-left: var(--app-space-xs);
}

.message__copy {
  color: var(--app-text-muted);
  opacity: 0;
  transition: opacity var(--app-transition-fast);
}

.message:hover .message__copy,
.message__copy:focus-visible {
  opacity: 1;
}

.message__artifacts {
  display: flex;
  flex-wrap: wrap;
  gap: var(--app-space-sm);
  margin-top: var(--app-space-sm);
}

.message__artifact {
  display: inline-flex;
  align-items: center;
  gap: var(--app-space-2xs);
  padding: var(--app-space-2xs) var(--app-space-sm);
  font-size: var(--app-font-size-sm);
  color: var(--app-color-primary);
  background-color: var(--app-color-primary-soft);
  border-radius: var(--app-radius-pill);
}

/* —— Markdown —— */
.markdown-body {
  overflow-wrap: anywhere;
}

.markdown-body :deep(p) {
  margin: 0 0 var(--app-space-sm);
}

.markdown-body :deep(p:last-child) {
  margin-bottom: 0;
}

.markdown-body :deep(h1),
.markdown-body :deep(h2),
.markdown-body :deep(h3),
.markdown-body :deep(h4) {
  margin: var(--app-space-md) 0 var(--app-space-sm);
  font-weight: 600;
  line-height: var(--app-line-height-tight);
  color: var(--app-text-primary);
}

.markdown-body :deep(h1) {
  font-size: var(--app-font-size-xl);
}

.markdown-body :deep(h2) {
  font-size: var(--app-font-size-lg);
}

.markdown-body :deep(h3),
.markdown-body :deep(h4) {
  font-size: var(--app-font-size-base);
}

.markdown-body :deep(ul),
.markdown-body :deep(ol) {
  padding-left: var(--app-space-lg);
  margin: 0 0 var(--app-space-sm);
  list-style: revert;
}

.markdown-body :deep(li) {
  margin-bottom: var(--app-space-2xs);
}

.markdown-body :deep(a) {
  color: var(--app-color-primary);
  text-decoration: underline;
  text-underline-offset: 2px;
}

.markdown-body :deep(blockquote) {
  padding: var(--app-space-sm) var(--app-space-md);
  margin: var(--app-space-sm) 0;
  color: var(--app-text-secondary);
  background-color: var(--app-bg-muted);
  border-left: 3px solid var(--app-color-primary-soft);
  border-radius: var(--app-radius-sm);
}

.markdown-body :deep(code) {
  padding: 2px 5px;
  font-family: var(--app-font-family-mono);
  font-size: 0.92em;
  background-color: var(--app-bg-subtle);
  border: 1px solid var(--app-border-color-light);
  border-radius: var(--app-radius-xs);
}

.markdown-body :deep(pre) {
  padding: var(--app-space-md);
  margin: var(--app-space-sm) 0;
  overflow-x: auto;
  color: var(--app-text-inverse);
  background-color: var(--app-bg-code);
  border-radius: var(--app-radius-md);
}

.markdown-body :deep(pre code) {
  padding: 0;
  color: inherit;
  background: transparent;
  border: 0;
}

.markdown-body :deep(table) {
  width: 100%;
  margin: var(--app-space-sm) 0;
  overflow: hidden;
  font-size: var(--app-font-size-sm);
  border-collapse: collapse;
  border: 1px solid var(--app-border-color);
  border-radius: var(--app-radius-sm);
}

.markdown-body :deep(th),
.markdown-body :deep(td) {
  padding: var(--app-space-sm) var(--app-space-md);
  text-align: left;
  border: 1px solid var(--app-border-color-light);
}

.markdown-body :deep(th) {
  font-weight: 600;
  background-color: var(--app-bg-muted);
}

.markdown-body :deep(hr) {
  margin: var(--app-space-md) 0;
  border: 0;
  border-top: 1px solid var(--app-border-color);
}

.markdown-body :deep(img) {
  max-width: 100%;
  height: auto;
  border-radius: var(--app-radius-sm);
}

@media (max-width: 900px) {
  .message__body {
    max-width: 88%;
  }
}
</style>
