<template>
  <article class="message" :class="[`message--${message.role}`, { 'message--streaming': isStreaming }]">
    <div v-if="message.role === 'assistant'" class="message__avatar">
      <AppLogo size="xs" />
    </div>

    <div class="message__body">
      <div v-if="message.role === 'assistant'" class="message__head">
        <span class="message__name">{{ appConfig.appName }}</span>
        <el-tag v-if="message.status !== 'completed'" size="small" type="warning" effect="plain">
          {{ statusLabel(message.status) }}
        </el-tag>
        <span class="message__time">{{ formatDateTime(message.created_at) }}</span>
      </div>

      <template v-if="message.role === 'user'">
        <div class="message__bubble message__bubble--user">{{ message.content }}</div>
        <div v-if="userImages.length || userFiles.length" class="message__attachments">
          <a
            v-for="(image, index) in userImages"
            :key="image.url || image.data || index"
            class="message__attachment"
            :href="image.url || undefined"
            target="_blank"
            rel="noopener noreferrer"
          >
            <el-icon><Picture /></el-icon>
            <span>图片</span>
          </a>
          <a
            v-for="file in userFiles"
            :key="file.file_id"
            class="message__attachment"
            :href="file.url || undefined"
            target="_blank"
            rel="noopener noreferrer"
          >
            <el-icon><Document /></el-icon>
            <span>{{ file.filename || file.file_id }}</span>
          </a>
        </div>
        <div class="message__foot message__foot--user">
          <span class="message__time">{{ formatDateTime(message.created_at) }}</span>
        </div>
      </template>

      <template v-else-if="message.role === 'tool'">
        <div class="tool-result">
          <div class="tool-result__head">
            <el-icon class="tool-result__icon"><Tools /></el-icon>
            <span class="tool-result__name">{{ message.name || '工具结果' }}</span>
            <el-tag
              v-if="message.status !== 'completed'"
              size="small"
              :type="message.status === 'failed' ? 'danger' : 'info'"
              effect="plain"
            >
              {{ statusLabel(message.status) }}
            </el-tag>
            <span v-if="message.tool_call_id" class="tool-result__id">
              {{ message.tool_call_id }}
            </span>
          </div>
          <div v-if="message.content" class="tool-result__content">{{ message.content }}</div>
          <details v-if="jsonText" class="tool-result__json">
            <summary>结构化结果</summary>
            <pre>{{ jsonText }}</pre>
          </details>
        </div>
      </template>

      <template v-else>
        <div class="message__bubble message__bubble--assistant">
          <ThinkingBlock
            v-if="showThinking && thinkingTexts.length"
            :items="thinkingTexts"
          />

          <div
            v-if="message.content"
            class="markdown-body"
            v-html="renderedContent"
          ></div>
          <div v-else-if="isStreaming" class="message__typing">
            <span></span><span></span><span></span>
          </div>

          <ToolCallList :calls="message.tool_calls" />

          <div v-if="artifacts.length" class="message__artifacts">
            <a
              v-for="artifact in artifacts"
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
          <span class="message__time">{{ formatDateTime(message.created_at) }}</span>
          <el-tooltip v-if="message.content && !isStreaming" content="复制回答" placement="top">
            <el-button class="message__copy" text circle size="small" @click="copyContent">
              <el-icon><CopyDocument /></el-icon>
            </el-button>
          </el-tooltip>
        </div>
      </template>
    </div>
  </article>
</template>

<script setup lang="ts">
import {
  CopyDocument,
  Document,
  Picture,
  Tools,
} from '@element-plus/icons-vue'

import AppLogo from '@/components/layout/AppLogo.vue'
import type { Artifact, ChatMessage, FileRef, ImageRef } from '@/api/types'
import { appConfig } from '@/config'
import { formatDateTime, statusLabel } from '@/utils/format'
import { renderMarkdown } from '@/utils/markdown'

const props = defineProps<{ message: ChatMessage }>()

const isStreaming = computed(() => props.message.status === 'streaming')
const renderedContent = computed(() => renderMarkdown(props.message.content))
const thinkingTexts = computed(() => (props.message.thinking ?? []).map((item) => item.text))
const showThinking = computed(() => appConfig.showThinking)
const artifacts = computed<Artifact[]>(() => props.message.artifacts ?? [])
const userImages = computed<ImageRef[]>(() => props.message.images ?? [])
const userFiles = computed<FileRef[]>(() => props.message.files ?? [])

const jsonText = computed(() => {
  const value = props.message.json
  if (value === undefined || value === null) return ''
  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return String(value)
  }
})

async function copyContent(): Promise<void> {
  try {
    await navigator.clipboard.writeText(props.message.content)
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

.message--user {
  flex-direction: row-reverse;
  margin-bottom: var(--app-space-lg);
}

.message--tool {
  max-width: var(--app-chat-max-width);
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

.message--user .message__body {
  align-items: flex-end;
  max-width: min(76%, 680px);
}

.message--tool .message__body {
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
}

.message__bubble--user {
  padding: var(--app-space-sm) var(--app-space-md);
  color: var(--app-user-bubble-text);
  white-space: pre-wrap;
  word-break: break-word;
  background-color: var(--app-bg-user-bubble);
  border: 1px solid var(--app-border-color-light);
  border-radius: var(--app-radius-lg) var(--app-radius-lg) var(--app-radius-xs)
    var(--app-radius-lg);
}

.message__bubble--assistant {
  padding: 0;
  color: var(--app-text-primary);
  background: transparent;
  border: 0;
  border-radius: 0;
  box-shadow: none;
}

.message__foot {
  display: flex;
  align-items: center;
  gap: var(--app-space-xs);
  min-height: 22px;
  padding-left: var(--app-space-xs);
}

.message__foot--user {
  justify-content: flex-end;
}

.message__attachments {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: var(--app-space-xs);
}

.message__attachment {
  display: inline-flex;
  align-items: center;
  gap: var(--app-space-2xs);
  padding: var(--app-space-2xs) var(--app-space-sm);
  font-size: var(--app-font-size-xs);
  color: var(--app-text-secondary);
  background-color: var(--app-bg-subtle);
  border: 1px solid var(--app-border-color-light);
  border-radius: var(--app-radius-pill);
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

@media (hover: none) {
  .message__copy {
    opacity: 1;
  }
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

/* —— 工具结果 —— */
.tool-result {
  width: 100%;
  padding: var(--app-space-md);
  background-color: var(--app-bg-tool);
  border: 1px solid var(--app-border-color-light);
  border-radius: var(--app-radius-md);
}

.tool-result__head {
  display: flex;
  align-items: center;
  gap: var(--app-space-xs);
  font-size: var(--app-font-size-sm);
}

.tool-result__icon {
  color: var(--app-color-primary);
}

.tool-result__name {
  font-weight: 500;
  color: var(--app-text-primary);
}

.tool-result__id {
  margin-left: auto;
  overflow: hidden;
  font-family: var(--app-font-family-mono);
  font-size: var(--app-font-size-xs);
  color: var(--app-text-muted);
  text-overflow: ellipsis;
  white-space: nowrap;
}

.tool-result__content {
  margin-top: var(--app-space-sm);
  font-size: var(--app-font-size-sm);
  line-height: var(--app-line-height-base);
  color: var(--app-text-regular);
  white-space: pre-wrap;
  word-break: break-word;
}

.tool-result__json {
  margin-top: var(--app-space-sm);
}

.tool-result__json summary {
  font-size: var(--app-font-size-xs);
  color: var(--app-text-secondary);
  cursor: pointer;
}

.tool-result__json pre {
  max-height: 280px;
  padding: var(--app-space-sm);
  margin: var(--app-space-xs) 0 0;
  overflow: auto;
  font-family: var(--app-font-family-mono);
  font-size: var(--app-font-size-xs);
  line-height: var(--app-line-height-tight);
  color: var(--app-text-regular);
  background-color: var(--app-bg-card);
  border: 1px solid var(--app-border-color-light);
  border-radius: var(--app-radius-sm);
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

  .message__bubble {
    padding: var(--app-space-sm) var(--app-space-md);
  }
}
</style>
