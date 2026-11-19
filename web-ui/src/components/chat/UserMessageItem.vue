<template>
  <article class="message message--user">
    <div class="message__body">
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
      <div class="message__foot">
        <span class="message__time">{{ formatDateTime(message.created_at) }}</span>
      </div>
    </div>
  </article>
</template>

<script setup lang="ts">
import { Document, Picture } from '@element-plus/icons-vue'

import type { ChatMessage, FileRef, ImageRef } from '@/api/types'
import { formatDateTime } from '@/utils/format'

const props = defineProps<{ message: ChatMessage }>()

const userImages = computed<ImageRef[]>(() => props.message.images ?? [])
const userFiles = computed<FileRef[]>(() => props.message.files ?? [])
</script>

<style scoped>
.message {
  display: flex;
  flex-direction: row-reverse;
  gap: var(--app-space-md);
  width: 100%;
  max-width: var(--app-chat-max-width);
  margin: 0 auto var(--app-space-lg);
}

.message__body {
  display: flex;
  flex-direction: column;
  gap: var(--app-space-xs);
  align-items: flex-end;
  min-width: 0;
  max-width: min(76%, 680px);
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

.message__time {
  font-size: var(--app-font-size-xs);
  color: var(--app-text-muted);
}

.message__foot {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: var(--app-space-xs);
  min-height: 22px;
  padding-left: var(--app-space-xs);
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

@media (max-width: 900px) {
  .message__body {
    max-width: 88%;
  }

  .message__bubble {
    padding: var(--app-space-sm) var(--app-space-md);
  }
}
</style>
