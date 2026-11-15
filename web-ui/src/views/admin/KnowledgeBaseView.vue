<template>
  <section class="kb-page">
    <header class="kb-page__topbar">
      <div class="kb-page__topbar-left">
        <BrandMark />
        <span class="kb-page__divider" aria-hidden="true"></span>
        <el-button text class="kb-page__back" @click="router.push({ name: 'home' })">
          <el-icon><ArrowLeft /></el-icon>
          <span>返回对话</span>
        </el-button>
      </div>
      <div class="kb-page__topbar-right">
        <el-tooltip content="刷新知识库与文档" placement="bottom">
          <el-button
            circle
            text
            :loading="loadingKbs"
            aria-label="刷新知识库与文档"
            @click="handleRefresh"
          >
            <el-icon><Refresh /></el-icon>
          </el-button>
        </el-tooltip>
      </div>
    </header>

    <div class="kb-page__body">
      <aside class="kb-panel">
        <div class="kb-panel__head">
          <h2 class="kb-panel__title">知识库管理</h2>
          <el-tag size="small" type="info" effect="plain">
            {{ knowledgeBases.length }}
          </el-tag>
        </div>

        <el-button class="kb-panel__create" @click="openCreate">
          <el-icon><Plus /></el-icon>
          <span>新建知识库</span>
        </el-button>

        <div class="kb-panel__list">
          <div v-if="loadingKbs && !knowledgeBases.length" class="kb-panel__loading">
            <el-skeleton :rows="3" animated />
          </div>

          <div v-else-if="!knowledgeBases.length" class="kb-panel__empty">
            <el-icon :size="28"><Collection /></el-icon>
            <p>还没有知识库</p>
            <el-button type="primary" text @click="openCreate">创建第一个知识库</el-button>
          </div>

          <button
            v-for="item in knowledgeBases"
            :key="item.id"
            type="button"
            class="kb-item"
            :class="{ 'kb-item--active': item.id === selectedId }"
            @click="selectKnowledgeBase(item.id)"
          >
            <span class="kb-item__name">{{ item.name }}</span>
            <span class="kb-item__desc">{{ item.description || '暂无描述' }}</span>
            <span class="kb-item__meta">
              <el-tag size="small" effect="plain">{{ item.required_permission_code }}</el-tag>
              <span>v{{ item.index_version }}</span>
            </span>
          </button>
        </div>

        <div class="kb-panel__foot">
          <UserMenu />
          <ThemeToggle />
        </div>
      </aside>

      <main class="doc-panel">
        <template v-if="selectedKb">
          <div class="doc-panel__head">
            <div class="doc-panel__identity">
              <h2 class="doc-panel__title">{{ selectedKb.name }}</h2>
              <p class="doc-panel__subtitle">
                {{ selectedKb.description || '暂无描述' }}
              </p>
              <div class="doc-panel__facts">
                <el-tag size="small" effect="plain" type="info">
                  权限：{{ selectedKb.required_permission_code }}
                </el-tag>
                <el-tag size="small" effect="plain" type="info">
                  {{ selectedKb.embedding_model_id }} · {{ selectedKb.embedding_dimensions }} 维
                </el-tag>
                <span>索引版本 v{{ selectedKb.index_version }}</span>
              </div>
            </div>
            <el-button type="primary" plain @click="openUpload">
              <el-icon><UploadFilled /></el-icon>
              <span>上传文档</span>
            </el-button>
          </div>

          <div class="doc-panel__upload-hint">
            支持 {{ acceptedExtensions.join('、') || '未启用文档类型' }}；单文件最大
            {{ maxFileLabel }}
          </div>

          <el-table
            v-loading="loadingDocs"
            class="doc-table"
            :data="documents"
            row-key="id"
            empty-text="还没有文档，点击右上角上传"
          >
            <el-table-column label="文档" min-width="260">
              <template #default="{ row }">
                <div class="doc-cell">
                  <el-icon class="doc-cell__icon"><Document /></el-icon>
                  <div class="doc-cell__body">
                    <span class="doc-cell__name">{{ row.original_filename }}</span>
                    <span class="doc-cell__meta">
                      {{ row.extension.toUpperCase() }} · {{ formatBytes(row.size_bytes) }}
                      <template v-if="row.description"> · {{ row.description }}</template>
                    </span>
                  </div>
                </div>
              </template>
            </el-table-column>

            <el-table-column label="状态" width="150">
              <template #default="{ row }">
                <div class="doc-status">
                  <el-tag size="small" :type="statusTagType(row.status)" effect="light">
                    {{ documentStatusLabel(row.status) }}
                  </el-tag>
                  <el-progress
                    v-if="row.status === 'processing' || row.status === 'pending'"
                    :percentage="Math.round((row.index_progress ?? 0) * 100)"
                    :show-text="false"
                    :stroke-width="5"
                  />
                </div>
              </template>
            </el-table-column>

            <el-table-column label="切片 / Token" width="130">
              <template #default="{ row }">
                <span class="doc-number">{{ row.chunk_count ?? '-' }}</span>
                <span class="doc-number-sep">/</span>
                <span class="doc-number">{{ row.token_count ?? '-' }}</span>
              </template>
            </el-table-column>

            <el-table-column label="更新时间" width="150">
              <template #default="{ row }">
                <span class="doc-time">{{ formatDateTime(row.updated_at) }}</span>
              </template>
            </el-table-column>

            <el-table-column label="操作" width="90" fixed="right">
              <template #default="{ row }">
                <el-button
                  text
                  type="danger"
                  :loading="deletingIds.includes(row.id)"
                  :disabled="row.status === 'deleting' || row.status === 'deleted'"
                  @click="handleDelete(row as KnowledgeDocument)"
                >
                  删除
                </el-button>
              </template>
            </el-table-column>
          </el-table>
        </template>

        <div v-else class="doc-panel__empty">
          <el-icon :size="40"><Collection /></el-icon>
          <h2>选择一个知识库</h2>
          <p>从左侧选择知识库，或先创建一个新的知识库。</p>
        </div>
      </main>
    </div>

    <el-dialog v-model="createVisible" title="新建知识库" width="520px" destroy-on-close>
      <el-form label-position="top">
        <el-form-item label="名称" required>
          <el-input
            v-model="createForm.name"
            maxlength="200"
            show-word-limit
            placeholder="例如：宠物医疗知识库"
          />
        </el-form-item>
        <el-form-item label="描述">
          <el-input
            v-model="createForm.description"
            type="textarea"
            :rows="3"
            maxlength="2000"
            show-word-limit
            placeholder="用于说明知识库覆盖范围"
          />
        </el-form-item>
        <el-form-item label="访问权限码" required>
          <el-input v-model="createForm.required_permission_code" placeholder="kb:read" />
          <p class="kb-form-hint">对话检索时按用户真实拥有的权限码校验。</p>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="createVisible = false">取消</el-button>
        <el-button type="primary" :loading="creating" @click="submitCreate">创建</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="uploadVisible" title="上传文档" width="560px" destroy-on-close>
      <el-upload
        drag
        :auto-upload="false"
        :limit="1"
        :accept="acceptString"
        :on-change="handleFileChange"
        :on-remove="handleFileRemove"
        :on-exceed="handleExceed"
      >
        <el-icon class="upload-icon"><UploadFilled /></el-icon>
        <div class="upload-text">拖拽文件到此处，或点击选择</div>
        <template #tip>
          <div class="upload-tip">
            支持 {{ acceptedExtensions.join('、') || '未启用文档类型' }}；单文件最大
            {{ maxFileLabel }}
          </div>
        </template>
      </el-upload>

      <el-form class="upload-form" label-position="top">
        <el-form-item label="文档描述（可选）">
          <el-input
            v-model="uploadDescription"
            type="textarea"
            :rows="2"
            maxlength="2000"
            placeholder="用于补充文档用途，留空则自动生成"
          />
        </el-form-item>
      </el-form>

      <el-progress
        v-if="uploading"
        :percentage="uploadPercent"
        :stroke-width="8"
        class="upload-progress"
      />

      <template #footer>
        <el-button :disabled="uploading" @click="uploadVisible = false">取消</el-button>
        <el-button
          type="primary"
          :loading="uploading"
          :disabled="!uploadFile"
          @click="submitUpload"
        >
          上传并入库
        </el-button>
      </template>
    </el-dialog>
  </section>
</template>

<script setup lang="ts">
import {
  ArrowLeft,
  Collection,
  Document,
  Plus,
  Refresh,
  UploadFilled,
} from '@element-plus/icons-vue'
import type { UploadFile, UploadRawFile } from 'element-plus'
import { ElMessageBox } from 'element-plus'

import BrandMark from '@/components/layout/BrandMark.vue'
import UserMenu from '@/components/layout/UserMenu.vue'
import ThemeToggle from '@/components/common/ThemeToggle.vue'
import {
  createKnowledgeBaseApi,
  deleteDocumentApi,
  getIndexJobApi,
  getKnowledgeOptionsApi,
  listDocumentsApi,
  listKnowledgeBasesApi,
  uploadDocumentApi,
} from '@/api/modules/knowledge'
import type {
  IndexJob,
  KnowledgeBase,
  KnowledgeDocument,
  KnowledgeOptions,
} from '@/api/types'
import { appConfig } from '@/config'
import {
  documentStatusLabel,
  formatBytes,
  formatDateTime,
} from '@/utils/format'

const DOC_TYPE_EXTENSIONS: Record<string, string[]> = {
  pdf: ['.pdf'],
  docx: ['.docx'],
  markdown: ['.md', '.markdown'],
  text: ['.txt', '.text'],
  spreadsheet: ['.xlsx', '.xls', '.csv'],
  pptx: ['.pptx'],
  html: ['.html', '.htm'],
}

const loadingKbs = ref(false)
const loadingDocs = ref(false)
const router = useRouter()
const creating = ref(false)
const uploading = ref(false)
const uploadPercent = ref(0)
const knowledgeBases = ref<KnowledgeBase[]>([])
const documents = ref<KnowledgeDocument[]>([])
const selectedId = ref<string | null>(null)
const options = ref<KnowledgeOptions | null>(null)
const deletingIds = ref<string[]>([])

const createVisible = ref(false)
const uploadVisible = ref(false)
const createForm = reactive({
  name: '',
  description: '',
  required_permission_code: appConfig.defaultKbPermissionCode,
})
const uploadFile = ref<File | null>(null)
const uploadDescription = ref('')

let disposed = false

const selectedKb = computed(
  () => knowledgeBases.value.find((item) => item.id === selectedId.value) ?? null,
)

const acceptedExtensions = computed(() => {
  const types = options.value?.enabled_doc_types ?? []
  return types.flatMap((item) => DOC_TYPE_EXTENSIONS[item] ?? [])
})

const acceptString = computed(() => acceptedExtensions.value.join(','))

const maxFileLabel = computed(() => formatBytes(options.value?.max_file_bytes ?? 0))

function statusTagType(status: string): 'success' | 'warning' | 'danger' | 'info' {
  switch (status) {
    case 'completed':
      return 'success'
    case 'processing':
    case 'pending':
      return 'warning'
    case 'failed':
      return 'danger'
    default:
      return 'info'
  }
}

async function loadOptions(): Promise<void> {
  options.value = await getKnowledgeOptionsApi()
}

async function loadKnowledgeBases(): Promise<void> {
  loadingKbs.value = true
  try {
    const result = await listKnowledgeBasesApi()
    knowledgeBases.value = result.items
    if (selectedId.value && !result.items.some((item) => item.id === selectedId.value)) {
      selectedId.value = null
    }
    if (!selectedId.value && result.items[0]) {
      selectedId.value = result.items[0].id
    }
  } finally {
    loadingKbs.value = false
  }
}

async function loadDocuments(kbId = selectedId.value): Promise<void> {
  if (!kbId) {
    documents.value = []
    return
  }
  loadingDocs.value = true
  try {
    const result = await listDocumentsApi(kbId)
    if (selectedId.value === kbId) documents.value = result.items
  } finally {
    loadingDocs.value = false
  }
}

async function selectKnowledgeBase(id: string): Promise<void> {
  selectedId.value = id
  await loadDocuments(id)
}

async function handleRefresh(): Promise<void> {
  await loadKnowledgeBases()
  await loadDocuments()
}

function openCreate(): void {
  createForm.name = ''
  createForm.description = ''
  createForm.required_permission_code =
    options.value?.default_permission_code ?? appConfig.defaultKbPermissionCode
  createVisible.value = true
}

async function submitCreate(): Promise<void> {
  const name = createForm.name.trim()
  const permission = createForm.required_permission_code.trim()
  if (!name) {
    ElMessage.warning('请输入知识库名称')
    return
  }
  if (!permission) {
    ElMessage.warning('请输入访问权限码')
    return
  }
  creating.value = true
  try {
    const kb = await createKnowledgeBaseApi({
      name,
      description: createForm.description.trim() || null,
      required_permission_code: permission,
    })
    knowledgeBases.value = [kb, ...knowledgeBases.value]
    selectedId.value = kb.id
    documents.value = []
    createVisible.value = false
    ElMessage.success('知识库已创建')
  } catch {
    // 错误提示由请求层统一处理
  } finally {
    creating.value = false
  }
}

function openUpload(): void {
  uploadFile.value = null
  uploadDescription.value = ''
  uploadPercent.value = 0
  uploadVisible.value = true
}

function handleFileChange(file: UploadFile): void {
  const raw = file.raw as UploadRawFile | undefined
  if (!raw) return
  const extension = `.${raw.name.split('.').pop()?.toLowerCase() ?? ''}`
  if (!acceptedExtensions.value.includes(extension)) {
    ElMessage.warning(`不支持的文件类型：${extension}`)
    uploadFile.value = null
    return
  }
  const maxBytes = options.value?.max_file_bytes ?? 0
  if (maxBytes > 0 && raw.size > maxBytes) {
    ElMessage.warning(`文件超过大小上限 ${maxFileLabel.value}`)
    uploadFile.value = null
    return
  }
  uploadFile.value = raw
}

function handleFileRemove(): void {
  uploadFile.value = null
}

function handleExceed(files: File[]): void {
  const first = files[0]
  if (!first) return
  uploadFile.value = first
}

async function submitUpload(): Promise<void> {
  const kbId = selectedId.value
  const file = uploadFile.value
  if (!kbId || !file) return
  uploading.value = true
  uploadPercent.value = 0
  try {
    const result = await uploadDocumentApi(
      kbId,
      file,
      uploadDescription.value,
      (percent) => {
        uploadPercent.value = percent
      },
    )
    documents.value = [result.document, ...documents.value]
    uploadVisible.value = false
    ElMessage.success('已上传，正在后台入库')
    void pollJob(result.job.job_id, kbId)
  } catch {
    // 错误提示由请求层统一处理
  } finally {
    uploading.value = false
  }
}

async function pollJob(jobId: string, kbId: string): Promise<void> {
  for (let attempt = 0; attempt < appConfig.indexJobPollMaxAttempts; attempt += 1) {
    await delay(appConfig.indexJobPollIntervalMs)
    if (disposed || selectedId.value !== kbId) return
    let job: IndexJob
    try {
      job = await getIndexJobApi(jobId)
    } catch {
      return
    }
    updateDocumentFromJob(job)
    if (isTerminal(job.status)) {
      await loadDocuments(kbId)
      if (job.status === 'failed' || job.status === 'dead') {
        ElMessage.error(job.error?.message || '文档入库失败')
      }
      return
    }
  }
  ElMessage.warning('入库仍在进行，请稍后刷新查看')
}

function updateDocumentFromJob(job: IndexJob): void {
  const target = documents.value.find((item) => item.id === job.doc_id)
  if (!target) return
  target.index_job_id = job.job_id
  target.index_progress = job.progress
  target.index_stage = job.stage
  target.index_error = job.error?.message ?? null
  if (job.status === 'running' || job.status === 'queued') {
    target.status = job.status === 'running' ? 'processing' : 'pending'
  }
}

function isTerminal(status: IndexJob['status']): boolean {
  return ['succeeded', 'failed', 'cancelled', 'dead'].includes(status)
}

async function handleDelete(row: KnowledgeDocument): Promise<void> {
  const kbId = selectedId.value
  if (!kbId) return
  try {
    await ElMessageBox.confirm(
      `确认删除文档「${row.original_filename}」及其索引数据？`,
      '删除文档',
      {
        type: 'warning',
        confirmButtonText: '删除',
        cancelButtonText: '取消',
      },
    )
  } catch {
    return
  }

  deletingIds.value = [...deletingIds.value, row.id]
  try {
    const result = await deleteDocumentApi(kbId, row.id)
    row.status = 'deleting'
    ElMessage.success('已提交删除任务')
    void pollDeletion(result.job.job_id, kbId)
  } catch {
    // 错误提示由请求层统一处理
  } finally {
    deletingIds.value = deletingIds.value.filter((id) => id !== row.id)
  }
}

async function pollDeletion(jobId: string, kbId: string): Promise<void> {
  for (let attempt = 0; attempt < appConfig.indexJobPollMaxAttempts; attempt += 1) {
    await delay(appConfig.indexJobPollIntervalMs)
    if (disposed || selectedId.value !== kbId) return
    let job: IndexJob
    try {
      job = await getIndexJobApi(jobId)
    } catch {
      return
    }
    if (isTerminal(job.status)) {
      await loadDocuments(kbId)
      if (job.status === 'failed' || job.status === 'dead') {
        ElMessage.error(job.error?.message || '文档删除失败')
      }
      return
    }
  }
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms)
  })
}

onMounted(async () => {
  try {
    await loadOptions()
    await loadKnowledgeBases()
    await loadDocuments()
  } catch {
    // 错误提示由请求层统一处理
  }
})

onBeforeUnmount(() => {
  disposed = true
})
</script>

<style scoped>
.kb-page {
  display: flex;
  flex: 1;
  flex-direction: column;
  min-width: 0;
  height: 100%;
  overflow: hidden;
  background: var(--app-chat-background);
}

.kb-page__topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex: 0 0 var(--app-layout-header-height);
  height: var(--app-layout-header-height);
  padding: 0 var(--app-space-lg);
  background-color: var(--app-bg-glass);
  border-bottom: 1px solid var(--app-border-color);
  backdrop-filter: blur(14px);
}

.kb-page__topbar-left,
.kb-page__topbar-right {
  display: flex;
  align-items: center;
  gap: var(--app-space-sm);
}

.kb-page__back {
  margin-left: var(--app-space-sm);
  color: var(--app-text-secondary);
}

.kb-page__back span {
  margin-left: var(--app-space-2xs);
}

.kb-page__divider {
  width: 1px;
  height: 18px;
  background-color: var(--app-border-color);
}

.kb-page__back {
  color: var(--app-text-secondary);
}

.kb-page__back span {
  margin-left: var(--app-space-2xs);
}

.kb-page__body {
  display: flex;
  flex: 1;
  min-height: 0;
}

/* —— 知识库列表 —— */
.kb-panel {
  display: flex;
  flex: 0 0 var(--app-sidebar-width);
  flex-direction: column;
  width: var(--app-sidebar-width);
  height: 100%;
  padding: var(--app-space-md);
  overflow: hidden;
  background-color: var(--app-bg-sidebar);
  border-right: 1px solid var(--app-border-color);
}

.kb-panel__list {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
}

.kb-panel__foot {
  display: flex;
  align-items: center;
  gap: var(--app-space-2xs);
  padding-top: var(--app-space-sm);
  margin-top: var(--app-space-sm);
  border-top: 1px solid var(--app-border-color-light);
}

.kb-panel__foot :deep(.user-menu-root) {
  flex: 1;
  min-width: 0;
}

.kb-panel__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: var(--app-space-sm);
}

.kb-panel__title {
  font-size: var(--app-font-size-base);
  font-weight: 600;
  color: var(--app-text-primary);
}

.kb-panel__create.el-button {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--app-space-xs);
  width: 100%;
  height: 38px;
  margin: 0 0 var(--app-space-sm);
  padding: 0 var(--app-space-md);
  font-size: var(--app-font-size-base);
  font-weight: 400;
  line-height: 1;
  color: var(--app-text-primary);
  background-color: var(--app-bg-card);
  border: 1px solid var(--app-border-color);
  border-radius: var(--app-radius-md);
  box-shadow: var(--app-shadow-bubble);
}

.kb-panel__create.el-button:hover {
  color: var(--app-color-primary);
  background-color: var(--app-color-primary-soft);
  border-color: var(--app-color-primary-soft);
}

.kb-panel__create :deep(.el-icon) {
  position: static;
  margin: 0;
  color: var(--app-color-primary);
  transform: none;
}

.kb-panel__create :deep(span) {
  display: inline-flex;
  align-items: center;
  gap: var(--app-space-xs);
  margin-left: 0;
  line-height: 1;
}

.kb-panel__loading,
.kb-panel__empty {
  padding: var(--app-space-lg) var(--app-space-sm);
  color: var(--app-text-secondary);
  text-align: center;
}

.kb-panel__empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--app-space-xs);
}

.kb-item {
  display: flex;
  flex-direction: column;
  gap: var(--app-space-2xs);
  width: 100%;
  padding: var(--app-space-sm) var(--app-space-md);
  margin-bottom: var(--app-space-xs);
  text-align: left;
  cursor: pointer;
  background-color: transparent;
  border: 1px solid transparent;
  border-radius: var(--app-radius-md);
  transition:
    background-color var(--app-transition-fast),
    border-color var(--app-transition-fast);
}

.kb-item:hover {
  background-color: var(--app-bg-hover);
}

.kb-item--active {
  background-color: var(--app-bg-active);
  border-color: var(--app-color-primary-soft);
}

.kb-item__name {
  overflow: hidden;
  font-size: var(--app-font-size-base);
  font-weight: 500;
  color: var(--app-text-primary);
  text-overflow: ellipsis;
  white-space: nowrap;
}

.kb-item__desc {
  overflow: hidden;
  font-size: var(--app-font-size-xs);
  color: var(--app-text-secondary);
  text-overflow: ellipsis;
  white-space: nowrap;
}

.kb-item__meta {
  display: flex;
  align-items: center;
  gap: var(--app-space-sm);
  font-size: var(--app-font-size-xs);
  color: var(--app-text-muted);
}

/* —— 文档面板 —— */
.doc-panel {
  display: flex;
  flex: 1;
  flex-direction: column;
  min-width: 0;
  padding: var(--app-space-lg) var(--app-space-xl);
  overflow-y: auto;
}

.doc-panel__head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: var(--app-space-lg);
}

.doc-panel__identity {
  min-width: 0;
}

.doc-panel__title {
  font-size: var(--app-font-size-lg);
  font-weight: 600;
  color: var(--app-text-primary);
}

.doc-panel__subtitle {
  margin-top: var(--app-space-2xs);
  font-size: var(--app-font-size-sm);
  color: var(--app-text-secondary);
}

.doc-panel__facts {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: var(--app-space-sm);
  margin-top: var(--app-space-xs);
  font-size: var(--app-font-size-xs);
  color: var(--app-text-muted);
}

.doc-panel__upload-hint {
  margin: var(--app-space-md) 0 var(--app-space-sm);
  font-size: var(--app-font-size-xs);
  color: var(--app-text-muted);
}

.doc-panel__empty {
  display: flex;
  flex: 1;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: var(--app-space-sm);
  color: var(--app-text-secondary);
  text-align: center;
}

.doc-panel__empty h2 {
  font-size: var(--app-font-size-lg);
  font-weight: 600;
  color: var(--app-text-primary);
}

.doc-table {
  flex: 1;
  min-height: 0;
  --el-table-bg-color: transparent;
  --el-table-tr-bg-color: transparent;
  --el-table-header-bg-color: var(--app-bg-muted);
  --el-table-row-hover-bg-color: var(--app-bg-hover);
  --el-table-border-color: var(--app-border-color-light);
}

.doc-cell {
  display: flex;
  align-items: flex-start;
  gap: var(--app-space-sm);
  min-width: 0;
}

.doc-cell__icon {
  margin-top: var(--app-space-2xs);
  color: var(--app-color-primary);
}

.doc-cell__body {
  display: flex;
  flex-direction: column;
  gap: var(--app-space-2xs);
  min-width: 0;
}

.doc-cell__name {
  overflow: hidden;
  font-size: var(--app-font-size-sm);
  color: var(--app-text-primary);
  text-overflow: ellipsis;
  white-space: nowrap;
}

.doc-cell__meta {
  overflow: hidden;
  font-size: var(--app-font-size-xs);
  color: var(--app-text-muted);
  text-overflow: ellipsis;
  white-space: nowrap;
}

.doc-status {
  display: flex;
  flex-direction: column;
  gap: var(--app-space-2xs);
}

.doc-number {
  font-variant-numeric: tabular-nums;
  color: var(--app-text-regular);
}

.doc-number-sep {
  margin: 0 var(--app-space-2xs);
  color: var(--app-text-muted);
}

.doc-time {
  font-size: var(--app-font-size-xs);
  color: var(--app-text-secondary);
}

/* —— 弹窗 —— */
.kb-form-hint,
.upload-tip {
  margin: var(--app-space-2xs) 0 0;
  font-size: var(--app-font-size-xs);
  color: var(--app-text-muted);
}

.upload-icon {
  font-size: 44px;
  color: var(--app-color-primary);
}

.upload-text {
  margin-top: var(--app-space-sm);
  color: var(--app-text-regular);
}

.upload-form {
  margin-top: var(--app-space-md);
}

.upload-progress {
  margin-top: var(--app-space-sm);
}

@media (max-width: 900px) {
  .kb-page__topbar {
    padding-inline: var(--app-space-md);
  }

  .kb-page__divider,
  .kb-page__topbar-left :deep(.brand-mark__name) {
    display: none;
  }

  .kb-page__back span {
    display: none;
  }

  .kb-page__body {
    flex-direction: column;
  }

  .kb-panel {
    flex: 0 0 auto;
    width: 100%;
    max-height: 260px;
    border-right: none;
    border-bottom: 1px solid var(--app-border-color);
  }

  .doc-panel {
    padding: var(--app-space-md);
  }
}
</style>
