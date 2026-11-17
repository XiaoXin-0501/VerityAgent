<template>
  <section class="kb-page">
    <div class="kb-page__body">
      <aside class="kb-panel">
        <div class="kb-panel__brand">
          <BrandMark />
          <div class="kb-panel__brand-actions">
            <el-tooltip content="返回对话" placement="bottom">
              <el-button
                circle
                text
                aria-label="返回对话"
                @click="router.push({ name: 'home' })"
              >
                <el-icon><ArrowLeft /></el-icon>
              </el-button>
            </el-tooltip>
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
        </div>

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
              <el-tag
                size="small"
                :type="kbStatusTagType(item.status)"
                effect="light"
              >
                {{ statusLabel(item.status) }}
              </el-tag>
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
                <el-tag
                  size="small"
                  :type="kbStatusTagType(selectedKb.status)"
                  effect="light"
                >
                  {{ statusLabel(selectedKb.status) }}
                </el-tag>
                <el-tag size="small" effect="plain" type="info">
                  权限：{{ selectedKb.required_permission_code }}
                </el-tag>
                <el-tag size="small" effect="plain" type="success">
                  领域：{{ selectedKb.domain_profile || '全局默认' }}
                </el-tag>
                <el-tag size="small" effect="plain" type="info">
                  {{ selectedKb.embedding_model_id }} · {{ selectedKb.embedding_dimensions }} 维
                </el-tag>
                <span>索引版本 v{{ selectedKb.index_version }}</span>
              </div>
            </div>
            <div class="doc-panel__actions">
              <el-button class="doc-action doc-action--primary" @click="openUpload">
                <el-icon><UploadFilled /></el-icon>
                <span>上传文档</span>
              </el-button>
              <el-button
                class="doc-action"
                text
                :loading="statusUpdating"
                @click="toggleKbStatus"
              >
                <el-icon>
                  <VideoPause v-if="selectedKb.status === 'active'" />
                  <CircleCheck v-else />
                </el-icon>
                <span>{{ selectedKb.status === 'active' ? '禁用' : '启用' }}</span>
              </el-button>
              <el-button
                class="doc-action doc-action--danger"
                text
                :loading="deletingKb"
                @click="openDeleteDialog"
              >
                <el-icon><Delete /></el-icon>
                <span>删除</span>
              </el-button>
            </div>
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

            <el-table-column label="状态" width="150" align="center" header-align="center">
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

            <el-table-column
              label="切片 / Token"
              width="130"
              align="center"
              header-align="center"
            >
              <template #default="{ row }">
                <span class="doc-number">{{ row.chunk_count ?? '-' }}</span>
                <span class="doc-number-sep">/</span>
                <span class="doc-number">{{ row.token_count ?? '-' }}</span>
              </template>
            </el-table-column>

            <el-table-column label="更新时间" width="150" align="center" header-align="center">
              <template #default="{ row }">
                <span class="doc-time">{{ formatDateTime(row.updated_at) }}</span>
              </template>
            </el-table-column>

            <el-table-column
              label="操作"
              width="90"
              align="center"
              header-align="center"
              fixed="right"
            >
              <template #default="{ row }">
                <el-tooltip content="删除文档" placement="top">
                  <el-button
                    class="doc-delete"
                    text
                    type="danger"
                    :loading="deletingIds.includes(row.id)"
                    :disabled="row.status === 'deleting' || row.status === 'deleted'"
                    aria-label="删除文档"
                    @click="handleDelete(row as KnowledgeDocument)"
                  >
                    <el-icon><Delete /></el-icon>
                  </el-button>
                </el-tooltip>
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
        <el-form-item label="领域画像">
          <el-select
            v-model="createForm.domain_profile"
            clearable
            placeholder="使用全局默认领域"
            style="width: 100%"
          >
            <el-option
              v-for="profile in availableDomainProfiles"
              :key="profile"
              :label="profile"
              :value="profile"
            />
          </el-select>
          <p class="kb-form-hint">
            入库建图时使用该领域词典与示例；修改后需重建索引。
          </p>
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
    <el-dialog
      v-model="deleteVisible"
      title="删除知识库"
      width="480px"
      :close-on-click-modal="false"
      @closed="resetDeleteDialog"
    >
      <div class="delete-dialog">
        <div class="delete-dialog__warning">
          <span class="delete-dialog__icon">
            <el-icon :size="20"><WarningFilled /></el-icon>
          </span>
          <div>
            <p class="delete-dialog__title">
              确认删除「{{ deleteTarget?.name }}」？
            </p>
            <p class="delete-dialog__text">
              知识库会立即从检索和管理列表隐藏，其下文档的向量、图与对象存储数据
              将异步清理。此操作无法在界面中恢复。
            </p>
          </div>
        </div>

        <el-form label-position="top" @submit.prevent="confirmDelete">
          <el-form-item label="管理员密码" :error="deleteError">
            <el-input
              v-model="deletePassword"
              type="password"
              show-password
              autocomplete="current-password"
              placeholder="再次输入当前管理员密码"
              @keyup.enter="confirmDelete"
            />
          </el-form-item>
        </el-form>
      </div>

      <template #footer>
        <el-button :disabled="deletingKb" @click="deleteVisible = false">取消</el-button>
        <el-button
          type="danger"
          :loading="deletingKb"
          :disabled="!deletePassword"
          @click="confirmDelete"
        >
          确认删除
        </el-button>
      </template>
    </el-dialog>
  </section>
</template>

<script setup lang="ts">
import 'element-plus/es/components/message-box/style/css'

import {
  ArrowLeft,
  CircleCheck,
  Collection,
  Delete,
  Document,
  Plus,
  Refresh,
  UploadFilled,
  VideoPause,
  WarningFilled,
} from '@element-plus/icons-vue'
import type { UploadFile, UploadRawFile } from 'element-plus'
import { ElMessageBox } from 'element-plus'

import BrandMark from '@/components/layout/BrandMark.vue'
import UserMenu from '@/components/layout/UserMenu.vue'
import ThemeToggle from '@/components/common/ThemeToggle.vue'
import {
  createKnowledgeBaseApi,
  deleteDocumentApi,
  deleteKnowledgeBaseApi,
  getIndexJobApi,
  getKnowledgeOptionsApi,
  listDocumentsApi,
  listKnowledgeBasesApi,
  updateKnowledgeBaseStatusApi,
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
  statusLabel,
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
const statusUpdating = ref(false)
const deletingKb = ref(false)
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
  domain_profile: '',
})
const uploadFile = ref<File | null>(null)
const uploadDescription = ref('')
const deleteVisible = ref(false)
const deleteTarget = ref<KnowledgeBase | null>(null)
const deletePassword = ref('')
const deleteError = ref('')

let disposed = false

const selectedKb = computed(
  () => knowledgeBases.value.find((item) => item.id === selectedId.value) ?? null,
)

const availableDomainProfiles = computed(
  () => options.value?.domain_profiles ?? [],
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

function kbStatusTagType(status: string): 'success' | 'warning' | 'danger' | 'info' {
  switch (status) {
    case 'active':
      return 'success'
    case 'disabled':
      return 'warning'
    case 'deleting':
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
  createForm.domain_profile = options.value?.default_domain_profile ?? ''
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
      domain_profile: createForm.domain_profile || null,
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

async function toggleKbStatus(): Promise<void> {
  const kb = selectedKb.value
  if (!kb || statusUpdating.value) return
  const target = kb.status === 'active' ? 'disabled' : 'active'
  statusUpdating.value = true
  try {
    const updated = await updateKnowledgeBaseStatusApi(kb.id, target)
    knowledgeBases.value = knowledgeBases.value.map((item) =>
      item.id === updated.id ? updated : item,
    )
    ElMessage.success(target === 'active' ? '知识库已启用' : '知识库已禁用')
  } catch {
    // 错误提示由请求层统一处理
  } finally {
    statusUpdating.value = false
  }
}

function openDeleteDialog(): void {
  const kb = selectedKb.value
  if (!kb) return
  deleteTarget.value = kb
  deletePassword.value = ''
  deleteError.value = ''
  deleteVisible.value = true
}

function resetDeleteDialog(): void {
  deleteTarget.value = null
  deletePassword.value = ''
  deleteError.value = ''
}

async function confirmDelete(): Promise<void> {
  const kb = deleteTarget.value
  const password = deletePassword.value
  if (!kb || !password || deletingKb.value) return
  deletingKb.value = true
  deleteError.value = ''
  try {
    const result = await deleteKnowledgeBaseApi(kb.id, password)
    deleteVisible.value = false
    knowledgeBases.value = knowledgeBases.value.filter((item) => item.id !== kb.id)
    const next = knowledgeBases.value[0]
    selectedId.value = next?.id ?? null
    documents.value = []
    if (next) await loadDocuments(next.id)
    if (result.failed > 0) {
      ElMessage.warning(
        `知识库已删除，${result.failed} 个文档的清理任务提交失败，需要人工检查`,
      )
    } else {
      ElMessage.success(`知识库已删除，已提交 ${result.submitted} 个文档清理任务`)
    }
  } catch (error) {
    deleteError.value =
      error instanceof Error ? error.message : '删除失败，请稍后重试'
  } finally {
    deletingKb.value = false
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

.kb-page__body {
  display: flex;
  flex: 1;
  min-height: 0;
}

/* —— 知识库侧栏 —— */
.kb-panel__brand {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--app-space-sm);
  margin-bottom: var(--app-space-md);
}

.kb-panel__brand-actions {
  display: flex;
  align-items: center;
  gap: var(--app-space-2xs);
}

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

.doc-panel__actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: flex-end;
  gap: var(--app-space-sm);
}

.doc-action.el-button {
  height: 34px;
  padding: 0 var(--app-space-md);
  font-size: var(--app-font-size-sm);
  color: var(--app-text-regular);
  background-color: var(--app-bg-card);
  border: 1px solid var(--app-border-color);
  border-radius: var(--app-radius-pill);
  box-shadow: var(--app-shadow-bubble);
  transition:
    color var(--app-transition-fast),
    background-color var(--app-transition-fast),
    border-color var(--app-transition-fast),
    transform var(--app-transition-fast);
}

.doc-action.el-button:hover {
  color: var(--app-color-primary);
  background-color: var(--app-color-primary-soft);
  border-color: var(--app-color-primary-soft);
  transform: translateY(-1px);
}

.doc-action :deep(.el-icon) {
  margin-right: var(--app-space-xs);
  color: var(--app-color-primary);
}

.doc-action--primary.el-button {
  color: var(--app-color-primary);
  background-color: var(--app-color-primary-soft);
  border-color: transparent;
}

.doc-action--primary.el-button:hover {
  color: var(--app-text-on-primary);
  background-color: var(--app-color-primary);
  border-color: var(--app-color-primary);
}

.doc-action--primary :deep(.el-icon) {
  color: currentColor;
}

.doc-action--danger.el-button {
  color: var(--app-text-regular);
}

.doc-action--danger :deep(.el-icon) {
  color: var(--app-color-danger);
}

.doc-action--danger.el-button:hover {
  color: var(--app-color-danger);
  background-color: var(--app-color-danger-soft);
  border-color: var(--app-color-danger-soft);
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
  align-items: center;
  gap: var(--app-space-2xs);
}

.doc-status :deep(.el-progress) {
  width: 80px;
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
  font-variant-numeric: tabular-nums;
}

.doc-delete {
  width: 28px;
  height: 28px;
  padding: 0;
  border-radius: var(--app-radius-sm);
  color: var(--app-text-muted);
}

.doc-delete:hover:not(.is-disabled) {
  color: var(--app-color-danger);
  background-color: var(--app-color-danger-soft);
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

.delete-dialog__warning {
  display: flex;
  gap: var(--app-space-sm);
  padding: var(--app-space-md);
  margin-bottom: var(--app-space-md);
  background-color: var(--app-color-danger-soft);
  border: 1px solid var(--app-color-danger-soft);
  border-radius: var(--app-radius-md);
}

.delete-dialog__icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex: 0 0 auto;
  width: 34px;
  height: 34px;
  color: var(--app-color-danger);
  background-color: var(--app-bg-card);
  border-radius: var(--app-radius-pill);
}

.delete-dialog__title {
  font-size: var(--app-font-size-base);
  font-weight: 600;
  color: var(--app-text-primary);
}

.delete-dialog__text {
  margin-top: var(--app-space-2xs);
  font-size: var(--app-font-size-sm);
  line-height: var(--app-line-height-base);
  color: var(--app-text-secondary);
}

.upload-form {
  margin-top: var(--app-space-md);
}

.upload-progress {
  margin-top: var(--app-space-sm);
}

@media (max-width: 900px) {
  .kb-page__body {
    flex-direction: column;
  }

  .kb-panel {
    flex: 0 0 auto;
    width: 100%;
    max-height: 320px;
    border-right: none;
    border-bottom: 1px solid var(--app-border-color);
  }

  .doc-panel {
    padding: var(--app-space-md);
  }

  .doc-panel__head {
    flex-direction: column;
  }

  .doc-panel__actions {
    justify-content: flex-start;
  }
}
</style>
