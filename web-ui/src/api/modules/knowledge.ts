import type {
  CreateKnowledgeBaseParams,
  DeleteKnowledgeBaseResult,
  DocumentListResult,
  IndexJob,
  KnowledgeBase,
  KnowledgeBaseListResult,
  KnowledgeOptions,
  UploadDocumentResult,
} from '@/api/types'
import http from '@/utils/request'

/** 管理员：知识库列表。 */
export function listKnowledgeBasesApi(): Promise<KnowledgeBaseListResult> {
  return http.get<KnowledgeBaseListResult>('/knowledge-bases')
}

/** 管理员：创建知识库。 */
export function createKnowledgeBaseApi(
  params: CreateKnowledgeBaseParams,
): Promise<KnowledgeBase> {
  return http.post<KnowledgeBase>('/knowledge-bases', params)
}

/** 管理员：启用/停用知识库。 */
export function updateKnowledgeBaseStatusApi(
  kbId: string,
  status: 'active' | 'disabled',
): Promise<KnowledgeBase> {
  return http.patch<KnowledgeBase>(`/knowledge-bases/${kbId}`, { status })
}

/** 管理员：删除知识库并提交文档索引清理任务；需要二次输入管理员密码。 */
export function deleteKnowledgeBaseApi(
  kbId: string,
  password: string,
): Promise<DeleteKnowledgeBaseResult> {
  return http.delete<DeleteKnowledgeBaseResult>(`/knowledge-bases/${kbId}`, {
    data: { password },
    silent: true,
  })
}

/** 管理员：上传限制、启用类型与默认权限码。 */
export function getKnowledgeOptionsApi(): Promise<KnowledgeOptions> {
  return http.get<KnowledgeOptions>('/knowledge-bases/options')
}

/** 管理员：知识库文档列表。 */
export function listDocumentsApi(kbId: string): Promise<DocumentListResult> {
  return http.get<DocumentListResult>(`/knowledge-bases/${kbId}/documents`)
}

/** 管理员：上传文档并提交后台入库。 */
export function uploadDocumentApi(
  kbId: string,
  file: File,
  description: string,
  onProgress?: (percent: number) => void,
): Promise<UploadDocumentResult> {
  const form = new FormData()
  form.append('file', file)
  if (description.trim()) form.append('description', description.trim())
  return http.post<UploadDocumentResult>(`/knowledge-bases/${kbId}/documents`, form, {
    onUploadProgress: (event) => {
      if (event.total && onProgress) {
        onProgress(Math.round((event.loaded / event.total) * 100))
      }
    },
  })
}

/** 管理员：提交文档删除任务。 */
export function deleteDocumentApi(
  kbId: string,
  docId: string,
): Promise<{ job: IndexJob }> {
  return http.delete<{ job: IndexJob }>(`/knowledge-bases/${kbId}/documents/${docId}`)
}

/** 管理员：轮询入库/删除任务。 */
export function getIndexJobApi(jobId: string): Promise<IndexJob> {
  return http.get<IndexJob>(`/index-jobs/${jobId}`)
}
