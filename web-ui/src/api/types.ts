/**
 * 前端公共 DTO。
 *
 * 字段与后端 FastAPI / Agent 公共 DTO 保持一致（snake_case），
 * 不直接消费 AgentRun/ContextPlan 等内部模型。
 */

// ——————————————————— 认证 ———————————————————

export interface LoginParams {
  username: string
  password: string
}

export interface UserInfo {
  id: string
  username: string
  role: string
  email: string | null
}

export interface LoginResult {
  access_token: string
  token_type: string
  expires_in: number
  user: UserInfo
}

export interface MeResult {
  user: UserInfo
  permissions: string[]
}

// ——————————————————— 会话 ———————————————————

export interface ConversationSummary {
  id: string
  title: string
  status: string
  last_message_at: string | null
  turn_count: number
}

export interface ConversationListResult {
  items: ConversationSummary[]
  next_cursor: string | null
}

export interface CreateConversationResult {
  id: string
  title: string
  status: string
  created_at: string
}

export interface SendMessageParams {
  client_message_id: string
  content: string
  stream: boolean
  images?: ImageRef[]
  files?: FileRef[]
  model_id?: string | null
  rag_mode?: string | null
  tool_choice?: Record<string, unknown> | null
  max_tool_rounds?: number | null
  metadata?: Record<string, unknown>
}

export interface SendMessageResult {
  message_id: string | null
  client_message_id: string
  run_id: string
  turn_id: string
  status: string
  events_url?: string
  stream_url?: string
  result?: AgentResult
}

// ——————————————————— 消息 ———————————————————

export type MessageRole = 'user' | 'assistant' | 'tool' | 'system'
export type MessageStatus = 'streaming' | 'completed' | 'partial' | 'failed'

export interface Thinking {
  text: string
  signature?: string | null
  redacted?: boolean
}

export interface ToolCall {
  call_id: string
  name: string
  arguments?: Record<string, unknown>
  raw_arguments?: string | null
  status?: string
}

export interface ImageRef {
  type?: 'image'
  url?: string | null
  data?: string | null
  mime_type?: string | null
  detail?: string | null
}

export interface FileRef {
  type?: 'file'
  file_id: string
  filename?: string | null
  url?: string | null
}

export interface Artifact {
  type?: 'artifact'
  kind?: string
  url: string
  filename?: string | null
}

export interface ChatMessage {
  id: string
  seq_no: number | null
  role: MessageRole
  status: MessageStatus
  content: string
  thinking: Thinking[]
  tool_calls: ToolCall[]
  json?: unknown
  name?: string | null
  tool_call_id?: string | null
  images?: ImageRef[]
  files?: FileRef[]
  artifacts?: Artifact[]
  created_at: string | null
}

export interface MessageListResult {
  items: ChatMessage[]
  next_cursor: string | null
}

// ——————————————————— 运行 / SSE ———————————————————

export type RunStatus =
  | 'queued'
  | 'running'
  | 'waiting_tool'
  | 'finalizing'
  | 'completed'
  | 'failed'
  | 'cancelled'
  | 'expired'
  | 'conflict'

export interface Usage {
  input_tokens: number
  output_tokens: number
  total_tokens: number
  cached_input_tokens: number
  reasoning_tokens: number
  source: string
}

export interface AgentError {
  code: string
  message: string
  user_visible: boolean
  retryable: boolean
  details: Record<string, unknown>
  cause?: string | null
  partial_result?: boolean
}

export interface AgentWarning {
  code: string
  message: string
  user_visible: boolean
  details: Record<string, unknown>
}

export interface AgentResult {
  run_id: string
  conversation_id: string
  turn_id: string
  status: RunStatus
  user_message_id: string
  assistant_message: ChatMessage | null
  message_ids: string[]
  usage: Usage
  warnings: AgentWarning[]
  error: AgentError | null
  duration_ms: number
}

export interface AgentRunView {
  run_id: string
  turn_id: string
  conversation_id: string
  status: RunStatus
  turn_status: string
  user_message_id: string | null
  assistant_message_id: string | null
  assistant_status: MessageStatus | null
  content: string
  thinking: Thinking[]
  tool_calls: ToolCall[]
  usage: Usage | null
  last_event_id: string | null
  result: AgentResult | null
  error: AgentError | null
  created_at: string
  updated_at: string
}

export type AgentEventType =
  | 'run.started'
  | 'run.completed'
  | 'run.failed'
  | 'run.cancelled'
  | 'turn.started'
  | 'turn.completed'
  | 'turn.failed'
  | 'turn.cancelled'
  | 'assistant.started'
  | 'assistant.delta'
  | 'assistant.tool_calls'
  | 'assistant.completed'
  | 'tool.started'
  | 'tool.result'
  | 'warning'
  | 'usage'
  | 'heartbeat'
  | 'done'

export interface AgentEvent<T = Record<string, unknown>> {
  seq: number
  event_type: AgentEventType
  run_id: string
  turn_id: string
  message_id: string | null
  call_id: string | null
  data: T
  created_at: string
}

export interface CancelRunResult {
  run_id: string
  status: string
}

// ——————————————————— 知识库管理（管理员） ———————————————————

export interface KnowledgeBase {
  id: string
  name: string
  description: string | null
  required_permission_code: string
  embedding_model_id: string
  embedding_dimensions: number
  embedding_version: string
  index_version: number
  status: string
  created_at: string
  updated_at: string
}

export interface KnowledgeBaseListResult {
  items: KnowledgeBase[]
  next_cursor: string | null
}

export interface CreateKnowledgeBaseParams {
  name: string
  description?: string | null
  required_permission_code: string
}

export type DocumentStatus =
  | 'pending'
  | 'processing'
  | 'completed'
  | 'failed'
  | 'deleting'
  | 'deleted'

export interface KnowledgeDocument {
  id: string
  kb_id: string
  original_filename: string
  description: string | null
  extension: string
  size_bytes: number
  sha256: string | null
  status: DocumentStatus
  token_count: number | null
  chunk_count: number | null
  index_job_id: string | null
  index_stage: string | null
  index_progress: number | null
  index_error: string | null
  created_at: string
  updated_at: string
  parsed_at: string | null
}

export interface DocumentListResult {
  items: KnowledgeDocument[]
  next_cursor: string | null
}

export type IndexingStatus =
  | 'queued'
  | 'running'
  | 'succeeded'
  | 'failed'
  | 'cancelled'
  | 'dead'

export interface IndexJob {
  job_id: string
  kb_id: string
  doc_id: string
  mode: string
  status: IndexingStatus
  stage: string
  progress: number
  attempts: number
  error: { code?: string; message?: string; details?: Record<string, unknown> } | null
  warnings: Array<{ code: string; message: string }>
  index_version_before: number
  index_version_after: number
  counts: Record<string, number>
  created_at: string
  started_at: string | null
  ended_at: string | null
}

export interface UploadDocumentResult {
  document: KnowledgeDocument
  job: IndexJob
}

export interface KnowledgeOptions {
  enabled_doc_types: string[]
  max_file_bytes: number
  default_permission_code: string
  embedding_model_id: string | null
  embedding_dimensions: number
  embedding_version: string
}
