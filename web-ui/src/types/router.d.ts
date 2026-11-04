import 'vue-router'

declare module 'vue-router' {
  interface RouteMeta {
    /** 页面标题，用于 document.title 与面包屑 */
    title?: string
    /** 无需登录即可访问 */
    public?: boolean
    /** 需要登录（默认受保护，保留字段便于扩展） */
    requiresAuth?: boolean
  }
}
