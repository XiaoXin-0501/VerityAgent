"""知识库可读目录适配器。

Agent 主循环（INJECTED RAG）与 ``search_kb`` 工具都需要在用户没有显式
提供 ``kb_ids`` 时解析“当前用户可读的知识库”。权限事实源只有一处：
``knowledge_bases.required_permission_code``。本适配器把该规则收口成两个
只读方法，避免工具层和 Agent 层各自拼 ``kb:read:<id>`` 造成漂移。

边界：

- 只解析知识库 id，不做检索、不读 chunk；
- ``kb:read:*`` 视为部署级通配，可读所有 active 知识库；
- 其它情况一律走 PG 的 ``list_accessible / is_accessible``，保证自定义
  权限码（非 ``kb:read:<id>``）同样生效；
- 数据库异常向上抛出，由调用方决定降级或失败，不在这里静默放行。
"""

from __future__ import annotations

from uuid import UUID

from store.pg.models.enums import KnowledgeBaseStatus
from store.pg.repositories import KnowledgeBaseRepository

__all__ = ["PgKbCatalogAdapter"]

# 单次解析的知识库数量上限；与 search_kb 参数 schema 的上限保持一致。
# 超过上限时不静默截断而不告知：调用方按 limit 取最新创建的库。
DEFAULT_KB_LIMIT = 20


class PgKbCatalogAdapter:
    """把知识库权限投影成 Agent/工具可消费的只读目录。"""

    def __init__(self, repository: KnowledgeBaseRepository | None = None) -> None:
        """保存知识库仓库；不主动建立连接。"""
        self.repository = repository or KnowledgeBaseRepository()

    async def list_readable(
        self,
        *,
        user_id: UUID,
        permissions: frozenset[str],
        limit: int = DEFAULT_KB_LIMIT,
    ) -> list[UUID]:
        """返回用户可读的 active 知识库 id（按创建时间倒序，最多 limit 个）。"""
        limit = max(1, int(limit))
        if "kb:read:*" in permissions:
            # 通配权限不在 user_permissions 里逐库登记，直接列 active 库。
            rows = await self.repository.list(
                status=KnowledgeBaseStatus.ACTIVE,
                limit=limit,
            )
        else:
            rows = await self.repository.list_accessible(user_id, limit=limit)
        return [item.id for item in rows]

    async def can_read(
        self,
        *,
        user_id: UUID,
        permissions: frozenset[str],
        kb_id: UUID,
    ) -> bool:
        """按知识库实际 ``required_permission_code`` 校验读取权限。

        ``kb:read:*`` 是部署级通配；其余情况查 PG，保证自定义权限码与
        检索流水线使用同一套判断。
        """
        if "kb:read:*" in permissions:
            return True
        return await self.repository.is_accessible(user_id, kb_id)
