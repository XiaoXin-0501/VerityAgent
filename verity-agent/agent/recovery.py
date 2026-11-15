"""启动期崩溃恢复：清理过期 run 租约。"""

from __future__ import annotations

import logging

from store.pg.models.enums import MessageStatus as PgMessageStatus
from store.pg.repositories import ConversationMessageRepository, ConversationRepository

__all__ = ["recover_expired_leases", "recover_streaming_messages"]

logger = logging.getLogger(__name__)


async def recover_expired_leases(
    repository: ConversationRepository | None = None,
    *,
    limit: int = 100,
) -> int:
    """扫描过期租约并清空 current_run_id；返回恢复数量。"""
    repo = repository or ConversationRepository()
    rows = await repo.list_expired_leases(limit=limit)
    recovered = 0
    for conversation in rows:
        try:
            cleared = await repo.clear_expired_run(conversation.id)
        except Exception:  # noqa: BLE001 - 单会话失败不阻断启动
            logger.warning(
                "清理过期 run 租约失败: %s",
                conversation.id,
                exc_info=True,
            )
            continue
        if cleared is not None:
            recovered += 1
    if recovered:
        logger.info("已恢复 %s 个过期 run 租约", recovered)
    return recovered


async def recover_streaming_messages(
    repository: ConversationMessageRepository | None = None,
    *,
    limit: int = 200,
) -> int:
    """把崩溃残留的 streaming 消息收敛为 failed；返回处理条数。"""
    repo = repository or ConversationMessageRepository()
    rows = await repo.list_streaming(limit=limit)
    recovered = 0
    for row in rows:
        try:
            await repo.update_status(row.id, PgMessageStatus.FAILED)
            recovered += 1
        except Exception:  # noqa: BLE001 - 单条失败不阻断恢复
            logger.warning("收敛 streaming 消息失败: %s", row.id, exc_info=True)
    if recovered:
        logger.info("已收敛 %s 条 streaming 消息", recovered)
    return recovered
