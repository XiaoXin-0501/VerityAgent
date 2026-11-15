"""启动期崩溃恢复：清理过期 run 租约。"""

from __future__ import annotations

import logging
from typing import Any

from agent.message.codec import decode_stored_row, encode_content
from agent.message.enum import MessageStatus as AgentMessageStatus
from agent.message.models import ToolMessage
from store.pg.models.enums import MessageRole as PgMessageRole
from store.pg.models.enums import MessageStatus as PgMessageStatus
from store.pg.repositories import ConversationMessageRepository, ConversationRepository

__all__ = ["recover_expired_leases", "recover_streaming_messages", "recover_missing_tool_messages"]

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


async def recover_missing_tool_messages(
    repository: ConversationMessageRepository | None = None,
    *,
    limit: int = 100,
) -> int:
    """为崩溃前未闭合的 tool_calls 补失败占位 ToolMessage。"""
    repo = repository or ConversationMessageRepository()
    assistants = await repo.list_tool_call_assistants(limit=limit)
    tool_cache: dict[Any, set[str]] = {}
    recovered = 0
    for row in assistants:
        try:
            message = decode_stored_row(row.model_dump()).message
        except Exception:  # noqa: BLE001 - 坏行跳过
            continue
        calls = list(getattr(message, "tool_calls", []) or [])
        if not calls:
            continue
        conversation_id = row.conversation_id
        if conversation_id not in tool_cache:
            tool_rows = await repo.list_tool_messages(conversation_id)
            ids: set[str] = set()
            for tool_row in tool_rows:
                try:
                    tool_message = decode_stored_row(tool_row.model_dump()).message
                except Exception:  # noqa: BLE001
                    continue
                tool_call_id = getattr(tool_message, "tool_call_id", None)
                if tool_call_id:
                    ids.add(str(tool_call_id))
            tool_cache[conversation_id] = ids
        for call in calls:
            if call.call_id in tool_cache[conversation_id]:
                continue
            placeholder = ToolMessage(
                conversation_id=conversation_id,
                tool_call_id=call.call_id,
                name=call.name,
                content="崩溃恢复：工具调用未完成，未产生结果。",
                status=AgentMessageStatus.FAILED,
            )
            await repo.append(
                conversation_id,
                PgMessageRole.TOOL,
                encode_content(placeholder),
                status=PgMessageStatus.FAILED,
            )
            tool_cache[conversation_id].add(call.call_id)
            recovered += 1
    if recovered:
        logger.info("已补齐 %s 条缺失 ToolMessage", recovered)
    return recovered
