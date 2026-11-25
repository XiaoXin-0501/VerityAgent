"""内置 ``get_tool_result``：按消息 id 回放历史工具调用结果。

历史消息只保留工具名、参数与状态，完整结果留在 ``conversation_messages``。
当模型确实需要当时的检索片段、表格或错误详情时，用历史工具摘要里的
``message_id`` 调用本工具取回原文；只允许读取当前会话内的工具消息。
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from pydantic import BaseModel

from agent.message.codec import decode_row
from agent.message.models import RawMessage, ToolMessage
from agent.tools.declaration import ToolBuildContext, ToolDeclaration
from agent.tools.factory import SimpleToolFactory
from agent.tools.models import ToolContext, ToolError, ToolResult
from agent.tools.services import ToolServiceKey, get_tool_service
from exception.tool import ToolErrorCode
from store.pg.repositories import ConversationMessageRepository
from utils.serialization import json_dumps

__all__ = [
    "GetToolResultArgs",
    "ToolHistoryExecutor",
    "ToolHistoryFactory",
]

logger = logging.getLogger(__name__)


class GetToolResultArgs(BaseModel):
    """``get_tool_result`` 的 LLM 参数模型。"""

    # 来自历史工具摘要的 message_id；只接受当前会话内的工具消息。
    message_id: UUID


class ToolHistoryExecutor:
    """按消息 id 读取并展开历史 ToolMessage。"""

    def __init__(
        self,
        *,
        repository: ConversationMessageRepository | None = None,
    ) -> None:
        """保存可选仓库；未注入时按默认连接池构造。"""
        self.repository = repository

    def _repository(self, context: ToolContext) -> ConversationMessageRepository:
        """优先使用应用共享仓库；测试可注入内存实现。"""
        if self.repository is not None:
            return self.repository
        shared = get_tool_service(
            context.services, ToolServiceKey.CONVERSATION_MESSAGES
        )
        if shared is not None:
            return shared
        return ConversationMessageRepository()

    async def __call__(
        self,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """读取消息并返回原始工具结果。"""
        parsed = GetToolResultArgs.model_validate(arguments)
        try:
            row = await self._repository(context).get(parsed.message_id)
        except Exception as exc:  # noqa: BLE001 - 工具边界统一转结构化错误
            logger.warning("工具结果回放读取失败: %s", exc, exc_info=True)
            return ToolResult.failure(
                call_id=context.call_id,
                tool_name=context.tool_name,
                error=ToolError(
                    code=ToolErrorCode.UPSTREAM_ERROR,
                    message="工具结果回放读取失败，请稍后重试",
                    retryable=True,
                    cause=str(exc),
                ),
            )

        if (
            row is None
            or str(row.conversation_id) != str(context.conversation_id)
        ):
            return ToolResult.failure(
                call_id=context.call_id,
                tool_name=context.tool_name,
                error=ToolError(
                    code=ToolErrorCode.INVALID_ARGUMENTS,
                    message="未找到该消息，或消息不属于当前会话",
                    retryable=False,
                    details={"message_id": str(parsed.message_id)},
                ),
            )
        message = decode_row(row.model_dump())
        if isinstance(message, RawMessage) or not isinstance(message, ToolMessage):
            return ToolResult.failure(
                call_id=context.call_id,
                tool_name=context.tool_name,
                error=ToolError(
                    code=ToolErrorCode.INVALID_ARGUMENTS,
                    message="该消息不是工具结果，无法回放",
                    retryable=False,
                    details={"message_id": str(parsed.message_id)},
                ),
            )

        body = (message.content or "").strip()
        if not body and message.json_data is not None:
            body = json_dumps(message.json_data)
        header = (
            f"[tool:{message.name} status:{message.status.value} "
            f"message_id:{message.id} call_id:{message.tool_call_id}]"
        )
        content = f"{header}\n{body or '（空结果）'}"
        return ToolResult.success(
            call_id=context.call_id,
            tool_name=context.tool_name,
            content=content,
            data={
                "message_id": str(message.id),
                "name": message.name,
                "status": message.status.value,
                "tool_call_id": message.tool_call_id,
                "content_chars": len(body),
            },
        )


class ToolHistoryFactory(SimpleToolFactory):
    """``get_tool_result`` 工厂；只依赖会话消息仓储。"""

    argument_model = GetToolResultArgs

    def create_executor(
        self,
        declaration: ToolDeclaration,
        ctx: ToolBuildContext,
    ):
        """构造历史工具结果回放执行器。"""
        return ToolHistoryExecutor()
