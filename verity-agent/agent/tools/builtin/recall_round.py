"""内置 ``recall_round`` 工具：按轮次 id 召回原始提问与最终回答。

一轮对话指“用户一次提问 → Agent 完整回答”之间的全部消息；中间的工具调用
与工具结果只用于过程展示，本工具只返回用户提问和该轮最后一条有正文的
assistant 消息，避免把冗长的中间消息重新塞回上下文。

工具暴露由 manifest 的 ``after_compressions`` 暴露条件控制（会话摘要
压缩次数达到阈值后才进入本次 ToolSelection）；本模块只负责读取与格式化。
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from agent.tools.declaration import ToolBuildContext, ToolDeclaration
from agent.tools.factory import SimpleToolFactory, with_exposure_param
from agent.tools.models import ToolContext, ToolError, ToolResult
from agent.tools.services import ToolServiceKey, get_tool_service
from config.tools import RecallRoundToolConfig
from exception.tool import ToolErrorCode
from store.pg.repositories import ConversationMessageRepository
from utils.text import extract_content_text, truncate

__all__ = [
    "RecallRoundArgs",
    "RecallRoundExecutor",
    "RecallRoundFactory",
]

logger = logging.getLogger(__name__)

# 单条提问/回答片段的最大字符数；避免一条超长回答挤占整个工具结果。
_ITEM_MAX_CHARS = 2000


class RecallRoundArgs(BaseModel):
    """``recall_round`` 的 LLM 参数模型。"""

    # 轮次 id 来自会话摘要 topic 的 ``[rounds: ...]`` 标注或历史工具结果
    round_ids: list[UUID] = Field(min_length=1, max_length=20)


class RecallRoundExecutor:
    """把 PG 消息链裁剪成“提问 + 最终回答”的执行器。"""

    def __init__(
        self,
        config: RecallRoundToolConfig | None = None,
        *,
        repository: ConversationMessageRepository | None = None,
    ) -> None:
        """保存配置与可选仓库；未注入时按默认连接池构造。"""
        self.config = config or RecallRoundToolConfig()
        self.repository = repository

    def _repository(self, context: ToolContext) -> ConversationMessageRepository:
        """优先使用应用共享的仓库；测试可注入内存实现。"""
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
        """校验轮次 id 并返回用户提问与最终回答。"""
        parsed = RecallRoundArgs.model_validate(arguments)
        # 保持调用顺序并去重；超出上限时只取前 N 个，避免一次拉回过多历史。
        round_ids = list(dict.fromkeys(parsed.round_ids))[: self.config.max_rounds]
        try:
            rows = await self._repository(context).list_rounds(
                context.conversation_id,
                round_ids,
            )
        except Exception as exc:  # noqa: BLE001 - 工具边界统一转结构化错误
            logger.warning("轮次召回读取失败: %s", exc, exc_info=True)
            return ToolResult.failure(
                call_id=context.call_id,
                tool_name=context.tool_name,
                error=ToolError(
                    code=ToolErrorCode.UPSTREAM_ERROR,
                    message="轮次召回读取失败，请稍后重试",
                    retryable=True,
                    cause=str(exc),
                ),
            )

        grouped: dict[UUID, list[Any]] = {}
        for row in rows:
            round_id = getattr(row, "round_id", None)
            if round_id is not None:
                grouped.setdefault(round_id, []).append(row)

        blocks: list[str] = []
        data: list[dict[str, Any]] = []
        missing: list[str] = []
        for round_id in round_ids:
            items = grouped.get(round_id)
            if not items:
                missing.append(str(round_id))
                continue
            question = self._question(items)
            answer = self._answer(items)
            if not question and not answer:
                missing.append(str(round_id))
                continue
            blocks.append(
                f"[round:{round_id}]\n"
                f"用户提问: {question or '（无文本提问）'}\n"
                f"最终回答: {answer or '（该轮没有完成文本回答）'}"
            )
            data.append(
                {
                    "round_id": str(round_id),
                    "question": question,
                    "answer": answer,
                }
            )

        if not blocks:
            return ToolResult.failure(
                call_id=context.call_id,
                tool_name=context.tool_name,
                error=ToolError(
                    code=ToolErrorCode.INVALID_ARGUMENTS,
                    message="未找到这些轮次，或轮次不属于当前会话",
                    details={"round_ids": [str(item) for item in round_ids]},
                ),
            )

        content = "\n\n".join(blocks)
        truncated = False
        if len(content) > self.config.max_chars:
            content = truncate(content, self.config.max_chars)
            truncated = True
        if missing:
            content += f"\n\n未找到的轮次: {', '.join(missing)}"
        return ToolResult.success(
            call_id=context.call_id,
            tool_name=context.tool_name,
            content=content,
            data={"rounds": data, "missing": missing},
            truncated=truncated,
        )

    @staticmethod
    def _question(items: list[Any]) -> str:
        """取本轮第一条用户消息的正文。"""
        for row in items:
            if getattr(getattr(row, "role", None), "value", None) == "user":
                text = extract_content_text(row.content).strip()
                if text:
                    return truncate(text, _ITEM_MAX_CHARS)
        return ""

    @staticmethod
    def _answer(items: list[Any]) -> str:
        """取本轮最后一条有正文的 assistant 消息；工具调用轮不占答案位。"""
        answer = ""
        for row in items:
            if getattr(getattr(row, "role", None), "value", None) != "assistant":
                continue
            text = extract_content_text(row.content).strip()
            if text:
                answer = text
        return truncate(answer, _ITEM_MAX_CHARS)


class RecallRoundFactory(SimpleToolFactory):
    """``recall_round`` 工厂：可用性、执行器与阈值覆盖都封装在此。"""

    argument_model = RecallRoundArgs

    def available(
        self,
        declaration: ToolDeclaration,
        ctx: ToolBuildContext,
    ) -> bool:
        """配置关闭时不注册。"""
        return ctx.tools_config.recall_round.enabled

    def create_executor(
        self,
        declaration: ToolDeclaration,
        ctx: ToolBuildContext,
    ):
        """构造轮次召回执行器。"""
        return RecallRoundExecutor(ctx.tools_config.recall_round)

    def spec_overrides(
        self,
        declaration: ToolDeclaration,
        ctx: ToolBuildContext,
    ):
        """运行配置覆盖 JSON 的 min_compressions 默认值。"""
        return {
            "exposure": with_exposure_param(
                declaration.exposure,
                "min_compressions",
                ctx.tools_config.recall_round.min_compressions,
            )
        }
