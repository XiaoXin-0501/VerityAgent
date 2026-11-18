"""内置 ``recall_round`` 工具：按轮次 id 召回原始提问与最终回答。

一轮对话指“用户一次提问 → Agent 完整回答”之间的全部消息；中间的工具调用
与工具结果只用于过程展示，本工具只返回用户提问和该轮最后一条有正文的
assistant 消息，避免把冗长的中间消息重新塞回上下文。

工具暴露由 Agent 按会话摘要压缩次数控制（见 ``AgentLoop._tool_permissions``），
本模块只负责读取与格式化。
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from agent.tools.enums import ToolKind, ToolSideEffect
from agent.tools.models import ToolContext, ToolError, ToolResult, ToolSpec
from agent.tools.policies import CachePolicy, ToolPolicy
from agent.tools.registry import ToolRegistry
from config.tools import RecallRoundToolConfig
from exception.tool import ToolErrorCode
from store.pg.repositories import ConversationMessageRepository
from utils.text import extract_content_text, truncate

__all__ = [
    "RecallRoundArgs",
    "RecallRoundExecutor",
    "register_recall_round",
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
        shared = getattr(context.services, "conversation_messages", None)
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


def register_recall_round(
    registry: ToolRegistry,
    *,
    config: RecallRoundToolConfig | None = None,
) -> None:
    """注册 ``recall_round``；是否对某次运行暴露由运行时权限决定。"""
    policy = config or RecallRoundToolConfig()
    if not policy.enabled:
        logger.info("轮次召回工具已禁用，跳过 recall_round 注册")
        return

    executor = RecallRoundExecutor(policy)
    registry.register_tool(
        ToolSpec(
            name="recall_round",
            description=(
                "按轮次 id 回看历史问答：返回指定轮次的用户提问和最终回答，"
                "省略中间工具调用。只在会话摘要中的主题标注了 rounds 时使用，"
                "round_ids 必须来自摘要或历史工具结果，不要编造。"
            ),
            group="memory",
            kind=ToolKind.EXECUTABLE,
            side_effect=ToolSideEffect.READ,
            # 由 Agent 在摘要压缩次数达到阈值后动态授予该权限码
            required_permission=policy.permission_code,
            policy=ToolPolicy(
                # 结果只对当前会话有效；缓存 key 不含 conversation_id，
                # 直接关闭缓存，避免任何跨会话命中风险。
                cache=CachePolicy(enabled=False),
            ),
            tags=["memory", "history"],
            metadata={"cache_permission_scoped": True},
        ),
        executor,
        argument_model=RecallRoundArgs,
        source="builtin",
    )
