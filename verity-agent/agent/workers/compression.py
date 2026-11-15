"""会话摘要压缩服务。

在回合结束后按 token 阈值把老消息合并进 ``conversation_summaries``；
使用 CAS（covered_through_seq）避免并发压缩覆盖。LLM 只输出 SessionSummary
JSON，代码负责校验和写入。
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any
from uuid import UUID

from agent.hooks.enum import HookEvent, HookSource
from agent.hooks.scope import HookScope
from agent.workers.text import extract_content_text, parse_json_object
from config.memory import MemoryConfig
from prompt import PromptId, PromptRegistry, get_default_prompt_registry
from provider.enums import ModelRole
from provider.llm.helpers import response_text as _response_text
from provider.llm.types import (
    ChatModelRequest,
    ModelGenerationOptions,
    ModelMessage,
    ModelTextPart,
)
from store.pg.models.conversation import SessionSummary
from store.pg.repositories import (
    ConversationMessageRepository,
    ConversationSummaryRepository,
)
from utils.tokens import estimate_tokens, truncate_to_tokens

__all__ = ["CompressionService"]

logger = logging.getLogger(__name__)


def _sanitize_source_ids(
    payload: dict[str, Any],
    allowed_ids: set[UUID],
) -> dict[str, Any]:
    """把模型输出的 source_msg_ids 收敛到输入消息/旧摘要范围内。

    模型可能返回不存在的 UUID 或非 UUID 字符串；直接交给 Pydantic 会让整次
    压缩失败，因此逐项过滤并去重，保留顺序。
    """
    # 新摘要统一写成 v2；模型复制旧摘要的 version=1 时不跟随。
    payload["version"] = 2
    for key in ("topics", "facts", "constraints"):
        items = payload.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            raw = item.get("source_msg_ids")
            if not isinstance(raw, list):
                item["source_msg_ids"] = []
                continue
            cleaned: list[str] = []
            for value in raw:
                try:
                    message_id = UUID(str(value))
                except (TypeError, ValueError):
                    continue
                if message_id not in allowed_ids:
                    continue
                text = str(message_id)
                if text not in cleaned:
                    cleaned.append(text)
            item["source_msg_ids"] = cleaned
    return payload


class CompressionService:
    """回合后异步压缩会话摘要。"""

    def __init__(
        self,
        config: MemoryConfig,
        *,
        model: Any,
        model_id: str,
        messages: ConversationMessageRepository | None = None,
        summaries: ConversationSummaryRepository | None = None,
        hook_dispatcher: Any | None = None,
        prompts: PromptRegistry | None = None,
    ) -> None:
        """保存模型、仓储与 Prompt 注册表；不建立连接。"""
        self.config = config
        self.model = model
        self.model_id = model_id
        self.messages = messages or ConversationMessageRepository()
        self.summaries = summaries or ConversationSummaryRepository()
        self.hook_dispatcher = hook_dispatcher
        self.prompts = prompts or get_default_prompt_registry()
        self.system_prompt = self.prompts.render_text(
            PromptId.MEMORY_COMPRESSION_SYSTEM
        )

    async def maybe_compress(
        self,
        conversation_id: UUID,
        *,
        run_id: str | None = None,
    ) -> bool:
        """按阈值判断并执行压缩；返回是否真正写入摘要。"""
        if not self.config.compression_enabled:
            return False
        current = await self.summaries.get(conversation_id)
        covered = current.covered_through_seq if current is not None else 0
        rows = await self.messages.list_range(
            conversation_id,
            after_seq=covered,
        )
        keep = self.config.compression_keep_recent_messages
        # 只在“用户消息”边界切分，避免把 assistant/tool 链从中间截断。
        user_indexes = [
            index
            for index, row in enumerate(rows)
            if getattr(getattr(row, "role", None), "value", None) == "user"
        ]
        if len(user_indexes) <= keep:
            return False
        compressible = rows[: user_indexes[-keep]]
        if not compressible:
            return False
        tokens = sum(
            estimate_tokens(extract_content_text(row.content))
            for row in compressible
        )
        if tokens < self.config.compression_trigger_tokens:
            return False

        scope = self._hook_scope(conversation_id, run_id)
        await self._emit(scope, HookEvent.SUMMARY_COMPRESS_START, {
            "conversation_id": str(conversation_id),
            "message_count": len(compressible),
            "token_estimate": tokens,
        })
        try:
            old = current.summary_json if current is not None else SessionSummary()
            allowed_ids = self._allowed_source_ids(old, compressible)
            payload = {
                "current_summary": old.model_dump(mode="json", by_alias=True),
                "new_messages": self._messages_text(compressible),
            }
            summary = await self._summarize(payload, allowed_ids=allowed_ids)
            await self.summaries.upsert(
                conversation_id,
                summary,
                compressible[-1].seq_no,
                expected_covered_through_seq=covered,
            )
            await self._emit(scope, HookEvent.SUMMARY_COMPRESS_END, {
                "conversation_id": str(conversation_id),
                "covered_through_seq": compressible[-1].seq_no,
                "status": "ok",
            })
            return True
        except Exception as exc:  # noqa: BLE001 - 压缩失败保留旧摘要
            logger.warning("会话摘要压缩失败: %s", conversation_id, exc_info=True)
            await self._emit(scope, HookEvent.SUMMARY_COMPRESS_WARNING, {
                "conversation_id": str(conversation_id),
                "status": "failed",
                "cause": str(exc)[:300],
            })
            return False

    async def _summarize(
        self,
        payload: dict[str, Any],
        *,
        allowed_ids: set[UUID],
    ) -> SessionSummary:
        """调用 LLM、过滤非法来源 id 并校验 SessionSummary。"""
        request = ChatModelRequest(
            model=self.model_id,
            messages=[
                ModelMessage(
                    role=ModelRole.SYSTEM,
                    content=[ModelTextPart(text=self.system_prompt)],
                ),
                ModelMessage(
                    role=ModelRole.USER,
                    content=[
                        ModelTextPart(
                            text=json.dumps(payload, ensure_ascii=False)
                        )
                    ],
                ),
            ],
            options=ModelGenerationOptions(
                temperature=0.1,
                max_output_tokens=self.config.compression_max_output_tokens,
            ),
        )
        response = await asyncio.wait_for(
            self.model.complete(request),
            timeout=self.config.compression_timeout_s,
        )
        return SessionSummary.model_validate(
            _sanitize_source_ids(
                parse_json_object(_response_text(response)),
                allowed_ids,
            )
        )

    @staticmethod
    def _allowed_source_ids(
        old: SessionSummary,
        rows: list[Any],
    ) -> set[UUID]:
        """允许的来源 id：本次待压缩消息 + 旧摘要已引用的消息。"""
        allowed = {
            row.id for row in rows if getattr(row, "id", None) is not None
        }
        for item in (*old.topics, *old.facts, *old.constraints):
            allowed.update(item.source_msg_ids)
        return allowed

    def _messages_text(self, rows: list[Any]) -> str:
        """渲染消息并带上 id/seq，供模型填写 source_msg_ids。"""
        lines: list[str] = []
        for row in rows:
            text = extract_content_text(row.content)
            if text:
                lines.append(
                    f"[msg:{row.id} seq:{row.seq_no}] {row.role.value}: {text}"
                )
        value = "\n".join(lines)
        if estimate_tokens(value) > self.config.compression_max_input_tokens:
            value = truncate_to_tokens(
                value,
                self.config.compression_max_input_tokens,
            )
        return value

    def _hook_scope(self, conversation_id: UUID, run_id: str | None) -> HookScope | None:
        """构造 HookScope；未注入 dispatcher 时返回 None。"""
        if self.hook_dispatcher is None:
            return None
        return HookScope(
            self.hook_dispatcher,
            conversation_id=conversation_id,
            run_id=run_id,
            source=HookSource.MEMORY,
        )

    async def _emit(
        self,
        scope: HookScope | None,
        event: HookEvent,
        payload: dict[str, Any],
    ) -> None:
        """发送压缩 Hook；失败不影响主流程。"""
        if scope is None:
            return
        try:
            await scope.emit(event, payload)
        except Exception:  # noqa: BLE001 - 观测失败不阻断
            logger.warning("压缩 Hook 发送失败: %s", event.value, exc_info=True)
