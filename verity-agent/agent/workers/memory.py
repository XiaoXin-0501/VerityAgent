"""长期记忆画像抽取 Worker。

消费 ``user_messages`` 队列：同一用户的一批消息 + 当前 7 段画像交给小模型，
模型输出画像更新 JSON，代码用乐观锁写回 ``user_memory``，成功后删除暂存行。
开启事实写入时，同一批抽取结果还会经过冲突判定写入 Weaviate UserFact。
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Mapping
from uuid import UUID

from agent.hooks.enum import HookEvent, HookSource
from agent.hooks.scope import HookScope
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
from store.pg.models.memory import (
    MemoryChange,
    MemoryHistorySections,
    MemorySection,
    MemoryUserSections,
    MemoryValue,
    UserMessage,
)
from store.pg.repositories import UserMemoryRepository, UserMessageRepository
from .text import messages_text, parse_json_object
from utils.time import utcnow

__all__ = ["MemoryProfileWorker"]

logger = logging.getLogger(__name__)


class MemoryProfileWorker:
    """单进程长期记忆画像抽取器。"""

    def __init__(
        self,
        config: MemoryConfig,
        *,
        model: Any,
        model_id: str,
        message_repository: UserMessageRepository | None = None,
        memory_repository: UserMemoryRepository | None = None,
        hook_dispatcher: Any | None = None,
        facts_store: Any | None = None,
        conflict_service: Any | None = None,
        prompts: PromptRegistry | None = None,
    ) -> None:
        """保存模型、仓库、Prompt 注册表与可选 Hook dispatcher；不建立连接。"""
        self.config = config
        self.model = model
        self.model_id = model_id
        self.messages = message_repository or UserMessageRepository()
        self.memories = memory_repository or UserMemoryRepository()
        self.hook_dispatcher = hook_dispatcher
        self.facts_store = facts_store
        self.conflict_service = conflict_service
        self.prompts = prompts or get_default_prompt_registry()
        self.system_prompt = self.prompts.render_text(
            PromptId.MEMORY_EXTRACTION_SYSTEM
        )
        self._stop = asyncio.Event()

    def stop(self) -> None:
        """请求 worker 在当前批次结束后退出。"""
        self._stop.set()

    async def run_forever(self) -> None:
        """轮询队列；空闲时等待 stop/scan interval。"""
        while not self._stop.is_set():
            try:
                worked = await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - worker 循环不能因单批异常退出
                logger.exception("记忆抽取 worker 批次异常")
                worked = False
            if worked:
                continue
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=self.config.worker_scan_interval_s,
                )
            except asyncio.TimeoutError:
                continue

    async def run_once(self) -> bool:
        """领取并处理一批；返回是否处理了消息。"""
        batch = await self.messages.claim_batch(
            self.config.worker_id,
            limit=self.config.worker_batch_size,
            lease_seconds=self.config.worker_lease_seconds,
        )
        if not batch:
            return False
        grouped: dict[UUID, list[UserMessage]] = {}
        for item in batch:
            grouped.setdefault(item.user_id, []).append(item)
        for user_id, items in grouped.items():
            await self._process_user(user_id, items)
        return True

    async def _process_user(
        self,
        user_id: UUID,
        items: list[UserMessage],
    ) -> None:
        """处理同一用户的一批消息；成功删除，失败按退避重试。"""
        ids = [item.id for item in items]
        scope = self._hook_scope(user_id)
        await self._emit(scope, HookEvent.LONG_TERM_EXTRACT_START, {
            "user_id": str(user_id),
            "message_count": len(items),
        })
        try:
            current = await self.memories.get_or_create(user_id)
            content_text = messages_text(items, self.config.extraction_max_input_tokens)
            data = await self._extract(current.value, content_text)
            fact_count: int | None = None
            if self.facts_store is not None and self.config.fact_write_enabled:
                facts = [
                    item for item in (data.get("facts") or [])
                    if isinstance(item, dict)
                ]
                if facts:
                    await self._apply_facts(
                        user_id=user_id,
                        facts=facts,
                        source_msg_ids=[str(item.message_id) for item in items],
                    )
                    limit = getattr(self.facts_store, "enforce_fact_limit", None)
                    if callable(limit):
                        await limit(user_id=user_id)
                fact_count = await self.facts_store.count_active(user_id=user_id)
            await self.memories.update_with_retry(
                user_id,
                lambda latest: MemoryChange(
                    value=_apply_profile(
                        latest.value,
                        data,
                        step=self.config.profile_confidence_step,
                    ),
                    fact_count=fact_count,
                ),
            )
            await self.messages.mark_processing_done(ids)
            await self._emit(scope, HookEvent.LONG_TERM_EXTRACT_END, {
                "user_id": str(user_id),
                "message_count": len(items),
                "status": "ok",
            })
        except Exception as exc:  # noqa: BLE001 - 单用户失败只回退队列
            logger.warning("长期画像抽取失败: user=%s", user_id, exc_info=True)
            await self.messages.mark_failed(
                ids,
                error=str(exc)[:500],
                retry_backoff_s=self.config.worker_retry_backoff_s,
                max_attempts=self.config.worker_max_attempts,
            )
            await self._emit(scope, HookEvent.LONG_TERM_EXTRACT_END, {
                "user_id": str(user_id),
                "message_count": len(items),
                "status": "failed",
            })

    async def _apply_facts(
        self,
        *,
        user_id: UUID,
        facts: list[dict[str, Any]],
        source_msg_ids: list[str],
    ) -> None:
        """写入事实；启用冲突判定时按 duplicate/supersede 决策处理。"""
        assert self.facts_store is not None
        if self.conflict_service is None:
            await self.facts_store.upsert_facts(
                user_id=user_id,
                facts=facts,
                source_msg_ids=source_msg_ids,
            )
            return
        for fact in facts:
            content = str(fact.get("content") or "").strip()
            if not content:
                continue
            candidates = await self.facts_store.find_similar(
                user_id=user_id,
                content=content,
                limit=self.conflict_service.max_candidates,
            )
            decision = await self.conflict_service.decide(
                new_fact=fact,
                candidates=candidates,
            )
            if decision.decision == "duplicate" and decision.target_id:
                target = next(
                    (
                        item for item in candidates
                        if str(item.get("id")) == decision.target_id
                    ),
                    None,
                )
                if target is not None:
                    fact = {**fact, "content": target.get("content") or content}
            await self.facts_store.upsert_facts(
                user_id=user_id,
                facts=[fact],
                source_msg_ids=source_msg_ids,
            )
            if decision.decision == "supersede" and decision.target_id:
                new_id = self.facts_store.fact_id_for(
                    user_id,
                    str(fact.get("content") or content),
                )
                await self.facts_store.supersede(
                    user_id=user_id,
                    old_fact_id=decision.target_id,
                    new_fact_id=new_id,
                )

    async def _extract(self, current: MemoryValue, content_text: str) -> dict[str, Any]:
        """调用小模型生成画像更新字段（只返回解析后的 JSON）。"""
        payload = {
            "current_profile": current.model_dump(mode="json", by_alias=True),
            "new_messages": content_text,
        }
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
                max_output_tokens=self.config.extraction_max_output_tokens,
            ),
        )
        response = await asyncio.wait_for(
            self.model.complete(request),
            timeout=max(30.0, float(self.config.worker_lease_seconds)),
        )
        data = parse_json_object(_response_text(response))
        return data

    def _hook_scope(self, user_id: UUID) -> HookScope | None:
        """构造可选 HookScope；未注入 dispatcher 时返回 None。"""
        if self.hook_dispatcher is None:
            return None
        return HookScope(
            self.hook_dispatcher,
            user_id=user_id,
            source=HookSource.MEMORY,
        )

    async def _emit(
        self,
        scope: HookScope | None,
        event: HookEvent,
        payload: dict[str, Any],
    ) -> None:
        """发送记忆 worker Hook；失败不影响抽取。"""
        if scope is None:
            return
        try:
            await scope.emit(event, payload)
        except Exception:  # noqa: BLE001 - 观测失败不阻断 worker
            logger.warning("记忆 worker Hook 发送失败: %s", event.value, exc_info=True)


def _apply_profile(
    current: MemoryValue,
    data: Mapping[str, Any],
    *,
    step: float = 0.05,
) -> MemoryValue:
    """只接受已知 7 段字段；未知字段忽略，缺失字段保留旧值。"""
    now = utcnow()

    def section(name: str, old: MemorySection) -> MemorySection:
        """合并单个画像段；空值保留旧段。"""
        value = data.get(name)
        if not isinstance(value, str) or not value.strip():
            return old
        return MemorySection(
            summary=value.strip(),
            updated_at=now,
            confidence=min(0.95, max(0.5, old.confidence + step)),
        )

    user = MemoryUserSections(
        work_context=section("work_context", current.user.work_context),
        personal_context=section("personal_context", current.user.personal_context),
        top_of_mind=section("top_of_mind", current.user.top_of_mind),
        cognitive_style=section("cognitive_style", current.user.cognitive_style),
    )
    history = MemoryHistorySections(
        recent_months=section("recent_months", current.history.recent_months),
        earlier_context=section("earlier_context", current.history.earlier_context),
        long_term_background=section(
            "long_term_background", current.history.long_term_background
        ),
    )
    return MemoryValue(last_updated=now, user=user, history=history)
