"""长期记忆画像抽取 Worker。

消费 ``user_messages`` 队列：同一用户的一批消息 + 当前 7 段画像交给小模型，
模型输出**增量**画像补丁（只含变化段）和新原子事实；代码按段覆盖写回
``user_memory``（乐观锁），事实逐条 upsert 到 Weaviate，成功后删除暂存行。
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
from utils.time import isoformat_z, utcnow

__all__ = ["MemoryProfileWorker"]

logger = logging.getLogger(__name__)

# 画像段在 Prompt/JSON 中可能使用 snake_case 或 current_profile 的 camelCase；
# 两种都接受，避免模型模仿输入结构时补丁被静默丢弃。
_SECTION_ALIASES = {
    "work_context": "workContext",
    "personal_context": "personalContext",
    "top_of_mind": "topOfMind",
    "cognitive_style": "cognitiveStyle",
    "recent_months": "recentMonths",
    "earlier_context": "earlierContext",
    "long_term_background": "longTermBackground",
}


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
            facts_written = 0
            if self.facts_store is not None and self.config.fact_write_enabled:
                facts = [
                    item for item in (data.get("facts") or [])
                    if isinstance(item, dict)
                ]
                if facts:
                    facts_written = await self._apply_facts(
                        user_id=user_id,
                        facts=facts,
                        allowed_msg_ids={
                            str(item.message_id) for item in items
                        },
                        default_source_msg_ids=[
                            str(item.message_id) for item in items
                        ],
                        message_times={
                            str(item.message_id): isoformat_z(item.created_at)
                            for item in items
                        },
                    )
                    limit = getattr(self.facts_store, "enforce_fact_limit", None)
                    if callable(limit):
                        # facts_store 是实现协议；可选上限方法由调用方按存在性探测。
                        await limit(user_id=user_id)  # pyright: ignore[reportGeneralTypeIssues]
                fact_count = await self.facts_store.count_active(user_id=user_id)
            await self.memories.update_with_retry(
                user_id,
                lambda latest: MemoryChange(
                    value=_apply_profile(
                        latest.value,
                        data,
                        step=self.config.profile_confidence_step,
                        source_msg_ids=[item.message_id for item in items],
                        source_limit=self.config.profile_source_msg_limit,
                    ),
                    fact_count=fact_count,
                    # 事实集发生写入时递增版本，供事实缓存失效；幂等重试只在
                    # 成功 apply 后 +1，不会因为队列重试无限增长。
                    facts_revision=(
                        latest.facts_revision + 1 if facts_written else None
                    ),
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
        allowed_msg_ids: set[str],
        default_source_msg_ids: list[str],
        message_times: Mapping[str, str],
    ) -> int:
        """写入事实；冲突判定结果决定覆盖、合并或部分改写。

        逐条写入让 ``source_msg_ids`` 精确到单条用户消息；直接覆盖/合并/
        部分冲突通过“写入新内容 + 旧事实 supersede”落库，保留审计痕迹，
        但活跃事实只有用户最新的表达。
        """
        assert self.facts_store is not None
        written = 0
        for fact in facts:
            content = str(fact.get("content") or "").strip()
            if not content:
                continue
            written += 1
            sources = _fact_sources(
                fact,
                allowed_msg_ids=allowed_msg_ids,
                default_source_msg_ids=default_source_msg_ids,
            )
            confirmed_at = _confirmed_at(sources, message_times)

            async def write(
                fact_content: str,
                *,
                status: str = "active",
                confidence: float | None = None,
            ) -> None:
                """写入一条事实；空内容或同内容重复跳过。"""
                text = fact_content.strip()
                if not text:
                    return
                payload = {**fact, "content": text}
                if confidence is not None:
                    payload["confidence"] = confidence
                await self.facts_store.upsert_facts(
                    user_id=user_id,
                    facts=[payload],
                    source_msg_ids=sources,
                    confirmed_at=confirmed_at,
                    status=status,
                )

            async def supersede(target_id: str | None, new_content: str) -> None:
                """把旧事实标记为被新内容取代；同内容不重复标记。"""
                if not target_id:
                    return
                new_id = self.facts_store.fact_id_for(user_id, new_content)
                if new_id != target_id:
                    await self.facts_store.supersede(
                        user_id=user_id,
                        old_fact_id=target_id,
                        new_fact_id=new_id,
                    )

            if self.conflict_service is None:
                await write(content)
                continue

            candidates = await self.facts_store.find_similar(
                user_id=user_id,
                content=content,
                limit=self.conflict_service.max_candidates,
            )
            decision = await self.conflict_service.decide(
                # observed_at 让模型能按“后出现的事实优先”判断，
                # 而不是只看候选的确认时间。
                new_fact={**fact, "observed_at": confirmed_at},
                candidates=candidates,
            )
            target = next(
                (
                    item
                    for item in candidates
                    if str(item.get("id")) == str(decision.target_id or "")
                ),
                None,
            )
            kind = decision.decision
            if kind == "unresolved":
                # 证据不足：新事实与候选都置 disputed，等待下次对话确认。
                await write(content, status="disputed")
                mark_disputed = getattr(self.facts_store, "mark_disputed", None)
                if decision.target_id and callable(mark_disputed):
                    await mark_disputed(
                        user_id=user_id,
                        fact_ids=[decision.target_id],
                    )
                continue
            if kind == "duplicate":
                # 同义确认：沿用候选正文，合并证据与确认计数。
                await write(str((target or {}).get("content") or content))
                continue
            if kind in {"direct_override", "merge", "supersede"}:
                merged = (decision.merged_content or content).strip()
                if target is None:
                    # 模型给了不存在的 target：保守按新增事实处理。
                    await write(merged)
                    continue
                confidence = (
                    max(
                        float(fact.get("confidence") or 0.5),
                        float((target or {}).get("confidence") or 0.0),
                    )
                    if kind == "merge"
                    else float(fact.get("confidence") or 0.5)
                )
                await write(merged, confidence=confidence)
                await supersede(str(target.get("id") or ""), merged)
                continue
            if kind == "partial_conflict":
                old_content = (
                    decision.merged_content
                    or str((target or {}).get("content") or "")
                ).strip()
                if target is not None and old_content:
                    # 改写旧事实：写入修订后的内容并 supersede 旧事实。
                    await write(
                        old_content,
                        confidence=float(target.get("confidence") or 0.5),
                    )
                    await supersede(str(target.get("id") or ""), old_content)
                new_content = (decision.new_content or "").strip()
                if new_content:
                    # 部分否定产生的新事实单独保留。
                    await write(new_content)
                elif content.strip() and content.strip() != old_content:
                    # 模型漏填 new_content 时，至少保留本次新事实本身。
                    await write(content)
                continue
            # compatible / 其他未知结果：作为一条可同时成立的新事实。
            await write(content)
        return written

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


def _fact_sources(
    fact: Mapping[str, Any],
    *,
    allowed_msg_ids: set[str],
    default_source_msg_ids: list[str],
) -> list[str]:
    """解析一条事实的 ``msg_id``；越界/缺失时回退到本批消息。

    回退而不是丢弃事实：旧 prompt 或模型偶尔漏字段时仍可沉淀事实，只是
    来源粒度变成本批消息；正常情况只保留单条最直接证据。
    """
    raw = fact.get("msg_id")
    try:
        message_id = str(UUID(str(raw)))
    except (TypeError, ValueError, AttributeError):
        return list(default_source_msg_ids)
    if message_id not in allowed_msg_ids:
        return list(default_source_msg_ids)
    return [message_id]


def _confirmed_at(
    sources: list[str],
    message_times: Mapping[str, str],
) -> str | None:
    """返回来源消息中最大的时间；缺失时交给存储层使用当前时间。"""
    values = [message_times[item] for item in sources if item in message_times]
    return max(values) if values else None


def _apply_profile(
    current: MemoryValue,
    data: Mapping[str, Any],
    *,
    step: float = 0.05,
    source_msg_ids: list[UUID] | None = None,
    source_limit: int = 20,
) -> MemoryValue | None:
    """把抽取结果中的画像增量补丁合并进当前画像。

    新 Prompt 输出 ``profile_patch``：只包含发生变化的分段；每个分段给出
    更新后的完整摘要（覆盖式），未出现的分段保持旧值。为兼容旧版 Prompt /
    滚动发布期间的输出，也接受顶层直接放 7 段扁平字段的旧格式。

    画像段被本次抽取更新时，把本批用户消息 id 并入该段的
    ``source_msg_ids``；只保留最近 ``source_limit`` 条证据。没有任何段变化
    时返回 ``None``，避免无意义地刷新 ``last_updated``/revision。

    注意：模型把旧值原样复制回来不算变化；只有文本实际改变才更新
    ``updated_at``/confidence，避免每次抽取都膨胀置信度。
    """
    now = utcnow()
    sources = list(source_msg_ids or [])
    changed = False
    # 新格式：profile_patch 只含变化段；旧格式：顶层扁平字段。
    patch = data.get("profile_patch")
    patch_data: Mapping[str, Any] = patch if isinstance(patch, Mapping) else data

    def section(name: str, old: MemorySection) -> MemorySection:
        """合并单个画像段；空值或原样返回保留旧段（含旧证据）。"""
        nonlocal changed
        value = patch_data.get(name)
        if value is None:
            value = patch_data.get(_SECTION_ALIASES.get(name, ""))
        if not isinstance(value, str) or not value.strip():
            return old
        text = value.strip()
        if text == old.summary.strip():
            return old
        changed = True
        merged = list(dict.fromkeys([*old.source_msg_ids, *sources]))
        if source_limit > 0:
            merged = merged[-source_limit:]
        # DBJsonModel 的字段带 JSON 别名；用 model_validate 同时接受 snake_case
        # 字段名并保留 Pydantic 校验，避免直接构造时只认别名。
        return MemorySection.model_validate(
            {
                "summary": text,
                "updated_at": now,
                "confidence": min(0.95, max(0.5, old.confidence + step)),
                "source_msg_ids": merged,
            }
        )

    user = MemoryUserSections.model_validate(
        {
            "work_context": section("work_context", current.user.work_context),
            "personal_context": section(
                "personal_context", current.user.personal_context
            ),
            "top_of_mind": section("top_of_mind", current.user.top_of_mind),
            "cognitive_style": section(
                "cognitive_style", current.user.cognitive_style
            ),
        }
    )
    history = MemoryHistorySections.model_validate(
        {
            "recent_months": section(
                "recent_months", current.history.recent_months
            ),
            "earlier_context": section(
                "earlier_context", current.history.earlier_context
            ),
            "long_term_background": section(
                "long_term_background", current.history.long_term_background
            ),
        }
    )
    if not changed:
        return None
    return MemoryValue.model_validate(
        {"last_updated": now, "user": user, "history": history}
    )
