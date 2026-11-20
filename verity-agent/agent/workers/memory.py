"""长期记忆画像抽取 Worker。

消费 ``user_messages`` 队列：同一用户的一批消息 + 当前 7 段画像 + 相关既有事实
候选交给小模型，模型输出**增量**画像补丁、新原子事实以及事实冲突结论；
代码按段覆盖写回 ``user_memory``（乐观锁），事实按冲突结论逐条 upsert 到
Weaviate，成功后删除暂存行。
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
            candidates = await self._load_fact_candidates(user_id, content_text)
            data = await self._extract(current.value, content_text, candidates)
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
                        candidates=candidates,
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
                        await limit(
                            user_id=user_id,
                            max_facts=self.config.fact_max_active,
                        )
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

    async def _load_fact_candidates(
        self,
        user_id: UUID,
        content_text: str,
    ) -> list[dict[str, Any]]:
        """用本批消息 BM25 召回既有事实，供抽取 Prompt 一并判断冲突。

        只回喂有限候选而不是全部 150 条，避免画像/消息输入被事实列表挤爆。
        召回失败不影响抽取，模型按无既有事实继续。
        """
        limit = int(self.config.extraction_fact_candidates)
        if not self.facts_store or not self.config.fact_write_enabled or limit <= 0:
            return []
        query = content_text.strip()
        if not query:
            return []
        try:
            # 取最近一段消息作为 BM25 query；用户最新表达通常比更早寒暄更能
            # 命中需要判断的偏好/属性事实。
            raw = await self.facts_store.find_similar(
                user_id=user_id,
                content=query[-4000:],
                limit=limit,
            )
            # 只回喂冲突判断需要的字段，避免 score/status 等噪声挤占抽取输入。
            fields = (
                "id",
                "content",
                "category",
                "confidence",
                "confirmation_count",
                "last_confirmed_at",
                "context",
                "scope",
            )
            return [
                {key: item.get(key) for key in fields}
                for item in raw
                if item.get("id") and item.get("content")
            ]
        except Exception:  # noqa: BLE001 - 候选召回失败不阻断抽取
            logger.warning("事实候选召回失败，按无冲突候选继续", exc_info=True)
            return []

    async def _apply_facts(
        self,
        *,
        user_id: UUID,
        facts: list[dict[str, Any]],
        candidates: list[dict[str, Any]],
        allowed_msg_ids: set[str],
        default_source_msg_ids: list[str],
        message_times: Mapping[str, str],
    ) -> int:
        """按抽取模型直接给出的冲突结论增量写入事实。

        事实冲突不再单独调用规则/第二个 LLM：
        - ``none``：新增事实；
        - ``override``：写入新事实并 supersede 旧事实；
        - ``merge``：把旧事实替换为模型给的 ``target_content``（确认次数重置 1）；
        - ``partial_conflict``：旧事实替换为 ``target_content``，同时新增本次
          被否定/补充的 ``content``。
        所有 target id 必须来自抽取时回喂的候选集，模型编造的 id 按新增处理。
        """
        assert self.facts_store is not None
        written = 0
        for fact in facts:
            content = str(fact.get("content") or "").strip()
            target_content = str(
                fact.get("target_content") or fact.get("merged_content") or ""
            ).strip()
            conflict_type = str(
                fact.get("conflict_type") or "none"
            ).strip().lower()
            # 兼容滚动发布期间旧版抽取输出；新版只输出四种类型。
            conflict_type = {
                "direct_override": "override",
                "supersede": "override",
                "duplicate": "none",
                "compatible": "none",
                "unresolved": "none",
            }.get(conflict_type, conflict_type)
            if conflict_type not in {"none", "override", "merge", "partial_conflict"}:
                conflict_type = "none"
            if conflict_type in {"merge", "partial_conflict"}:
                if not target_content and not content:
                    continue
            elif not content:
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
                confirmation_count: int | None = None,
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
                    confirmation_count=confirmation_count,
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

            target_key = str(
                fact.get("target_fact_id") or fact.get("target_id") or ""
            ).strip().lower()
            target_fact = (
                next(
                    (
                        item
                        for item in candidates
                        if str(item.get("id") or "").strip().lower() == target_key
                    ),
                    None,
                )
                if target_key
                else None
            )
            target_id = str((target_fact or {}).get("id") or "")
            if conflict_type in {"merge", "partial_conflict"}:
                if target_fact is None or not target_content:
                    # 模型给了不存在的 target 或漏填改写正文：保守按新增处理，
                    # 不能误覆盖用户历史事实。
                    await write(content or target_content)
                    continue
                await write(target_content, confirmation_count=1)
                await supersede(target_id, target_content)
                if (
                    conflict_type == "partial_conflict"
                    and content
                    and content != target_content
                ):
                    await write(content)
                continue
            if conflict_type == "override" and target_fact is not None:
                await write(content)
                await supersede(target_id, content)
                continue
            # none / target 不存在 / 未知类型：按新增事实处理。
            await write(content)
        return written

    async def _extract(
        self,
        current: MemoryValue,
        content_text: str,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """调用小模型生成画像更新、事实与事实冲突结论（只返回解析后的 JSON）。"""
        payload = {
            "current_profile": current.model_dump(mode="json", by_alias=True),
            "existing_facts": candidates,
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
