"""会话摘要压缩服务。

在回合结束后按 token 阈值把老消息合并进 ``conversation_summaries``；
使用 CAS（covered_through_seq）避免并发压缩覆盖。LLM 只输出 SessionSummary
JSON，代码负责校验和写入。
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

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
from store.pg.models.conversation import SessionSummary, SummaryTopic
from store.pg.repositories import (
    ConversationMessageRepository,
    ConversationSummaryRepository,
)
from utils.text import normalize_whitespace
from utils.time import ensure_utc, parse_datetime, utcnow
from utils.tokens import estimate_tokens, truncate_to_tokens

__all__ = ["CompressionService"]

logger = logging.getLogger(__name__)


def _parse_uuid(value: Any) -> UUID | None:
    """把模型输出的 id 解析成 UUID；非法/缺失返回 None。"""
    try:
        return UUID(str(value))
    except (TypeError, ValueError):
        return None


def _parse_datetime(value: Any) -> datetime | None:
    """解析 ISO8601 时间并归一为 UTC；非法/缺失返回 None。"""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return parse_datetime(value)
    except ValueError:
        return None


def _positive_int(value: Any) -> int | None:
    """解析正整数；非法/非正数返回 None。"""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 1 else None


def _normalize_text(value: str) -> str:
    """规范化文本用于事实/主题内容匹配。"""
    return normalize_whitespace(value or "", keep_newlines=False).lower()


def _dedupe_ids(values: list[UUID], limit: int) -> list[UUID]:
    """按出现顺序去重 UUID，并保留最近 ``limit`` 条。"""
    seen: list[UUID] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen[-limit:] if limit > 0 else seen


def _clean_ids(raw: Any, allowed_ids: set[UUID]) -> list[str]:
    """把模型输出的 id 列表收敛到允许集合；非法/越界项直接丢弃。"""
    if not isinstance(raw, list):
        return []
    cleaned: list[str] = []
    for value in raw:
        item_id = _parse_uuid(value)
        if item_id is None or item_id not in allowed_ids:
            continue
        text = str(item_id)
        if text not in cleaned:
            cleaned.append(text)
    return cleaned


def _sanitize_source_ids(
    payload: dict[str, Any],
    *,
    allowed_msg_ids: set[UUID],
    allowed_round_ids: set[UUID],
    existing_topic_ids: set[UUID] | None = None,
) -> dict[str, Any]:
    """收敛模型增量输出的主题/事实来源，并保证 topic id 可被 facts 引用。

    - topic/fact/constraint 都是**增量 upsert**：模型只返回修改/新增项；
    - topic 只保留 ``source_round_ids``；旧 ``source_msg_ids``/``topic_id``
      在进入模型前已由 SessionSummary 兼容层归一；
    - fact 的 ``topic_id`` 可以指向旧摘要已有 topic，也可以指向本次新增 topic；
      其它 id 清空；
    - facts/constraints 的 ``source_msg_ids`` 只允许输入消息或旧摘要证据。
    """
    payload["version"] = 3
    allowed_topic_ids: set[UUID] = set(existing_topic_ids or ())
    seen_topic_ids: set[UUID] = set()

    topics = payload.get("topics")
    if isinstance(topics, list):
        cleaned_topics: list[dict[str, Any]] = []
        for item in topics:
            if not isinstance(item, dict):
                continue
            # 已有主题沿用模型复制的 id；新主题可省略 id，由代码生成。
            topic_id = _parse_uuid(item.get("id") or item.get("topic_id"))
            if topic_id is None or topic_id in seen_topic_ids:
                topic_id = uuid4()
            seen_topic_ids.add(topic_id)
            allowed_topic_ids.add(topic_id)
            item["id"] = str(topic_id)
            item.pop("topic_id", None)
            item.pop("source_msg_ids", None)
            # last_mentioned_at 由代码按是否新增讨论维护，不接受模型时间。
            item.pop("last_mentioned_at", None)
            mention_count = _positive_int(item.get("mention_count"))
            if mention_count is not None:
                item["mention_count"] = mention_count
            else:
                item.pop("mention_count", None)
            item["source_round_ids"] = _clean_ids(
                item.get("source_round_ids"), allowed_round_ids
            )
            cleaned_topics.append(item)
        payload["topics"] = cleaned_topics

    seen_item_ids: set[UUID] = set()
    for key, id_name in (
        ("facts", "fact_id"),
        ("constraints", "constraint_id"),
    ):
        items = payload.get(key)
        if not isinstance(items, list):
            continue
        cleaned_items: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            item_id = _parse_uuid(item.get(id_name))
            if item_id is None or item_id in seen_item_ids:
                item.pop(id_name, None)
            else:
                item[id_name] = str(item_id)
                seen_item_ids.add(item_id)
            if key == "facts":
                topic_id = _parse_uuid(item.get("topic_id"))
                if topic_id is not None and topic_id in allowed_topic_ids:
                    item["topic_id"] = str(topic_id)
                else:
                    item.pop("topic_id", None)
            count = _positive_int(item.get("confirmation_count"))
            if count is not None:
                item["confirmation_count"] = count
            else:
                item.pop("confirmation_count", None)
            confirmed_at = _parse_datetime(item.get("last_confirmed_at"))
            if confirmed_at is not None:
                item["last_confirmed_at"] = confirmed_at.isoformat()
            else:
                item.pop("last_confirmed_at", None)
            item["source_msg_ids"] = _clean_ids(
                item.get("source_msg_ids"), allowed_msg_ids
            )
            cleaned_items.append(item)
        payload[key] = cleaned_items
    return payload


def _split_turns(rows: list[Any]) -> list[list[Any]]:
    """按用户消息边界切完整回合；首条 assistant/tool 归入当前回合。"""
    turns: list[list[Any]] = []
    current: list[Any] = []
    for row in rows:
        role = getattr(getattr(row, "role", None), "value", None)
        if role == "user" and current:
            turns.append(current)
            current = []
        current.append(row)
    if current:
        turns.append(current)
    return turns


def _row_tokens(row: Any) -> int:
    """估算一条待压缩消息的 content token。"""
    return estimate_tokens(extract_content_text(getattr(row, "content", None)))


def _keep_turns(turns: list[list[Any]], config: MemoryConfig) -> int:
    """按窗口回合数与最近回合长度动态决定压缩后保留几轮。

    规则与《记忆模块数据结构设计》§4.3 一致：
    - 窗口 < 3 轮：保留 0 轮；
    - 3~4 轮：最近一轮长度合适保留 1 轮，否则 0；
    - >= 5 轮：最近两轮都合适且合计不超上限保留 2 轮，否则最近一轮合适
      保留 1 轮，否则 0。
    始终至少压掉 1 轮，避免没有可推进的水位线。
    """
    count = len(turns)
    if count <= 1:
        return 0
    latest = sum(_row_tokens(row) for row in turns[-1])
    if count < config.compression_keep_zero_below_turns:
        keep = 0
    elif count < config.compression_keep_one_below_turns:
        keep = 1 if latest <= config.compression_keep_turn_max_tokens else 0
    else:
        previous = sum(_row_tokens(row) for row in turns[-2])
        if (
            latest <= config.compression_keep_turn_max_tokens
            and previous <= config.compression_keep_turn_max_tokens
            and latest + previous <= config.compression_keep_pair_max_tokens
        ):
            keep = 2
        elif latest <= config.compression_keep_turn_max_tokens:
            keep = 1
        else:
            keep = 0
    return max(0, min(keep, config.compression_keep_recent_turns_max, count - 1))


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
        # 按用户提问切完整回合；压缩/保留都不能拆 assistant/tool 链。
        turns = _split_turns(rows)
        keep_turns = _keep_turns(turns, self.config)
        if len(turns) <= keep_turns:
            return False
        compressible_turns = turns[: len(turns) - keep_turns] if keep_turns else turns
        compressible = [row for turn in compressible_turns for row in turn]
        if not compressible:
            return False
        tokens = sum(_row_tokens(row) for row in compressible)
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
            allowed_msg_ids = self._allowed_msg_ids(old, compressible)
            allowed_round_ids = self._allowed_round_ids(old, compressible)
            payload = {
                "current_summary": old.model_dump(mode="json", by_alias=True),
                "new_messages": self._messages_text(compressible),
            }
            summary = await self._summarize(
                payload,
                allowed_msg_ids=allowed_msg_ids,
                allowed_round_ids=allowed_round_ids,
                existing_topic_ids={topic.id for topic in old.topics},
            )
            # 模型只负责内容合并；计数/时间/ID 稳定性与 topic 遗忘由代码维护。
            summary = self._merge_summary(old, summary)
            compression_count_after = (
                current.compression_count if current is not None else 0
            ) + 1
            summary = self._apply_topic_forgetting(
                summary,
                compression_count_after=compression_count_after,
                conversation_id=conversation_id,
            )
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
                "topic_count": len(summary.topics),
                "fact_count": len(summary.facts),
                "compression_count": compression_count_after,
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
        allowed_msg_ids: set[UUID],
        allowed_round_ids: set[UUID],
        existing_topic_ids: set[UUID],
    ) -> SessionSummary:
        """调用 LLM、过滤非法来源 id 并校验增量 SessionSummary。"""
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
                allowed_msg_ids=allowed_msg_ids,
                allowed_round_ids=allowed_round_ids,
                existing_topic_ids=existing_topic_ids,
            )
        )

    def _merge_summary(
        self,
        old: SessionSummary,
        delta: SessionSummary,
    ) -> SessionSummary:
        """把模型返回的增量 topic/fact/constraint 应用到旧摘要。

        模型只返回“需要修改或新增”的条目；代码以旧摘要为基底 upsert，
        并权威维护：
        - topic：mention_count、last_mentioned_at、source_round_ids；
        - fact/constraint：confirmation_count、last_confirmed_at、source_msg_ids。
        未在增量中出现的旧条目原样保留。
        """
        now = utcnow()
        topics, topic_remap = self._apply_topic_deltas(
            old.topics,
            delta.topics,
            now=now,
        )
        topic_ids = {topic.id for topic in topics}
        facts = self._apply_record_deltas(
            old.facts,
            delta.facts,
            id_name="fact_id",
            now=now,
            source_limit=self.config.compression_source_msg_limit,
            topic_ids=topic_ids,
            topic_remap=topic_remap,
        )
        facts = self._limit_summary_facts(facts)
        constraints = self._apply_record_deltas(
            old.constraints,
            delta.constraints,
            id_name="constraint_id",
            now=now,
            source_limit=self.config.compression_source_msg_limit,
            topic_ids=None,
            topic_remap=None,
        )
        return SessionSummary.model_validate(
            {
                "version": 3,
                "topics": topics,
                "facts": facts,
                "constraints": constraints,
            }
        )

    def _apply_topic_deltas(
        self,
        old_topics: list[SummaryTopic],
        deltas: list[SummaryTopic],
        *,
        now: datetime,
    ) -> tuple[list[SummaryTopic], dict[UUID, UUID]]:
        """在旧 topic 列表上应用增量；返回合并结果与 id 重映射。

        - 按 id 或规范化 name 命中旧主题；
        - 命中时保留旧 id，合并轮次，mention_count 按新增轮次数增加；
        - 未命中时视为新主题，生成/沿用稳定 id，mention_count=1；
        - 未出现在增量中的旧主题原样保留。
        """
        result = [topic.model_copy(deep=True) for topic in old_topics]
        by_id = {topic.id: index for index, topic in enumerate(result)}
        by_name = {
            _normalize_text(topic.name): index
            for index, topic in enumerate(result)
        }
        used = set(by_id)
        remap: dict[UUID, UUID] = {}
        round_limit = self.config.compression_topic_round_limit
        for delta in deltas:
            original_id = delta.id
            index = by_id.get(delta.id)
            if index is None:
                index = by_name.get(_normalize_text(delta.name))
            if index is not None:
                previous = result[index]
                new_rounds = [
                    round_id
                    for round_id in delta.source_round_ids
                    if round_id not in previous.source_round_ids
                ]
                updated = previous.model_copy(
                    update={
                        "name": delta.name.strip() or previous.name,
                        "summary": delta.summary.strip() or previous.summary,
                        "source_round_ids": _dedupe_ids(
                            [
                                *previous.source_round_ids,
                                *delta.source_round_ids,
                            ],
                            round_limit,
                        ),
                        # 讨论次数只按本轮新增轮次增加；没有新轮次时保持旧值，
                        # 避免模型回显未变化主题导致计数膨胀。
                        "mention_count": previous.mention_count + len(new_rounds),
                        "last_mentioned_at": (
                            now if new_rounds else previous.last_mentioned_at
                        ),
                    }
                )
                result[index] = updated
                by_name[_normalize_text(updated.name)] = index
                remap[original_id] = previous.id
                continue
            new_id = delta.id if delta.id not in used else uuid4()
            topic = delta.model_copy(
                update={
                    "id": new_id,
                    "name": delta.name.strip() or "未命名主题",
                    "summary": delta.summary.strip(),
                    "source_round_ids": _dedupe_ids(
                        delta.source_round_ids,
                        round_limit,
                    ),
                    "mention_count": 1,
                    "last_mentioned_at": now,
                }
            )
            result.append(topic)
            index = len(result) - 1
            by_id[new_id] = index
            by_name[_normalize_text(topic.name)] = index
            used.add(new_id)
            remap[original_id] = new_id
        return result, remap

    def _apply_record_deltas(
        self,
        old_items: list[Any],
        deltas: list[Any],
        *,
        id_name: str,
        now: datetime,
        source_limit: int,
        topic_ids: set[UUID] | None,
        topic_remap: dict[UUID, UUID] | None,
    ) -> list[Any]:
        """在旧 fact/constraint 列表上应用增量 upsert。

        同 id（或同内容）命中旧条目时复用旧 id：
        - 内容不变：确认次数按新增证据数增加，最近确认时间更新；
        - 内容变化：确认次数重置为 1，旧证据不再支持新内容；
        未命中时按新条目写入，确认次数为 1。
        """
        result = [item.model_copy(deep=True) for item in old_items]
        by_id = {
            getattr(item, id_name): index
            for index, item in enumerate(result)
        }
        by_content = {
            _normalize_text(item.content): index
            for index, item in enumerate(result)
        }
        used = set(by_id)
        for delta in deltas:
            content = str(delta.content or "").strip()
            if not content:
                continue
            delta_id = getattr(delta, id_name)
            index = by_id.get(delta_id)
            if index is None:
                index = by_content.get(_normalize_text(content))
            if index is not None:
                previous = result[index]
                old_content_key = _normalize_text(previous.content)
                same_content = (
                    _normalize_text(previous.content) == _normalize_text(content)
                )
                if same_content:
                    added = len(
                        set(delta.source_msg_ids) - set(previous.source_msg_ids)
                    )
                    confirmation_count = previous.confirmation_count + added
                    sources = _dedupe_ids(
                        [
                            *previous.source_msg_ids,
                            *delta.source_msg_ids,
                        ],
                        source_limit,
                    )
                    confidence = max(previous.confidence, delta.confidence)
                    last_confirmed_at = (
                        now if added else previous.last_confirmed_at
                    )
                else:
                    # 冲突改写/覆盖：旧证据不再支持新正文。
                    confirmation_count = 1
                    sources = _dedupe_ids(delta.source_msg_ids, source_limit)
                    confidence = delta.confidence
                    last_confirmed_at = now
                    by_content.pop(old_content_key, None)
                topic_id = self._resolve_topic_id(
                    getattr(delta, "topic_id", None),
                    topic_remap,
                    topic_ids,
                    fallback=getattr(previous, "topic_id", None),
                )
                update = {
                    id_name: getattr(previous, id_name),
                    "content": content,
                    "source_msg_ids": sources,
                    "confidence": confidence,
                    "confirmation_count": confirmation_count,
                    "last_confirmed_at": last_confirmed_at,
                }
                if topic_ids is not None:
                    update["topic_id"] = topic_id
                updated = previous.model_copy(update=update)
                result[index] = updated
                by_content[_normalize_text(content)] = index
                continue
            new_id = delta_id if delta_id not in used else uuid4()
            topic_id = self._resolve_topic_id(
                getattr(delta, "topic_id", None),
                topic_remap,
                topic_ids,
                fallback=None,
            )
            update = {
                id_name: new_id,
                "content": content,
                "source_msg_ids": _dedupe_ids(
                    delta.source_msg_ids,
                    source_limit,
                ),
                "confidence": delta.confidence,
                "confirmation_count": 1,
                "last_confirmed_at": now,
            }
            if topic_ids is not None:
                update["topic_id"] = topic_id
            item = delta.model_copy(update=update)
            result.append(item)
            index = len(result) - 1
            by_id[new_id] = index
            by_content[_normalize_text(content)] = index
            used.add(new_id)
        return result

    @staticmethod
    def _resolve_topic_id(
        topic_id: UUID | None,
        topic_remap: dict[UUID, UUID] | None,
        topic_ids: set[UUID] | None,
        *,
        fallback: UUID | None,
    ) -> UUID | None:
        """把模型给的 topic id 映射到合并后的 topic；无效时回退。"""
        if topic_ids is None:
            return None
        mapped = topic_id
        if mapped is not None and topic_remap:
            mapped = topic_remap.get(mapped, mapped)
        if mapped in topic_ids:
            return mapped
        return fallback if fallback in topic_ids else None

    def _limit_summary_facts(self, facts: list[Any]) -> list[Any]:
        """超过 ``compression_max_facts`` 时遗忘最弱事实。

        排序依据：确认次数少、最近确认时间旧、置信度低；保留原有顺序。
        """
        limit = max(1, int(self.config.compression_max_facts))
        if len(facts) <= limit:
            return facts
        def rank(item: Any) -> tuple:
            return (
                item.confirmation_count,
                ensure_utc(item.last_confirmed_at),
                item.confidence,
            )
        ordered = sorted(facts, key=rank)
        removed = {
            item.fact_id for item in ordered[: len(facts) - limit]
        }
        return [item for item in facts if item.fact_id not in removed]

    def _apply_topic_forgetting(
        self,
        summary: SessionSummary,
        *,
        compression_count_after: int,
        conversation_id: UUID,
    ) -> SessionSummary:
        """按配置遗忘“插话” topic；最新 topic 与高 mention 计数优先保留。"""
        config = self.config
        topics = summary.topics
        if (
            compression_count_after
            <= config.compression_topic_forget_after
            or len(topics) <= config.compression_max_topics
        ):
            return summary
        latest = max(
            (ensure_utc(topic.last_mentioned_at) for topic in topics),
            default=None,
        )
        protected = {
            topic.id
            for topic in topics
            if config.compression_protect_latest_topic
            and ensure_utc(topic.last_mentioned_at) == latest
        }
        candidates = [
            topic
            for topic in topics
            if topic.id not in protected
            and topic.mention_count <= config.compression_interjection_max_mentions
        ]
        candidates.sort(
            key=lambda topic: (
                topic.mention_count,
                ensure_utc(topic.last_mentioned_at),
            )
        )
        forget_count = min(
            len(topics) - config.compression_max_topics,
            config.compression_max_forget_per_compression,
            len(candidates),
        )
        if forget_count <= 0:
            return summary
        removed = {
            topic.id for topic in candidates[:forget_count]
        }
        kept = [topic for topic in topics if topic.id not in removed]
        facts = [
            # topic 被遗忘只解除关联，不删除事实本身；约束没有 topic_id，
            # 始终保留。
            fact.model_copy(update={"topic_id": None})
            if fact.topic_id in removed
            else fact
            for fact in summary.facts
        ]
        logger.info(
            "会话摘要 topic 遗忘: conversation=%s removed=%d kept=%d",
            conversation_id,
            len(removed),
            len(kept),
        )
        return summary.model_copy(update={"topics": kept, "facts": facts})

    @staticmethod
    def _allowed_msg_ids(
        old: SessionSummary,
        rows: list[Any],
    ) -> set[UUID]:
        """允许的消息 id：本次待压缩消息 + 旧摘要事实/约束的证据。"""
        allowed = {
            row.id for row in rows if getattr(row, "id", None) is not None
        }
        for item in (*old.facts, *old.constraints):
            allowed.update(item.source_msg_ids)
        return allowed

    @staticmethod
    def _allowed_round_ids(
        old: SessionSummary,
        rows: list[Any],
    ) -> set[UUID]:
        """允许的轮次 id：本次待压缩消息的 round_id + 旧 topic 轮次锚点。"""
        allowed = {
            row.round_id
            for row in rows
            if getattr(row, "round_id", None) is not None
        }
        for topic in old.topics:
            allowed.update(topic.source_round_ids)
        return allowed

    def _messages_text(self, rows: list[Any]) -> str:
        """渲染消息并带上轮次/消息 id，供模型填写两类来源。"""
        lines: list[str] = []
        for row in rows:
            text = extract_content_text(row.content)
            if text:
                round_id = getattr(row, "round_id", None)
                lines.append(
                    f"[round:{round_id or '-'} msg:{row.id} seq:{row.seq_no}] "
                    f"{row.role.value}: {text}"
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
