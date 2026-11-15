"""Agent 运行状态与事件 Redis Stream 持久化（多实例 replay 基础）。

内存 ring buffer 仍是单实例主路径；本类负责：
- run 状态快照（status/last_event_id/result）；
- 事件流（XADD/XRANGE）；
- 跨实例取消标记（cancel_requested）。

Redis 不可用时自动禁用自身，不阻塞 Agent 主循环。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from agent.public import AgentEvent, AgentResult, AgentRunView
from store.redis import RedisConnectionManager, get_redis
from store.redis.keys import RedisKeyPrefix, make_key
from utils.serialization import json_dumps, json_loads
from utils.time import utcnow

__all__ = ["RedisEventStore"]

logger = logging.getLogger(__name__)



class RedisEventStore:
    """Redis Stream + run snapshot；所有方法 fail-open。"""

    def __init__(
        self,
        redis: RedisConnectionManager | None = None,
        *,
        ttl_s: int = 3600,
    ) -> None:
        """保存 Redis 管理器与 TTL；不建立连接。"""
        self.redis = redis or get_redis()
        self.ttl_s = max(60, int(ttl_s))
        self._disabled = False

    def _run_key(self, run_id: str) -> str:
        """run 快照 hash key。"""
        return make_key(RedisKeyPrefix.RUN, run_id)

    def _stream_key(self, run_id: str) -> str:
        """run 事件 stream key。"""
        return make_key(RedisKeyPrefix.STREAM, "run", run_id)

    async def _call(self, func, *args, **kwargs):
        """执行 Redis 命令；失败后禁用自身并返回 None。"""
        if self._disabled:
            return None
        try:
            return await func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - 事件持久化 fail-open
            self._disabled = True
            logger.warning("Agent Redis 事件存储不可用，降级为内存: %s", exc)
            return None

    async def save_run(
        self,
        *,
        run_id: str,
        conversation_id: Any,
        turn_id: str,
        status: str,
        result: AgentResult | None = None,
    ) -> None:
        """写入运行快照并刷新 TTL。"""
        payload = {
            "run_id": run_id,
            "conversation_id": str(conversation_id),
            "turn_id": turn_id,
            "status": status,
            "updated_at": utcnow().isoformat(),
        }
        if result is not None:
            payload["result"] = json_dumps(result)
        await self._call(
            self.redis.hset,
            self._run_key(run_id),
            mapping=payload,
        )
        await self._call(self.redis.expire, self._run_key(run_id), self.ttl_s)

    async def append_event(self, event: AgentEvent) -> None:
        """XADD 事件并更新快照 last_event_id/status。"""
        data = json_dumps(event)
        await self._call(
            self.redis.execute_command,
            "XADD",
            self._stream_key(event.run_id),
            "*",
            "data",
            data,
        )
        await self._call(
            self.redis.hset,
            self._run_key(event.run_id),
            mapping={
                "last_event_id": event.id,
                "status": _status_from_event(event),
            },
        )
        await self._call(
            self.redis.expire,
            self._stream_key(event.run_id),
            self.ttl_s,
        )

    async def snapshot(self, run_id: str) -> AgentRunView | None:
        """读取 run 视图；不存在/损坏返回 None。"""
        raw = await self._call(self.redis.hgetall, self._run_key(run_id))
        if not raw:
            return None
        try:
            result = (
                AgentResult.model_validate(json_loads(raw["result"]))
                if raw.get("result")
                else None
            )
            return AgentRunView(
                run_id=raw.get("run_id", run_id),
                turn_id=raw.get("turn_id", ""),
                conversation_id=raw["conversation_id"],
                status=raw.get("status", "running"),
                last_event_id=raw.get("last_event_id"),
                result=result,
            )
        except Exception:  # noqa: BLE001 - 快照损坏按不存在
            logger.warning("Agent run 快照不合法: %s", run_id)
            return None

    async def replay(
        self,
        run_id: str,
        after_event_id: str | None = None,
    ) -> list[AgentEvent]:
        """读取事件流；过滤 ``after_event_id`` 之前的序号。"""
        raw = await self._call(
            self.redis.execute_command,
            "XRANGE",
            self._stream_key(run_id),
            "-",
            "+",
        )
        if not raw:
            return []
        after_seq = _seq(after_event_id)
        events: list[AgentEvent] = []
        for _stream_id, fields in raw:
            data = fields.get("data") if isinstance(fields, dict) else None
            if not data:
                continue
            try:
                event = AgentEvent.model_validate(json_loads(data))
            except Exception:  # noqa: BLE001 - 坏事件跳过
                continue
            if event.seq > after_seq:
                events.append(event)
        return events

    async def request_cancel(self, run_id: str, reason: str = "user") -> None:
        """写跨实例取消标记，并发布 Pub/Sub 通知。"""
        await self._call(
            self.redis.hset,
            self._run_key(run_id),
            mapping={"cancel_requested": "1", "cancel_reason": reason},
        )
        await self._call(
            self.redis.publish,
            self._cancel_channel(run_id),
            reason,
        )

    async def watch_cancel(self, run_id: str):
        """订阅取消 Pub/Sub；连接失败/Redis 不可用时自动结束。"""
        if self._disabled:
            return
        pubsub = None
        try:
            pubsub = self.redis.client.pubsub()
            await pubsub.subscribe(self._cancel_channel(run_id))
            while not self._disabled:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=1.0,
                )
                if message and message.get("type") == "message":
                    yield str(message.get("data") or "")
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - Pub/Sub 失败退回轮询
            logger.warning("取消 Pub/Sub 订阅失败: %s", exc)
            return
        finally:
            if pubsub is not None:
                try:
                    await pubsub.aclose()
                except Exception:  # pragma: no cover - 关闭失败不抛出
                    pass

    def _cancel_channel(self, run_id: str) -> str:
        """取消通知 channel。"""
        return make_key(RedisKeyPrefix.RUN, run_id, "cancel")

    async def cancel_requested(self, run_id: str) -> bool:
        """读取取消标记；Redis 不可用时返回 False。"""
        value = await self._call(self.redis.hget, self._run_key(run_id), "cancel_requested")
        return str(value) == "1"


def _status_from_event(event: AgentEvent) -> str:
    """事件类型 -> run 状态投影。"""
    value = event.event_type.value
    if value == "run.completed":
        return "completed"
    if value == "run.failed":
        return "failed"
    if value == "run.cancelled":
        return "cancelled"
    if value in {"tool.started", "tool.result", "assistant.tool_calls"}:
        return "waiting_tool"
    return "running"


def _seq(event_id: str | None) -> int:
    """解析 ``run_id:seq``；非法值返回 0（全量回放）。"""
    if not event_id:
        return 0
    try:
        return int(str(event_id).rsplit(":", 1)[-1])
    except (TypeError, ValueError):
        return 0
