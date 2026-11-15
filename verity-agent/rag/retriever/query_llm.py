"""Query 改写与意图识别的 LLM 适配。

规则层在 :mod:`rag.retriever.query` 中完成；本模块只处理“规则无法判断”或
“明确开启改写”的场景。LLM 结果按 query + 模型 + Prompt 版本缓存到 Redis，
避免同一 query 重复消耗 token。

设计约束：

- 失败/超时只返回空结果或 None，由检索流水线降级，不抛穿到 Agent；
- 输出解析尽量宽松（JSON 数组/换行/编号列表），但只接受枚举内意图；
- 不在这里拼检索请求，只负责一次文本生成。
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from provider.llm.types import (
    ChatModelRequest,
    ModelGenerationOptions,
    ModelMessage,
    ModelTextPart,
)
from provider.enums import ModelRole
from rag.retriever.cache import RetrievalCache
from rag.shared.enums import QueryIntent
from store.redis.cache import CacheKey, CacheScope
from utils.hashing import sha256_text

__all__ = ["ChatQueryRewriter", "ChatIntentClassifier"]

logger = logging.getLogger(__name__)

# Prompt 版本变化会让缓存自然失效；改 prompt 时同步递增。
_REWRITE_PROMPT_VERSION = "v1"
_INTENT_PROMPT_VERSION = "v1"

_REWRITE_SYSTEM = """你是检索查询改写器。把用户问题改写成适合知识库检索的简短查询。
要求：
1. 保留原始实体、数字、否定和语言；
2. 最多输出 {max_variants} 条，每条一行，不要编号、不要解释；
3. 不要引入原问题没有的事实；
4. 如果原问题已经足够清晰，可以只输出原问题。"""

_INTENT_SYSTEM = """你是查询意图分类器。只能从以下枚举中选择一个：
LOOKUP（查定义/事实）、COMPARE（比较/区别）、SUMMARIZE（总结/概括）、HOWTO（步骤/方法）。
只输出枚举名，不要解释；无法判断输出 UNKNOWN。"""


def _model_text(response: Any) -> str:
    """从 provider ChatModelResponse 中提取普通文本，忽略 thinking。"""
    message = getattr(response, "message", None)
    parts = getattr(message, "content", None) or []
    return "".join(
        part.text
        for part in parts
        if isinstance(part, ModelTextPart)
    ).strip()


def _parse_variants(text: str, *, max_variants: int) -> list[str]:
    """解析改写结果：优先 JSON 数组，其次逐行文本。"""
    if not text:
        return []
    values: list[str] = []
    stripped = text.strip()
    if stripped.startswith("["):
        try:
            payload = json.loads(stripped)
            if isinstance(payload, list):
                values = [str(item).strip() for item in payload]
        except (ValueError, TypeError):
            values = []
    if not values:
        values = [
            line.strip().lstrip("-*0123456789.、)） ").strip()
            for line in stripped.splitlines()
        ]
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
        if len(result) >= max_variants:
            break
    return result


def _parse_intent(text: str) -> QueryIntent | None:
    """解析意图输出；只接受枚举内值，其他一律按无法判断处理。"""
    value = (text or "").strip().upper()
    for item in QueryIntent:
        if item.value.upper() == value or item.name == value:
            return item
    # 兼容模型输出“意图：COMPARE”这类带前缀的文本。
    for item in QueryIntent:
        if item.name in value:
            return item
    return None


class ChatQueryRewriter:
    """基于 Provider ChatModel 的多查询改写器。"""

    def __init__(
        self,
        model: Any,
        *,
        model_id: str,
        max_variants: int = 2,
        timeout_s: float = 20.0,
        max_output_tokens: int = 256,
        cache: RetrievalCache | None = None,
        cache_ttl_s: int = 86_400,
    ) -> None:
        """保存模型、超时与可选缓存；不发起网络请求。"""
        self.model = model
        self.model_id = model_id
        self.max_variants = max(1, int(max_variants))
        self.timeout_s = max(0.1, float(timeout_s))
        self.max_output_tokens = max(32, int(max_output_tokens))
        self.cache = cache
        self.cache_ttl_s = max(1, int(cache_ttl_s))

    async def rewrite(self, query: str) -> list[str]:
        """返回改写变体（不含原 query）；失败/超时返回空列表。"""
        cached = await self._read_cache(query)
        if cached is not None:
            return cached
        request = ChatModelRequest(
            model=self.model_id,
            messages=[
                ModelMessage(
                    role=ModelRole.SYSTEM,
                    content=[
                        ModelTextPart(
                            text=_REWRITE_SYSTEM.format(
                                max_variants=self.max_variants
                            )
                        )
                    ],
                ),
                ModelMessage(
                    role=ModelRole.USER,
                    content=[ModelTextPart(text=query)],
                ),
            ],
            options=ModelGenerationOptions(
                temperature=0.2,
                max_output_tokens=self.max_output_tokens,
            ),
        )
        try:
            response = await asyncio.wait_for(
                self.model.complete(request),
                timeout=self.timeout_s,
            )
        except asyncio.TimeoutError:
            logger.warning("query 改写超时: model=%s", self.model_id)
            return []
        except Exception as exc:  # noqa: BLE001 - LLM 改写可降级
            logger.warning("query 改写失败: %s", exc)
            return []
        variants = _parse_variants(_model_text(response), max_variants=self.max_variants)
        variants = [item for item in variants if item and item != query]
        if variants:
            await self._write_cache(query, variants)
        return variants

    def _cache_key(self, query: str) -> CacheKey:
        """改写缓存 key：query 摘要 + 模型 + Prompt 版本。"""
        return CacheKey(
            scope=CacheScope.QUERY_REWRITE,
            query_digest=sha256_text(query),
            model_id=self.model_id,
            version=f"{_REWRITE_PROMPT_VERSION}:{self.max_variants}",
        )

    async def _read_cache(self, query: str) -> list[str] | None:
        """读取改写缓存；非法值按 miss 处理。"""
        if self.cache is None:
            return None
        try:
            payload = await self.cache.get(self._cache_key(query))
            if payload is None:
                return None
            if isinstance(payload, list):
                return [str(item) for item in payload if str(item).strip()]
        except Exception as exc:  # noqa: BLE001 - 缓存失败回源
            logger.warning("query 改写缓存读取失败: %s", exc)
        return None

    async def _write_cache(self, query: str, variants: list[str]) -> None:
        """写入改写缓存；失败不影响本次检索。"""
        if self.cache is None:
            return
        try:
            await self.cache.set(
                self._cache_key(query),
                variants,
                ttl_s=self.cache_ttl_s,
            )
        except Exception as exc:  # noqa: BLE001 - 缓存失败不阻断
            logger.warning("query 改写缓存写入失败: %s", exc)


class ChatIntentClassifier:
    """规则未命中时的 LLM 意图兜底分类器。"""

    def __init__(
        self,
        model: Any,
        *,
        model_id: str,
        timeout_s: float = 5.0,
        max_output_tokens: int = 16,
        cache: RetrievalCache | None = None,
        cache_ttl_s: int = 86_400,
    ) -> None:
        """保存模型、超时与可选缓存。"""
        self.model = model
        self.model_id = model_id
        self.timeout_s = max(0.1, float(timeout_s))
        self.max_output_tokens = max(4, int(max_output_tokens))
        self.cache = cache
        self.cache_ttl_s = max(1, int(cache_ttl_s))

    async def classify(self, query: str) -> QueryIntent | None:
        """规则未命中时调用 LLM；失败/超时返回 None。"""
        cached = await self._read_cache(query)
        if cached is not None:
            return cached
        request = ChatModelRequest(
            model=self.model_id,
            messages=[
                ModelMessage(
                    role=ModelRole.SYSTEM,
                    content=[ModelTextPart(text=_INTENT_SYSTEM)],
                ),
                ModelMessage(
                    role=ModelRole.USER,
                    content=[ModelTextPart(text=query)],
                ),
            ],
            options=ModelGenerationOptions(
                temperature=0.0,
                max_output_tokens=self.max_output_tokens,
            ),
        )
        try:
            response = await asyncio.wait_for(
                self.model.complete(request),
                timeout=self.timeout_s,
            )
        except asyncio.TimeoutError:
            logger.warning("意图分类超时: model=%s", self.model_id)
            return None
        except Exception as exc:  # noqa: BLE001 - 意图兜底可降级
            logger.warning("意图分类失败: %s", exc)
            return None
        intent = _parse_intent(_model_text(response))
        if intent is not None:
            await self._write_cache(query, intent)
        return intent

    def _cache_key(self, query: str) -> CacheKey:
        """意图缓存 key：query 摘要 + 模型 + Prompt 版本。"""
        return CacheKey(
            scope=CacheScope.QUERY_INTENT,
            query_digest=sha256_text(query),
            model_id=self.model_id,
            version=_INTENT_PROMPT_VERSION,
        )

    async def _read_cache(self, query: str) -> QueryIntent | None:
        """读取意图缓存；非法值按 miss。"""
        if self.cache is None:
            return None
        try:
            payload = await self.cache.get(self._cache_key(query))
            if isinstance(payload, str):
                return _parse_intent(payload)
        except Exception as exc:  # noqa: BLE001 - 缓存失败回源
            logger.warning("意图缓存读取失败: %s", exc)
        return None

    async def _write_cache(self, query: str, intent: QueryIntent) -> None:
        """写入意图缓存；失败不阻断。"""
        if self.cache is None:
            return
        try:
            await self.cache.set(
                self._cache_key(query),
                intent.value,
                ttl_s=self.cache_ttl_s,
            )
        except Exception as exc:  # noqa: BLE001 - 缓存失败不阻断
            logger.warning("意图缓存写入失败: %s", exc)
