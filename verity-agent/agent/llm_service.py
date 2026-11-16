"""LLM 应用层 ChatService：模型 fallback 与流式边界。

Provider 负责单模型调用；Agent 只依赖 ``LLMPort``。本服务在 LLMPort 之上做
**模型级 fallback**：仅在首 token 前失败时切换模型，流式已经产生输出后不再
重试，避免重复文本。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from agent.llm import LLMRequest, LLMResponse, LLMStreamEvent
from exception import ConfigurationError

__all__ = ["ChatService", "ModelRegistry"]


class ModelRegistry:
    """应用模型 id -> Provider 句柄的轻量注册表。

    不做能力选型/评分；只负责 allowed/fallback 集合与禁用模型校验，
    真正的路由由 ``RoutingChatAdapter`` 完成。
    """

    def __init__(
        self,
        provider_registry: Any,
        *,
        default_model_id: str,
        model_ids: list[str] | None = None,
        fallback_model_ids: list[str] | None = None,
    ) -> None:
        """保存 ProviderRegistry 与模型白名单。"""
        self.provider_registry = provider_registry
        self.default_model_id = default_model_id
        self.model_ids = list(model_ids or [])
        self.fallback_model_ids = list(fallback_model_ids or [])

    def get_model_by_id(self, model_id: str, *, include_disabled: bool = False):
        """按应用模型 id 取句柄；白名单非空时拒绝白名单外模型。"""
        if self.model_ids and model_id not in self.model_ids:
            raise ConfigurationError(
                f"模型不在 LLM model_ids 白名单: {model_id}",
                details={"model_id": model_id},
            )
        return self.provider_registry.get_model_by_id(
            model_id,
            include_disabled=include_disabled,
        )

    def resolve(self, model_id: str | None = None):
        """解析请求模型；为空时返回默认模型句柄。"""
        return self.get_model_by_id(model_id or self.default_model_id)


class ChatService:
    """包一层 LLMPort，提供模型 fallback；不实现 Prompt/Context 组装。"""

    def __init__(
        self,
        llm: Any,
        *,
        fallback_model_ids: list[str] | None = None,
        enable_fallback: bool = True,
    ) -> None:
        """保存底层 LLMPort 与 fallback 列表。"""
        self.llm = llm
        self.fallback_model_ids = list(fallback_model_ids or [])
        self.enable_fallback = enable_fallback

    def _targets(self, request: LLMRequest) -> list[str]:
        """按请求模型 + fallback 顺序生成候选，去重且保持顺序。"""
        values: list[str] = []
        for item in [request.model_id, *self.fallback_model_ids]:
            if item and item not in values:
                values.append(item)
        return values

    async def chat(self, request: LLMRequest) -> LLMResponse:
        """非流式调用；主模型失败后按配置尝试 fallback。"""
        targets = self._targets(request)
        if not self.enable_fallback:
            targets = targets[:1]
        last_error: Exception | None = None
        for index, model_id in enumerate(targets):
            try:
                response = await self.llm.chat(
                    request.model_copy(update={"model_id": model_id})
                )
                if index > 0:
                    response = response.model_copy(
                        update={"warnings": [*response.warnings, "MODEL_FALLBACK"]}
                    )
                return response
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - fallback 由本层决定
                last_error = exc
                continue
        assert last_error is not None
        raise last_error

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        """流式调用；只在首事件前失败时切换 fallback。"""
        targets = self._targets(request)
        if not self.enable_fallback:
            targets = targets[:1]
        last_error: Exception | None = None
        for model_id in targets:
            iterator = self.llm.stream(
                request.model_copy(update={"model_id": model_id})
            ).__aiter__()
            emitted = False
            try:
                while True:
                    try:
                        event = await iterator.__anext__()
                    except StopAsyncIteration:
                        return
                    if event.type == "error" and not emitted:
                        # provider adapter 把首事件失败编码成 error event；
                        # 此时还没有对外输出，允许切到 fallback 模型。
                        raise RuntimeError(event.error_message or "stream error")
                    emitted = True
                    yield event
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - 首事件前可 fallback
                last_error = exc
                if emitted:
                    raise
                continue
        if last_error is not None:
            raise last_error
