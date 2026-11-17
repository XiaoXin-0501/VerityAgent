"""Chat provider / ChatModel 抽象。"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from provider.base import BaseProvider, ProviderModelSpec
from config.provider import ChatApiStyle, ChatProviderConfig
from provider.enums import ProviderKind

from .types import ChatModelRequest, ChatModelResponse, ChatStreamEvent

__all__ = ["ChatModel", "ChatProvider"]


class ChatModel:
    """单个 chat 模型句柄；只做厂商调用，不组装 Prompt。"""

    def __init__(self, spec: ProviderModelSpec) -> None:
        """校验模型类型并缓存模型级默认参数。"""
        if spec.kind.value != ProviderKind.CHAT.value:
            raise ValueError(f"{spec.id} 不是 chat 模型")
        self.spec = spec
        self._defaults = dict(spec.metadata.get("defaults") or {})

    def option(self, name: str, value):
        """请求值优先；未提供时使用模型 metadata.defaults。"""
        return value if value is not None else self._defaults.get(name)

    def thinking_payload(self) -> dict[str, Any]:
        """返回模型级 thinking 开关对应的请求字段。

        关闭时必须返回厂商的关闭参数（``thinking_disabled_options``），
        而不是仅在客户端丢弃 reasoning 内容；未配置时返回空字典，由各
        provider 按自身默认行为处理。
        """
        enabled = self.spec.metadata.get("thinking_enabled")
        if enabled is False:
            return dict(self.spec.metadata.get("thinking_disabled_options") or {})
        if enabled is True:
            return dict(self.spec.metadata.get("thinking_options") or {})
        return {}

    async def complete(self, request: ChatModelRequest) -> ChatModelResponse:
        """非流式调用；由具体厂商适配器实现。"""
        raise NotImplementedError

    async def stream(
        self, request: ChatModelRequest
    ) -> AsyncIterator[ChatStreamEvent]:
        """流式调用；实现方必须是 async generator。"""
        raise NotImplementedError
        yield  # pragma: no cover - 仅用于声明 async generator 接口

    async def count_tokens(self, request: ChatModelRequest) -> int | None:
        """可选精确计数；默认返回 None，由 LLM 层用估算。"""
        return None


class ChatProvider(BaseProvider[ChatProviderConfig]):
    kind = ProviderKind.CHAT

    def __init__(self, config: ChatProviderConfig) -> None:
        """保存 Provider 配置；不创建网络客户端。"""
        super().__init__(config)

    @property
    def api_style(self) -> ChatApiStyle:
        """返回该 Provider 使用的 Chat API 风格。"""
        return self.config.api_style

    @abstractmethod
    def get_model(self, model_id: str) -> ChatModel:
        """返回指定模型句柄；同一 model_id 由实现方负责复用。"""
        raise NotImplementedError
