"""OpenAI Embeddings 兼容 adapter（OpenAI / BGE / 其他兼容服务）。"""

from __future__ import annotations

from typing import Any, Mapping

import openai

from config.provider import EmbeddingProviderConfig
from provider.embedding.base import EmbeddingModel, EmbeddingProvider
from provider.embedding.types import EmbeddingRequest, EmbeddingResponse, EmbeddingUsage
from exception.provider import ProviderError, ProviderErrorCode
from provider.sdk import (
    build_openai_client,
    sdk_dump,
    sdk_error,
    split_sdk_kwargs,
)

__all__ = ["OpenAIEmbeddingModel", "OpenAIEmbeddingProvider", "BgeEmbeddingProvider"]

_EMBEDDING_KWARGS = frozenset(
    {"model", "input", "encoding_format", "dimensions", "user"}
)


class OpenAIEmbeddingModel(EmbeddingModel):
    def __init__(self, spec, provider: "OpenAIEmbeddingProvider") -> None:
        super().__init__(spec)
        self.provider = provider

    def _payload(self, request: EmbeddingRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.spec.model,
            "input": request.inputs,
            "encoding_format": "float",
        }
        caps = self.spec.embedding_capabilities
        if request.dimensions is not None and caps.supports_dimensions:
            payload["dimensions"] = request.dimensions
        if request.input_type and self.provider.config.metadata.get("send_input_type"):
            payload["input_type"] = request.input_type
        if self.provider.config.metadata.get("send_normalize"):
            payload["normalize"] = request.normalize
        return payload

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        payload = self._payload(request)
        try:
            response = await self.provider.client.embeddings.create(
                **split_sdk_kwargs(payload, _EMBEDDING_KWARGS)
            )
        except Exception as exc:
            raise sdk_error(
                exc,
                provider=self.provider.config.name,
                model=self.spec.model,
            ) from exc
        data = sdk_dump(response)
        items = data.get("data")
        if not isinstance(items, list) or not items:
            raise ProviderError(
                ProviderErrorCode.RESPONSE_INVALID,
                "Embedding 响应缺少 data",
                provider=self.provider.config.name,
                model=self.spec.model,
            )
        ordered = sorted(items, key=lambda item: int(item.get("index") or 0))
        vectors: list[list[float]] = []
        for item in ordered:
            vector = item.get("embedding") if isinstance(item, Mapping) else None
            if not isinstance(vector, list):
                raise ProviderError(
                    ProviderErrorCode.RESPONSE_INVALID,
                    "Embedding 响应 embedding 不是数组",
                    provider=self.provider.config.name,
                    model=self.spec.model,
                )
            vectors.append([float(value) for value in vector])
        dimensions = len(vectors[0])
        for vector in vectors:
            if len(vector) != dimensions:
                raise ProviderError(
                    ProviderErrorCode.RESPONSE_INVALID,
                    "Embedding 向量维度不一致",
                    provider=self.provider.config.name,
                    model=self.spec.model,
                )
        usage = data.get("usage") or {}
        return EmbeddingResponse(
            vectors=vectors,
            dimensions=dimensions,
            usage=EmbeddingUsage(
                input_tokens=int(usage.get("prompt_tokens") or usage.get("total_tokens") or 0),
                total_tokens=int(usage.get("total_tokens") or 0),
            ),
            model=str(data.get("model") or self.spec.model),
            request_id=data.get("id"),
        )


class OpenAIEmbeddingProvider(EmbeddingProvider):
    model_class = OpenAIEmbeddingModel
    default_capabilities: dict[str, Any] = {}

    def __init__(self, config: EmbeddingProviderConfig) -> None:
        super().__init__(config)
        self._client: openai.AsyncOpenAI | None = None
        self._handles: dict[str, EmbeddingModel] = {}

    @property
    def client(self) -> openai.AsyncOpenAI:
        if self._client is None:
            raise ProviderError(
                ProviderErrorCode.UNAVAILABLE,
                "OpenAI SDK 客户端尚未初始化",
                provider=self.config.name,
            )
        return self._client

    async def open(self) -> None:
        client = build_openai_client(self.config)
        await super().open()
        self._client = client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
        await super().close()

    def get_model(self, model_id: str) -> EmbeddingModel:
        model = self._handles.get(model_id)
        if model is None:
            model = self.model_class(self.get_spec(model_id), self)
            self._handles[model_id] = model
        return model


class BgeEmbeddingProvider(OpenAIEmbeddingProvider):
    """BGE-M3 等 OpenAI-compatible embedding 服务。"""

    default_capabilities = {
        "supports_normalize": True,
        "input_type": "query/document",
    }
