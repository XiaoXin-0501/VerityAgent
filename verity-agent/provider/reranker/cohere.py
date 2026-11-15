"""Cohere / BGE 兼容 Rerank adapter。

默认使用 ``POST /rerank``，请求体兼容 Cohere Rerank 与常见 BGE reranker
服务；文档格式可通过 ``metadata["documents_format"]="object"`` 切换。
"""

from __future__ import annotations

from typing import Any, Mapping

import cohere

from config.provider import RerankProviderConfig
from exception.provider import ProviderError, ProviderErrorCode
from provider.reranker.base import RerankModel, RerankProvider
from provider.sdk import (
    build_cohere_client,
    close_cohere_client,
    sdk_dump,
    sdk_error,
)
from cohere.core.request_options import RequestOptions
from provider.reranker.types import (
    RerankRequest,
    RerankResponse,
    RerankResult,
    RerankUsage,
)

__all__ = ["CohereRerankModel", "CohereRerankProvider", "BgeRerankProvider"]


class CohereRerankModel(RerankModel):
    def __init__(self, spec, provider: "CohereRerankProvider") -> None:
        """绑定模型规格与所属 Provider。"""
        super().__init__(spec)
        self.provider = provider

    async def rerank(self, request: RerankRequest) -> RerankResponse:
        """调用 Cohere rerank 接口并转换为统一响应。"""
        object_format = self.provider.config.metadata.get("documents_format") == "object"
        documents = (
            [{"text": doc.text} for doc in request.documents]
            if object_format
            else [doc.text for doc in request.documents]
        )
        extra_body: dict[str, Any] = {}
        if self.provider.config.metadata.get("send_input_type"):
            extra_body["input_type"] = self.provider.config.metadata.get("input_type")
        options: RequestOptions = {
            "max_retries": self.provider.config.max_retries,
            "timeout_in_seconds": int(self.provider.config.timeout_s),
        }
        if self.provider.config.headers:
            options["additional_headers"] = dict(self.provider.config.headers)
        if extra_body:
            options["additional_body_parameters"] = extra_body
        kwargs: dict[str, Any] = {
            "model": self.spec.model,
            "query": request.query,
            "documents": documents,
            "return_documents": request.return_documents,
        }
        if request.top_k is not None:
            kwargs["top_n"] = min(request.top_k, len(request.documents))
        try:
            response = await self.provider.client.rerank(
                request_options=options,
                **kwargs,
            )
        except Exception as exc:
            raise sdk_error(
                exc,
                provider=self.provider.config.name,
                model=self.spec.model,
            ) from exc
        data = sdk_dump(response)
        results = data.get("results")
        if not isinstance(results, list):
            raise ProviderError(
                ProviderErrorCode.RESPONSE_INVALID,
                "Rerank 响应缺少 results",
                provider=self.provider.config.name,
                model=self.spec.model,
            )
        mapped: list[RerankResult] = []
        for item in results:
            if not isinstance(item, Mapping):
                continue
            index = int(item.get("index") or 0)
            if index < 0 or index >= len(request.documents):
                continue
            document = item.get("document")
            if isinstance(document, Mapping):
                document = document.get("text")
            mapped.append(
                RerankResult(
                    index=index,
                    id=request.documents[index].id,
                    score=float(item.get("relevance_score") or item.get("score") or 0.0),
                    document=str(document) if document is not None else None,
                )
            )
        mapped.sort(key=lambda item: item.score, reverse=True)
        meta = data.get("meta") or {}
        billed = meta.get("billed_units") or {}
        return RerankResponse(
            results=mapped,
            model=str(data.get("model") or self.spec.model),
            usage=RerankUsage(
                total_tokens=int(billed.get("search_units") or 0),
            ),
            request_id=data.get("id"),
        )


class CohereRerankProvider(RerankProvider):
    model_class = CohereRerankModel
    default_capabilities = {
        "supports_return_documents": True,
        "max_documents": 100,
    }

    def __init__(self, config: RerankProviderConfig) -> None:
        """初始化 SDK 客户端占位与模型句柄缓存。"""
        super().__init__(config)
        self._client: cohere.AsyncClient | None = None
        self._handles: dict[str, RerankModel] = {}

    @property
    def client(self) -> cohere.AsyncClient:
        """返回已初始化的 SDK 客户端；未 open 时抛 UNAVAILABLE。"""
        if self._client is None:
            raise ProviderError(
                ProviderErrorCode.UNAVAILABLE,
                "Cohere SDK 客户端尚未初始化",
                provider=self.config.name,
            )
        return self._client

    async def open(self) -> None:
        """构建并保存 Cohere 客户端。"""
        client = build_cohere_client(self.config)
        await super().open()
        self._client = client

    async def close(self) -> None:
        """关闭 Cohere 客户端并清理状态。"""
        if self._client is not None:
            await close_cohere_client(self._client)
            self._client = None
        await super().close()

    def get_model(self, model_id: str) -> RerankModel:
        """按 model_id 复用或创建模型句柄。"""
        model = self._handles.get(model_id)
        if model is None:
            model = self.model_class(self.get_spec(model_id), self)
            self._handles[model_id] = model
        return model


class BgeRerankProvider(CohereRerankProvider):
    """BGE reranker 的 Cohere-compatible HTTP 服务。"""

    default_capabilities = {
        "supports_return_documents": True,
        "max_documents": 100,
    }
