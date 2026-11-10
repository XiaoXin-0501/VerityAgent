"""Cohere / BGE 兼容 Rerank adapter。

默认使用 ``POST /rerank``，请求体兼容 Cohere Rerank 与常见 BGE reranker
服务；文档格式可通过 ``metadata["documents_format"]="object"`` 切换。
"""

from __future__ import annotations

from typing import Any, Mapping

from config.provider import RerankProviderConfig
from exception.provider import ProviderError, ProviderErrorCode
from api.network import HttpProviderMixin
from provider.reranker.base import RerankModel, RerankProvider
from provider.reranker.types import (
    RerankRequest,
    RerankResponse,
    RerankResult,
    RerankUsage,
)

__all__ = ["CohereRerankModel", "CohereRerankProvider", "BgeRerankProvider"]


class CohereRerankModel(RerankModel):
    def __init__(self, spec, provider: "CohereRerankProvider") -> None:
        super().__init__(spec)
        self.provider = provider

    def _payload(self, request: RerankRequest) -> dict[str, Any]:
        object_format = self.provider.config.metadata.get("documents_format") == "object"
        documents = (
            [{"text": doc.text} for doc in request.documents]
            if object_format
            else [doc.text for doc in request.documents]
        )
        payload: dict[str, Any] = {
            "model": self.spec.model,
            "query": request.query,
            "documents": documents,
            "return_documents": request.return_documents,
        }
        if request.top_k is not None:
            payload["top_n"] = min(request.top_k, len(request.documents))
        if self.provider.config.metadata.get("send_input_type"):
            payload["input_type"] = self.provider.config.metadata.get("input_type")
        return payload

    async def rerank(self, request: RerankRequest) -> RerankResponse:
        data = await self.provider.request_json(
            "POST",
            "rerank",
            payload=self._payload(request),
            model=self.spec.model,
        )
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


class CohereRerankProvider(RerankProvider, HttpProviderMixin):
    model_class = CohereRerankModel
    default_capabilities = {
        "supports_return_documents": True,
        "max_documents": 100,
    }

    def __init__(self, config: RerankProviderConfig) -> None:
        super().__init__(config)
        self._init_http()
        self._handles: dict[str, RerankModel] = {}

    async def open(self) -> None:
        await super().open()
        await self.open_http()

    async def close(self) -> None:
        await self.close_http()
        await super().close()

    def get_model(self, model_id: str) -> RerankModel:
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
