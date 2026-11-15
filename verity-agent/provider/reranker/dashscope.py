"""DashScope / 百炼原生 rerank adapter。

百炼 rerank 的 OpenAI-compatible 网关并不提供 ``/rerank``；原生接口是：

``POST /api/v1/services/rerank/text-rerank/text-rerank``

请求体为 ``{model, input: {query, documents}, parameters: {top_n, ...}}``。
该适配器直接使用 httpx，不依赖 Cohere SDK，避免把原生接口硬套成 Cohere wire。

接口路径可通过 ``ModelConfig.metadata.endpoint_path`` 覆盖；默认值只针对
百炼标准接口，换网关时不需要改代码。
"""

from __future__ import annotations

from typing import Any, Mapping

import httpx

from config.provider import RerankProviderConfig
from exception.provider import ProviderError, ProviderErrorCode
from provider.reranker.base import RerankModel, RerankProvider
from provider.reranker.types import (
    RerankRequest,
    RerankResponse,
    RerankResult,
    RerankUsage,
)

__all__ = ["DashScopeRerankModel", "DashScopeRerankProvider"]

_DEFAULT_ENDPOINT_PATH = "/api/v1/services/rerank/text-rerank/text-rerank"
# 常见 HTTP 状态 → ProviderErrorCode；其余按上游错误处理。
_STATUS_CODES = {
    400: ProviderErrorCode.INVALID_REQUEST,
    401: ProviderErrorCode.AUTHENTICATION,
    403: ProviderErrorCode.PERMISSION,
    404: ProviderErrorCode.MODEL_NOT_FOUND,
    429: ProviderErrorCode.RATE_LIMITED,
}
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _error_from_response(response: httpx.Response, *, provider: str, model: str) -> ProviderError:
    """把 DashScope HTTP 错误转成统一 ProviderError，不泄漏密钥。"""
    code = _STATUS_CODES.get(
        response.status_code,
        ProviderErrorCode.UPSTREAM
        if response.status_code >= 500
        else ProviderErrorCode.INVALID_REQUEST,
    )
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    if not isinstance(payload, Mapping):
        payload = {}
    message = (
        payload.get("message")
        or payload.get("code")
        or response.text[:300]
        or f"HTTP {response.status_code}"
    )
    return ProviderError(
        code,
        str(message),
        provider=provider,
        model=model,
        retryable=response.status_code in _RETRYABLE_STATUS,
        details={"status_code": response.status_code},
    )


class DashScopeRerankModel(RerankModel):
    """百炼原生 rerank 接口。"""

    def __init__(self, spec, provider: "DashScopeRerankProvider") -> None:
        """绑定模型规格与所属 Provider。"""
        super().__init__(spec)
        self.provider = provider

    def _endpoint_url(self) -> str:
        """拼接原生接口 URL；``metadata.endpoint_path`` 可覆盖默认路径。"""
        path = str(
            self.provider.config.metadata.get("endpoint_path")
            or _DEFAULT_ENDPOINT_PATH
        )
        base = self.provider.config.base_url.rstrip("/")
        if not base:
            raise ProviderError(
                ProviderErrorCode.INVALID_REQUEST,
                "DashScope rerank 缺少 base_url",
                provider=self.provider.name,
                model=self.spec.model,
            )
        return base if base.endswith(path) else f"{base}{path}"

    async def rerank(self, request: RerankRequest) -> RerankResponse:
        """调用 DashScope 原生 rerank，并映射到统一 ``RerankResponse``。"""
        documents = [document.text for document in request.documents]
        parameters: dict[str, Any] = {
            "return_documents": request.return_documents,
        }
        if request.top_k is not None:
            parameters["top_n"] = min(request.top_k, len(documents))
        payload = {
            "model": self.spec.model,
            "input": {
                "query": request.query,
                "documents": documents,
            },
            "parameters": parameters,
        }
        try:
            response = await self.provider.client.post(
                self._endpoint_url(),
                json=payload,
            )
        except httpx.TimeoutException as exc:
            raise ProviderError(
                ProviderErrorCode.TIMEOUT,
                "DashScope rerank 请求超时",
                provider=self.provider.name,
                model=self.spec.model,
                retryable=True,
            ) from exc
        except httpx.RequestError as exc:
            raise ProviderError(
                ProviderErrorCode.CONNECTION,
                "DashScope rerank 连接失败",
                provider=self.provider.name,
                model=self.spec.model,
                retryable=True,
            ) from exc
        if response.status_code >= 400:
            raise _error_from_response(
                response,
                provider=self.provider.name,
                model=self.spec.model,
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(
                ProviderErrorCode.RESPONSE_INVALID,
                "DashScope rerank 响应不是 JSON",
                provider=self.provider.name,
                model=self.spec.model,
            ) from exc
        output = data.get("output") if isinstance(data, Mapping) else None
        items = output.get("results") if isinstance(output, Mapping) else None
        if not isinstance(items, list):
            raise ProviderError(
                ProviderErrorCode.RESPONSE_INVALID,
                "DashScope rerank 响应缺少 output.results",
                provider=self.provider.name,
                model=self.spec.model,
            )

        mapped: list[RerankResult] = []
        for item in items:
            if not isinstance(item, Mapping):
                continue
            index = int(item.get("index") or 0)
            if index < 0 or index >= len(request.documents):
                continue
            document = item.get("document")
            text = (
                str(document.get("text"))
                if isinstance(document, Mapping) and document.get("text") is not None
                else None
            )
            score = item.get("relevance_score", item.get("score", 0.0))
            mapped.append(
                RerankResult(
                    index=index,
                    id=request.documents[index].id,
                    score=float(score),
                    document=text,
                )
            )
        mapped.sort(key=lambda item: item.score, reverse=True)
        usage = data.get("usage") if isinstance(data, Mapping) else None
        total_tokens = (
            int(usage.get("total_tokens") or 0)
            if isinstance(usage, Mapping)
            else 0
        )
        return RerankResponse(
            results=mapped,
            model=str(data.get("model") or self.spec.model),
            usage=RerankUsage(total_tokens=total_tokens),
            request_id=(
                str(data.get("request_id"))
                if isinstance(data, Mapping) and data.get("request_id")
                else None
            ),
        )


class DashScopeRerankProvider(RerankProvider):
    """百炼 rerank provider；httpx client 在 ``open()`` 时创建。"""

    default_capabilities = {
        "max_documents": 100,
        "max_query_tokens": 512,
        "max_document_tokens": 4096,
        "max_batch_size": 16,
        "supports_return_documents": True,
    }

    def __init__(self, config: RerankProviderConfig) -> None:
        """初始化 HTTP 客户端占位与模型句柄缓存。"""
        super().__init__(config)
        self._client: httpx.AsyncClient | None = None
        self._handles: dict[str, DashScopeRerankModel] = {}

    @property
    def client(self) -> httpx.AsyncClient:
        """返回已初始化的 HTTP 客户端；未 open 时抛 UNAVAILABLE。"""
        if self._client is None:
            raise ProviderError(
                ProviderErrorCode.UNAVAILABLE,
                "DashScope rerank 客户端尚未初始化",
                provider=self.name,
            )
        return self._client

    async def open(self) -> None:
        """创建带认证/超时/重试的 httpx 客户端。"""
        headers = dict(self.config.headers)
        api_key = self.config.resolve_api_key()
        if api_key:
            headers.setdefault("Authorization", f"Bearer {api_key}")
        self._client = httpx.AsyncClient(
            headers=headers,
            timeout=self.config.timeout_s,
            transport=httpx.AsyncHTTPTransport(retries=self.config.max_retries),
        )
        await super().open()

    async def close(self) -> None:
        """关闭 httpx 连接池；幂等。"""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        await super().close()

    def get_model(self, model_id: str) -> DashScopeRerankModel:
        """按 model_id 复用或创建模型句柄。"""
        model = self._handles.get(model_id)
        if model is None:
            model = DashScopeRerankModel(self.get_spec(model_id), self)
            self._handles[model_id] = model
        return model
