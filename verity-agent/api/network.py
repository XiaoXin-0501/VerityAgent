"""HTTP 网络层公共能力（放在 api 模块）。

具体厂商 adapter 只负责 wire 映射；超时、transport retry、认证头、SSE 解析
和 HTTP 错误翻译在这里统一处理。
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from typing import Any

import httpx

from config.provider import ProviderConfig
from exception.provider import ProviderError, ProviderErrorCode

__all__ = ["HttpProviderMixin", "translate_http_error", "translate_transport_error"]


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _error_message(response: httpx.Response) -> tuple[str, str | None, dict[str, Any]]:
    """提取厂商错误 message / code / details。"""
    try:
        payload = response.json()
    except ValueError:
        return response.text[:500] or response.reason_phrase, None, {}
    if not isinstance(payload, Mapping):
        return str(payload)[:500], None, {}
    error = payload.get("error")
    if isinstance(error, Mapping):
        message = str(error.get("message") or error.get("detail") or response.reason_phrase)
        code = error.get("code") or error.get("type")
        return message, str(code) if code is not None else None, dict(error)
    message = payload.get("message") or payload.get("detail") or response.reason_phrase
    return str(message), None, dict(payload)


def translate_http_error(
    response: httpx.Response,
    *,
    provider: str,
    model: str | None = None,
) -> ProviderError:
    status = response.status_code
    message, provider_code, details = _error_message(response)
    retry_after = _retry_after(response)
    if status in (401, 403):
        code = (
            ProviderErrorCode.AUTHENTICATION
            if status == 401
            else ProviderErrorCode.PERMISSION
        )
    elif status == 404:
        code = ProviderErrorCode.MODEL_NOT_FOUND
    elif status == 429:
        code = ProviderErrorCode.RATE_LIMITED
    elif status in (408, 504):
        code = ProviderErrorCode.TIMEOUT
    elif status >= 500:
        code = ProviderErrorCode.UPSTREAM
    else:
        code = ProviderErrorCode.INVALID_REQUEST
    return ProviderError(
        code,
        message,
        provider=provider,
        model=model,
        retry_after_s=retry_after,
        provider_code=provider_code,
        details={"status_code": status, **details},
    )


def translate_transport_error(
    exc: Exception,
    *,
    provider: str,
    model: str | None = None,
) -> ProviderError:
    if isinstance(exc, ProviderError):
        return exc
    if isinstance(exc, httpx.TimeoutException):
        code = ProviderErrorCode.TIMEOUT
    elif isinstance(exc, httpx.TransportError):
        code = ProviderErrorCode.CONNECTION
    else:
        code = ProviderErrorCode.INTERNAL
    return ProviderError(
        code,
        f"Provider 请求失败: {type(exc).__name__}",
        provider=provider,
        model=model,
        cause=str(exc),
    )


class HttpProviderMixin:
    """给 provider 提供 AsyncClient 与重试/流式读帧。"""

    config: ProviderConfig

    def _init_http(self) -> None:
        self._http: httpx.AsyncClient | None = None

    @property
    def http(self) -> httpx.AsyncClient:
        if self._http is None:
            raise ProviderError(
                ProviderErrorCode.UNAVAILABLE,
                "Provider HTTP 客户端尚未初始化",
                provider=self.config.name,
            )
        return self._http

    def auth_headers(self) -> dict[str, str]:
        # 默认 Bearer；Anthropic 等 provider 可覆盖
        api_key = self.config.resolve_api_key()
        return {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def default_headers(self) -> dict[str, str]:
        return dict(self.config.headers)

    async def open_http(self) -> None:
        if not self.config.enabled:
            return
        if not self.config.base_url.strip():
            raise ProviderError(
                ProviderErrorCode.UNAVAILABLE,
                f"Provider {self.config.name} 未配置 base_url",
                provider=self.config.name,
            )
        headers = {"Content-Type": "application/json", **self.default_headers()}
        headers.update(self.auth_headers())
        self._http = httpx.AsyncClient(
            base_url=self.config.base_url.rstrip("/") + "/",
            timeout=httpx.Timeout(self.config.timeout_s),
            headers=headers,
        )

    async def close_http(self) -> None:
        client, self._http = self._http, None
        if client is not None:
            await client.aclose()

    async def request_json(
        self,
        method: str,
        path: str,
        *,
        payload: Mapping[str, Any] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        last_error: ProviderError | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = await self.http.request(method, path, json=payload)
            except Exception as exc:
                last_error = translate_transport_error(
                    exc, provider=self.config.name, model=model
                )
                if last_error.retryable_value and attempt < self.config.max_retries:
                    await asyncio.sleep(
                        self.config.retry_backoff_ms / 1000 * (2**attempt)
                    )
                    continue
                raise last_error from exc

            if response.status_code >= 400:
                error = translate_http_error(
                    response, provider=self.config.name, model=model
                )
                if error.retryable_value and attempt < self.config.max_retries:
                    last_error = error
                    delay = error.retry_after_s or (
                        self.config.retry_backoff_ms / 1000 * (2**attempt)
                    )
                    await asyncio.sleep(delay)
                    continue
                raise error

            try:
                data = response.json()
            except ValueError as exc:
                raise ProviderError(
                    ProviderErrorCode.RESPONSE_INVALID,
                    "Provider 返回的不是合法 JSON",
                    provider=self.config.name,
                    model=model,
                    cause=str(exc),
                ) from exc
            if not isinstance(data, dict):
                raise ProviderError(
                    ProviderErrorCode.RESPONSE_INVALID,
                    "Provider 返回的 JSON 顶层不是对象",
                    provider=self.config.name,
                    model=model,
                )
            return data
        raise last_error or ProviderError(
            ProviderErrorCode.INTERNAL,
            "Provider 请求失败",
            provider=self.config.name,
            model=model,
        )

    async def stream_sse(
        self,
        path: str,
        *,
        payload: Mapping[str, Any],
        model: str | None = None,
    ) -> AsyncIterator[tuple[str, str]]:
        """流式读取 ``(event_name, data_text)``。

        只在**尚未产生任何事件**时重试 transport；一旦已输出，不再重试，
        由 LLM 层按 partial 处理。
        """
        emitted = False
        attempt = 0
        while True:
            try:
                async with self.http.stream("POST", path, json=payload) as response:
                    if response.status_code >= 400:
                        await response.aread()
                        raise translate_http_error(
                            response, provider=self.config.name, model=model
                        )
                    event_name = ""
                    data_lines: list[str] = []
                    async for line in response.aiter_lines():
                        if line == "":
                            if data_lines:
                                emitted = True
                                yield event_name, "\n".join(data_lines)
                            event_name = ""
                            data_lines = []
                            continue
                        if line.startswith(":"):
                            continue
                        if line.startswith("event:"):
                            event_name = line[6:].strip()
                        elif line.startswith("data:"):
                            data_lines.append(line[5:].lstrip())
                    if data_lines:
                        emitted = True
                        yield event_name, "\n".join(data_lines)
                    return
            except ProviderError:
                raise
            except Exception as exc:
                error = translate_transport_error(
                    exc, provider=self.config.name, model=model
                )
                if (
                    not emitted
                    and error.retryable_value
                    and attempt < self.config.max_retries
                ):
                    attempt += 1
                    await asyncio.sleep(
                        self.config.retry_backoff_ms / 1000 * (2**attempt)
                    )
                    continue
                raise error from exc
