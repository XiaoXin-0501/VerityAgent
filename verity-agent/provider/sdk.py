"""Official SDK clients and SDK error normalization.

Provider adapters use the vendor SDKs (``openai`` / ``anthropic`` / ``cohere``)
for HTTP, SSE, auth, timeouts and retries. This module only builds clients from
``ProviderConfig`` and maps SDK exceptions to ``ProviderError``; it deliberately
does not reimplement any wire protocol.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Mapping
from typing import Any, cast

import anthropic
import cohere
import openai

from config.provider import ProviderConfig
from exception.provider import ProviderError, ProviderErrorCode

__all__ = [
    "build_openai_client",
    "build_anthropic_client",
    "build_cohere_client",
    "close_cohere_client",
    "sdk_error",
    "split_sdk_kwargs",
    "sdk_dump",
]


def build_openai_client(config: ProviderConfig) -> openai.AsyncOpenAI:
    """Build an SDK client; ``base_url`` is optional for the official API."""
    return openai.AsyncOpenAI(
        api_key=config.resolve_api_key() or "EMPTY",
        base_url=config.base_url or None,
        timeout=config.timeout_s,
        max_retries=config.max_retries,
        default_headers=dict(config.headers),
    )


def build_anthropic_client(config: ProviderConfig) -> anthropic.AsyncAnthropic:
    """构建 Anthropic 客户端；base_url 去掉重复的 /v1。"""
    headers = dict(config.headers)
    version = config.metadata.get("anthropic_version")
    if version:
        headers.setdefault("anthropic-version", str(version))
    return anthropic.AsyncAnthropic(
        api_key=config.resolve_api_key() or "EMPTY",
        base_url=_base_without_v1(config.base_url),
        timeout=config.timeout_s,
        max_retries=config.max_retries,
        default_headers=headers,
    )


def build_cohere_client(config: ProviderConfig) -> cohere.AsyncClient:
    """构建 Cohere 客户端；base_url 去掉重复的 /v1。"""
    return cohere.AsyncClient(
        api_key=config.resolve_api_key() or "EMPTY",
        base_url=_base_without_v1(config.base_url),
        timeout=config.timeout_s,
    )


async def close_cohere_client(client: cohere.AsyncClient) -> None:
    """Close the underlying httpx client.

    Cohere SDK 5.x does not expose ``AsyncClient.close()``; the transport lives
    under ``_client_wrapper.httpx_client.httpx_client``. Keep the access here so
    adapters do not depend on private SDK details.
    """
    wrapper = getattr(client, "_client_wrapper", None)
    http = getattr(wrapper, "httpx_client", None)
    raw = getattr(http, "httpx_client", None)
    close = getattr(raw, "aclose", None)
    if callable(close):
        await cast(Awaitable[None], close())


def _base_without_v1(url: str) -> str | None:
    """Anthropic/Cohere SDK 自己追加 /v1，兼容旧配置里重复写 /v1 的情况。"""
    text = url.strip().rstrip("/")
    if not text:
        return None
    if text.endswith("/v1"):
        text = text[:-3].rstrip("/")
    return text or None


def sdk_error(
    exc: BaseException,
    *,
    provider: str,
    model: str | None = None,
) -> ProviderError:
    """Map OpenAI/Anthropic/Cohere SDK exceptions to ``ProviderError``."""
    if isinstance(exc, ProviderError):
        return exc

    status = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    retry_after = _retry_after(headers)

    code = _status_code(status)
    if code is None:
        name = type(exc).__name__.lower()
        if (
            isinstance(exc, asyncio.TimeoutError)
            or "timeout" in name
            or "deadline" in name
        ):
            code = ProviderErrorCode.TIMEOUT
        elif "responsevalidation" in name:
            code = ProviderErrorCode.RESPONSE_INVALID
        elif "connection" in name:
            code = ProviderErrorCode.CONNECTION
        elif "cancel" in name:
            code = ProviderErrorCode.CANCELLED
        elif (
            ("rate" in name and "limit" in name)
            or "toomanyrequests" in name
            or "throttl" in name
        ):
            code = ProviderErrorCode.RATE_LIMITED
        elif "gatewaytimeout" in name or "timeout" in name:
            code = ProviderErrorCode.TIMEOUT
        elif "serviceunavailable" in name:
            code = ProviderErrorCode.UPSTREAM
        elif "authentication" in name or "unauthorized" in name or "invalidtoken" in name:
            code = ProviderErrorCode.AUTHENTICATION
        elif "permission" in name or "forbidden" in name:
            code = ProviderErrorCode.PERMISSION
        elif "notfound" in name or "not_found" in name:
            code = ProviderErrorCode.MODEL_NOT_FOUND
        elif "badrequest" in name or "invalid" in name:
            code = ProviderErrorCode.INVALID_REQUEST
        else:
            code = ProviderErrorCode.INTERNAL

    provider_code = getattr(exc, "code", None) or getattr(exc, "type", None)
    message = getattr(exc, "message", None) or str(exc) or type(exc).__name__
    details: dict[str, Any] = {"error_type": type(exc).__name__}
    if status is not None:
        details["status_code"] = status
    return ProviderError(
        code,
        message[:500],
        provider=provider,
        model=model,
        retry_after_s=retry_after,
        provider_code=str(provider_code) if provider_code is not None else None,
        details=details,
        cause=type(exc).__name__,
    )


def split_sdk_kwargs(
    payload: Mapping[str, Any],
    allowed: frozenset[str],
) -> dict[str, Any]:
    """Split typed SDK kwargs from vendor extensions passed via ``extra_body``."""
    kwargs = {key: value for key, value in payload.items() if key in allowed}
    extra = {key: value for key, value in payload.items() if key not in allowed}
    if extra:
        merged = dict(kwargs.get("extra_body") or {})
        merged.update(extra)
        kwargs["extra_body"] = merged
    return kwargs


def sdk_dump(value: Any) -> Any:
    """Pydantic-like SDK object -> plain dict; already-plain values pass through."""
    dump = getattr(value, "model_dump", None)
    return dump() if callable(dump) else value


def _status_code(status: Any) -> ProviderErrorCode | None:
    """把 HTTP 状态码映射为统一 ProviderErrorCode。"""
    if not isinstance(status, int):
        return None
    if status in (401,):
        return ProviderErrorCode.AUTHENTICATION
    if status in (403,):
        return ProviderErrorCode.PERMISSION
    if status == 404:
        return ProviderErrorCode.MODEL_NOT_FOUND
    if status in (408, 504):
        return ProviderErrorCode.TIMEOUT
    if status == 429:
        return ProviderErrorCode.RATE_LIMITED
    if status >= 500:
        return ProviderErrorCode.UPSTREAM
    if status in (499,):
        return ProviderErrorCode.CANCELLED
    if 400 <= status < 500:
        return ProviderErrorCode.INVALID_REQUEST
    return None


def _retry_after(headers: Any) -> float | None:
    """从响应头读取 retry-after 秒数；非法值返回 None。"""
    value = None
    if isinstance(headers, Mapping):
        value = headers.get("retry-after") or headers.get("Retry-After")
    elif hasattr(headers, "get"):
        value = headers.get("retry-after") or headers.get("Retry-After")
    if not value:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
