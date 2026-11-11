"""Provider adapter 的 wire 映射测试（httpx.MockTransport，不访问网络）。"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import anthropic
import cohere
import httpx
import openai

_ROOT = Path(__file__).resolve().parents[1]
_AGENT_ROOT = _ROOT / "verity-agent"
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

from config.model import ModelConfig  # noqa: E402
from provider.embedding.types import EmbeddingRequest  # noqa: E402
from provider.factory import build_registry  # noqa: E402
from provider.llm.types import (  # noqa: E402
    ChatModelRequest,
    ModelMessage,
    ModelTextPart,
)
from provider.reranker.types import RerankDocument, RerankRequest  # noqa: E402
from provider.sdk import close_cohere_client  # noqa: E402


def _chat_model(
    *,
    model_id: str = "test-model",
    provider: str = "openai",
    api_style: str | None = None,
    capabilities: dict | None = None,
    defaults: dict | None = None,
    base_url: str = "https://provider.test/v1",
) -> ModelConfig:
    return ModelConfig(
        id=model_id,
        provider=provider,
        model="test-model",
        kind="chat",
        api_style=api_style,
        base_url=base_url,
        capabilities=capabilities or {},
        defaults=defaults or {},
    )


def _registry_with(model: ModelConfig):
    registry = build_registry([model])
    provider = registry.get(f"{model.provider}:{model.id}")
    return registry, provider


def _request() -> ChatModelRequest:
    return ChatModelRequest(
        model="test-model",
        messages=[ModelMessage(role="user", content=[ModelTextPart(text="hi")])],
    )


def _openai_client(handler) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(
        api_key="test-key",
        base_url="https://provider.test/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def _anthropic_client(handler) -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(
        api_key="test-key",
        base_url="https://provider.test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def _cohere_client(handler) -> cohere.AsyncClient:
    return cohere.AsyncClient(
        api_key="test-key",
        base_url="https://provider.test",
        httpx_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def test_openai_chat_complete() -> None:
    async def run() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/v1/chat/completions"
            body = json.loads(request.content)
            assert body["messages"][0]["content"] == "hi"
            return httpx.Response(
                200,
                json={
                    "id": "resp-1",
                    "model": "test-model",
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "hello",
                                "tool_calls": [
                                    {
                                        "id": "call-1",
                                        "type": "function",
                                        "function": {
                                            "name": "search_kb",
                                            "arguments": '{"query":"x"}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 2,
                        "total_tokens": 5,
                    },
                },
            )

        registry, provider = _registry_with(_chat_model())
        provider._client = _openai_client(handler)
        try:
            response = await registry.get_model_by_id("test-model").complete(_request())
        finally:
            await provider._client.close()
        assert response.finish_reason.value == "tool_calls"
        assert response.message.content[0].text == "hello"
        assert response.message.tool_calls[0].arguments == {"query": "x"}
        assert response.usage.total_tokens == 5

    asyncio.run(run())


def test_openai_chat_stream() -> None:
    async def run() -> None:
        chunks = [
            {"id": "resp", "choices": [{"delta": {"content": "he"}}]},
            {"id": "resp", "choices": [{"delta": {"content": "llo"}}]},
            {
                "id": "resp",
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            },
        ]
        body = "".join(f"data: {json.dumps(item)}\n\n" for item in chunks)
        body += "data: [DONE]\n\n"

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, text=body, headers={"content-type": "text/event-stream"}
            )

        registry, provider = _registry_with(_chat_model())
        provider._client = _openai_client(handler)
        try:
            events = [
                event
                async for event in registry.get_model_by_id("test-model").stream(
                    _request()
                )
            ]
        finally:
            await provider._client.close()
        assert "".join(
            event.text for event in events if type(event).__name__ == "ChatTextDelta"
        ) == "hello"
        assert events[-1].finish_reason.value == "stop"
        assert events[-1].usage.total_tokens == 3

    asyncio.run(run())


def test_anthropic_complete() -> None:
    async def run() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert request.url.path == "/v1/messages"
            assert body["messages"][0]["role"] == "user"
            return httpx.Response(
                200,
                json={
                    "id": "msg-1",
                    "content": [
                        {"type": "text", "text": "hello"},
                        {
                            "type": "tool_use",
                            "id": "tool-1",
                            "name": "search_kb",
                            "input": {"query": "x"},
                        },
                    ],
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 3, "output_tokens": 2},
                },
            )

        registry, provider = _registry_with(
            _chat_model(
                provider="anthropic",
                api_style="anthropic_messages",
                base_url="https://provider.test",
            )
        )
        provider._client = _anthropic_client(handler)
        try:
            response = await registry.get_model_by_id("test-model").complete(_request())
        finally:
            await provider._client.close()
        assert response.finish_reason.value == "tool_calls"
        assert response.message.tool_calls[0].arguments == {"query": "x"}
        assert response.usage.total_tokens == 5

    asyncio.run(run())


def test_embedding_complete() -> None:
    async def run() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/v1/embeddings"
            return httpx.Response(
                200,
                json={
                    "model": "emb",
                    "data": [
                        {"index": 0, "embedding": [0.1, 0.2]},
                        {"index": 1, "embedding": [0.3, 0.4]},
                    ],
                    "usage": {"prompt_tokens": 2, "total_tokens": 2},
                },
            )

        model = ModelConfig(
            id="emb",
            provider="openai",
            model="emb",
            kind="embedding",
            base_url="https://provider.test/v1",
            capabilities={"dimensions": 2},
        )
        registry, provider = _registry_with(model)
        provider._client = _openai_client(handler)
        try:
            result = await registry.get_model_by_id("emb").embed(
                EmbeddingRequest(model="emb", inputs=["a", "b"])
            )
        finally:
            await provider._client.close()
        assert result.dimensions == 2
        assert result.vectors[1] == [0.3, 0.4]

    asyncio.run(run())


def test_rerank_complete() -> None:
    async def run() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/v1/rerank"
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"index": 1, "relevance_score": 0.9},
                        {"index": 0, "relevance_score": 0.2},
                    ]
                },
            )

        model = ModelConfig(
            id="rr",
            provider="cohere",
            model="rr",
            kind="rerank",
            base_url="https://provider.test",
        )
        registry, provider = _registry_with(model)
        provider._client = _cohere_client(handler)
        try:
            result = await registry.get_model_by_id("rr").rerank(
                RerankRequest(
                    model="rr",
                    query="q",
                    documents=[
                        RerankDocument(id="a", text="a"),
                        RerankDocument(id="b", text="b"),
                    ],
                )
            )
        finally:
            await close_cohere_client(provider._client)
        assert [item.id for item in result.results] == ["b", "a"]

    asyncio.run(run())


def test_factory_builds_multiple_models() -> None:
    registry = build_registry(
        [
            _chat_model(model_id="chat-main"),
            _chat_model(
                model_id="chat-claude",
                provider="anthropic",
                api_style="anthropic_messages",
            ),
        ]
    )
    assert type(registry.get("openai:chat-main")).__name__ == "OpenAIChatProvider"
    assert type(registry.get("anthropic:chat-claude")).__name__ == "AnthropicChatProvider"
    assert registry.get_model_by_id("chat-main") is not None
    assert registry.get_model_by_id("chat-claude") is not None
