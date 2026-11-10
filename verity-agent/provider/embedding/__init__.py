"""Embedding provider 抽象与 wire 模型。"""

from .base import EmbeddingModel, EmbeddingProvider
from .openai import BgeEmbeddingProvider, OpenAIEmbeddingModel, OpenAIEmbeddingProvider
from .types import EmbeddingRequest, EmbeddingResponse, EmbeddingUsage

__all__ = [
    "EmbeddingModel",
    "EmbeddingProvider",
    "OpenAIEmbeddingModel",
    "OpenAIEmbeddingProvider",
    "BgeEmbeddingProvider",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "EmbeddingUsage",
]
