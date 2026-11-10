"""BGE embedding adapter（OpenAI-compatible /embeddings）。"""

from .openai import BgeEmbeddingProvider, OpenAIEmbeddingModel

__all__ = ["BgeEmbeddingProvider", "OpenAIEmbeddingModel"]
