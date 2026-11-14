"""跨模块共享的领域枚举。

只放 Agent / RAG 等模块都需要、又不应由某一方拥有的枚举，避免公共 DTO
反向依赖实现包（例如 ``agent.public`` 直接 import ``rag.shared``）。
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["RAGMode"]


class RAGMode(StrEnum):
    """RAG 使用模式；与 RAG 检索契约保持一致。"""

    TOOL = "tool"
    INJECTED = "injected"
    OFF = "off"
