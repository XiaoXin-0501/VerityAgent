"""Parser / Processor / Chunker 注册表。

实现按文档类型/扩展名/块类型选择实现；业务层只依赖注册表，不直接 import
具体 parser/processor/chunker。
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from exception import RagError, RagErrorCode

from .enums import BlockType, DocumentType
from .models import ChunkingSpec, DocumentProfile, DocumentSource, ParserSpec

__all__ = ["ParserRegistry", "ProcessorSpec", "ProcessorRegistry", "ChunkerRegistry"]


class ParserRegistry:
    def __init__(self) -> None:
        self._items: dict[str, tuple[ParserSpec, Any]] = {}

    def register(self, spec: ParserSpec, parser: Any) -> None:
        if not spec.parser_id:
            raise RagError(RagErrorCode.INVALID_CONFIG, "parser_id 不能为空")
        if spec.parser_id in self._items:
            raise RagError(
                RagErrorCode.CONFLICT, f"parser 重复注册: {spec.parser_id}"
            )
        self._items[spec.parser_id] = (spec, parser)

    def get(self, parser_id: str) -> Any | None:
        item = self._items.get(parser_id)
        return item[1] if item else None

    def list(self, doc_type: DocumentType | None = None) -> list[ParserSpec]:
        specs = [spec for spec, _ in self._items.values()]
        if doc_type is not None:
            specs = [spec for spec in specs if doc_type in spec.doc_types]
        return specs

    def resolve(
        self,
        source: DocumentSource,
        profile: DocumentProfile | None = None,
        *,
        parser_hint: str | None = None,
    ) -> tuple[ParserSpec, Any]:
        """选择顺序：hint → MIME → 扩展名 → doc_type → 任意可用 parser。"""
        if parser_hint:
            item = self._items.get(parser_hint)
            if item is None:
                raise RagError(
                    RagErrorCode.INVALID_REQUEST,
                    f"指定的 parser 不存在: {parser_hint}",
                )
            return item

        doc_type = profile.doc_type if profile else DocumentType.UNKNOWN
        extension = source.extension.lower()
        if extension and not extension.startswith("."):
            extension = f".{extension}"
        mime_type = source.mime_type.lower()

        candidates = [
            (spec, impl)
            for spec, impl in self._items.values()
            if spec.enabled
        ]
        if not candidates:
            raise RagError(RagErrorCode.INVALID_CONFIG, "没有可用的 parser")

        def score(spec: ParserSpec) -> int:
            value = 0
            if mime_type and mime_type in {item.lower() for item in spec.mime_types}:
                value += 100
            if extension and extension in {item.lower() for item in spec.extensions}:
                value += 50
            if doc_type in spec.doc_types:
                value += 20
            if value == 0 and DocumentType.UNKNOWN in spec.doc_types:
                value += 1
            return value

        scored = [(score(spec), spec, impl) for spec, impl in candidates]
        scored = [item for item in scored if item[0] > 0]
        if not scored:
            # 文档要求：未知类型 fallback 到 text parser
            fallback = self._items.get("text")
            if fallback is not None and fallback[0].enabled:
                return fallback
            raise RagError(
                RagErrorCode.INVALID_REQUEST,
                f"没有匹配的 parser: {source.filename}",
            )
        scored.sort(key=lambda item: (-item[0], -item[1].priority, item[1].parser_id))
        _, spec, impl = scored[0]
        return spec, impl


class ProcessorSpec:
    def __init__(
        self,
        processor_id: str,
        *,
        block_types: Iterable[BlockType] = (),
        doc_types: Iterable[DocumentType] = (),
        priority: int = 100,
        enabled: bool = True,
        options: dict[str, Any] | None = None,
    ) -> None:
        self.processor_id = processor_id
        self.block_types = frozenset(block_types)
        self.doc_types = frozenset(doc_types)
        self.priority = priority
        self.enabled = enabled
        self.options = options or {}


class ProcessorRegistry:
    def __init__(self) -> None:
        self._items: dict[str, tuple[ProcessorSpec, Any]] = {}

    def register(self, spec: ProcessorSpec, processor: Any) -> None:
        if spec.processor_id in self._items:
            raise RagError(
                RagErrorCode.CONFLICT,
                f"processor 重复注册: {spec.processor_id}",
            )
        self._items[spec.processor_id] = (spec, processor)

    def resolve(
        self,
        block_type: BlockType,
        doc_type: DocumentType = DocumentType.UNKNOWN,
    ) -> tuple[ProcessorSpec, Any] | None:
        candidates = [
            (spec, impl)
            for spec, impl in self._items.values()
            if spec.enabled and block_type in spec.block_types
        ]
        if not candidates:
            return None
        candidates.sort(
            key=lambda item: (
                doc_type not in item[0].doc_types,
                -item[0].priority,
                item[0].processor_id,
            )
        )
        return candidates[0]

    def list(self) -> list[ProcessorSpec]:
        return [spec for spec, _ in self._items.values()]


class ChunkerRegistry:
    def __init__(self) -> None:
        self._items: dict[str, tuple[ChunkingSpec, Any]] = {}

    def register(self, spec: ChunkingSpec, chunker: Any) -> None:
        if spec.strategy_id in self._items:
            raise RagError(
                RagErrorCode.CONFLICT,
                f"chunker 重复注册: {spec.strategy_id}",
            )
        self._items[spec.strategy_id] = (spec, chunker)

    def resolve(
        self,
        doc_type: DocumentType,
        block_type: BlockType,
        *,
        profile: str | None = None,
    ) -> tuple[ChunkingSpec, Any] | None:
        candidates = [
            (spec, impl)
            for spec, impl in self._items.values()
            if spec.enabled and block_type in spec.block_types
        ]
        if not candidates:
            return None
        candidates.sort(
            key=lambda item: (
                doc_type not in item[0].doc_types,
                -item[0].priority,
                item[0].strategy_id,
            )
        )
        return candidates[0]

    def list(self) -> list[ChunkingSpec]:
        return [spec for spec, _ in self._items.values()]
