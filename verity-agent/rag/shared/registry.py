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
    """解析器注册表：按扩展名/MIME/doc_type 选择实现。"""

    def __init__(self) -> None:
        """初始化空注册表。"""
        self._items: dict[str, tuple[ParserSpec, Any]] = {}

    def register(self, spec: ParserSpec, parser: Any) -> None:
        """注册一个 parser；重复 id 视为配置冲突。"""
        if not spec.parser_id:
            raise RagError(RagErrorCode.INVALID_CONFIG, "parser_id 不能为空")
        if spec.parser_id in self._items:
            raise RagError(
                RagErrorCode.CONFLICT, f"parser 重复注册: {spec.parser_id}"
            )
        self._items[spec.parser_id] = (spec, parser)

    def get(self, parser_id: str) -> Any | None:
        """按 id 取 parser 实现；不存在返回 None。"""
        item = self._items.get(parser_id)
        return item[1] if item else None

    def list(self, doc_type: DocumentType | None = None) -> list[ParserSpec]:
        """列出已注册 spec，可按 doc_type 过滤。"""
        specs = [spec for spec, _ in self._items.values()]
        if doc_type is not None:
            specs = [spec for spec in specs if doc_type in spec.doc_types]
        return specs

    def detect_doc_type(
        self,
        extension: str,
        mime_type: str = "",
    ) -> DocumentType:
        """按已注册 parser 的 extensions/mime_types 推断文档类型。

        扩展名和 MIME 只在注册表里维护一份，避免 ``parsers/base.py`` 再维护
        一套手工映射导致两边漂移。扩展名命中优先于 MIME；同分取高优先级。
        """
        ext = (extension or "").lower().strip()
        if ext and not ext.startswith("."):
            ext = f".{ext}"
        mime = (mime_type or "").lower().strip()
        best: tuple[int, int, str, DocumentType] | None = None
        for spec, _impl in self._items.values():
            if not spec.enabled:
                continue
            score = 0
            if ext and ext in {item.lower() for item in spec.extensions}:
                score += 2
            if mime and mime in {item.lower() for item in spec.mime_types}:
                score += 1
            if score == 0:
                continue
            # 一个 parser 可能声明多个 doc_type；按稳定顺序取第一个，
            # 避免集合迭代顺序影响结果。
            for doc_type in sorted(spec.doc_types, key=lambda item: item.value):
                candidate = (score, spec.priority, doc_type.value, doc_type)
                if best is None or candidate[:3] > best[:3]:
                    best = candidate
        return best[3] if best is not None else DocumentType.UNKNOWN

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
            """按 MIME、扩展名、doc_type 的顺序为候选 parser 打分。"""
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
    """处理器元数据；v1 只保留协议，按 block/doc 类型选择实现。"""

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
        """保存处理器标识、适用类型与优先级。"""
        self.processor_id = processor_id
        self.block_types = frozenset(block_types)
        self.doc_types = frozenset(doc_types)
        self.priority = priority
        self.enabled = enabled
        self.options = options or {}


class ProcessorRegistry:
    """处理器注册表；同一 block_type 取优先级最高且 doc_type 匹配的实现。"""

    def __init__(self) -> None:
        """初始化空注册表。"""
        self._items: dict[str, tuple[ProcessorSpec, Any]] = {}

    def register(self, spec: ProcessorSpec, processor: Any) -> None:
        """注册处理器；重复 id 视为配置冲突。"""
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
        """选择处理器；没有匹配返回 None。"""
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
        """列出全部已注册处理器 spec。"""
        return [spec for spec, _ in self._items.values()]


class ChunkerRegistry:
    """切片器注册表；按 ``(doc_type, block_type)`` 选择策略。"""

    def __init__(self) -> None:
        """初始化空注册表。"""
        self._items: dict[str, tuple[ChunkingSpec, Any]] = {}

    def register(self, spec: ChunkingSpec, chunker: Any) -> None:
        """注册切片器；重复 strategy_id 视为配置冲突。"""
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
        """选择切片器；没有精确匹配时返回 None，由调用方回退文本策略。"""
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
        """列出全部已注册切片 spec。"""
        return [spec for spec, _ in self._items.values()]
