"""知识库片段冲突处理：根据冲突类型、文档元数据与邻居片段给出结论。"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any
from uuid import UUID

from prompt import PromptId, PromptRegistry, get_default_prompt_registry
from provider.enums import ModelRole
from provider.llm.helpers import response_text
from provider.llm.types import (
    ChatModelRequest,
    ModelGenerationOptions,
    ModelMessage,
    ModelTextPart,
)
from store.pg.models.enums import DocumentStatus
from store.pg.repositories import DocumentRepository
from store.weaviate import WeaviateStore
from utils.serialization import parse_json_object
from utils.time import parse_datetime

__all__ = ["ChunkConflictService"]

logger = logging.getLogger(__name__)

# 返回给模型的属性；与 DocChunk 检索契约保持一致。
_CHUNK_PROPERTIES = [
    "kb_id",
    "doc_id",
    "doc_type",
    "heading",
    "chunk_index",
    "content_zh",
    "content_en",
    "quality_score",
    "index_version",
    "created_at",
]

_MAX_CHUNKS = 8
_MAX_NEIGHBORS = 3


class ChunkConflictService:
    """解决知识库 chunk 之间的冲突。

    - factual：数值/日期/定义等硬冲突，按文档更新时间、入库版本、质量分判断；
    - viewpoint：观点/经验差异，补充邻居片段并综合回答；
    - ambiguous：信息不足时明确说明需要澄清。
    """

    def __init__(
        self,
        *,
        model: Any,
        model_id: str,
        store: WeaviateStore | None = None,
        documents: DocumentRepository | None = None,
        timeout_s: float = 20.0,
        max_output_tokens: int = 1024,
        neighbor_window: int = 1,
        prompts: PromptRegistry | None = None,
        kb_catalog: Any | None = None,
    ) -> None:
        """保存模型、Weaviate/文档仓库与冲突处理策略。

        ``kb_catalog`` 是知识库可读目录（实现 ``can_read``）；为空时退回
        ``kb:read:<id>`` 约定码，仅供离线/单测使用。生产必须注入，保证
        自定义 ``required_permission_code`` 与检索流水线一致。
        """
        self.model = model
        self.model_id = model_id
        self.store = store or WeaviateStore()
        self.documents = documents or DocumentRepository()
        self.kb_catalog = kb_catalog
        self.timeout_s = max(1.0, float(timeout_s))
        self.max_output_tokens = max(64, int(max_output_tokens))
        self.neighbor_window = max(0, int(neighbor_window))
        self.prompts = prompts or get_default_prompt_registry()
        self.system_prompt = self.prompts.render_text(
            PromptId.RAG_CHUNK_CONFLICT_SYSTEM
        )

    async def resolve(
        self,
        *,
        user_id: UUID,
        permissions: frozenset[str],
        question: str,
        conflict_type: str,
        chunk_ids: list[str],
    ) -> dict[str, Any]:
        """返回 ``answer/resolution/preferred_chunk_ids/conflict_summary``。"""
        unique_ids = list(dict.fromkeys(str(item) for item in chunk_ids if item))[
            :_MAX_CHUNKS
        ]
        if len(unique_ids) < 2:
            return {
                "answer": "无法定位到足够的冲突片段，请提供至少两个片段。",
                "resolution": "insufficient_chunks",
                "preferred_chunk_ids": [],
                "conflict_summary": "",
            }
        chunks = await self._fetch_chunks(
            unique_ids,
            user_id=user_id,
            permissions=permissions,
        )
        if len(chunks) < 2:
            return {
                "answer": "部分片段不存在或当前无权访问，无法完成冲突处理。",
                "resolution": "insufficient_chunks",
                "preferred_chunk_ids": [item["chunk_id"] for item in chunks],
                "conflict_summary": "",
            }
        for chunk in chunks:
            chunk["neighbors"] = await self._fetch_neighbors(chunk)
        payload = {
            "question": question,
            "conflict_type": conflict_type,
            "chunks": chunks,
        }
        try:
            result = await self._call_model(payload)
        except Exception:  # noqa: BLE001 - 冲突处理失败降级为确定性摘要
            logger.warning("chunk 冲突 LLM 处理失败，使用降级摘要", exc_info=True)
            result = self._fallback(chunks, conflict_type)
        allowed_ids = {
            str(item.get("chunk_id") or "").strip().lower(): str(
                item.get("chunk_id") or ""
            )
            for item in chunks
        }
        preferred: list[str] = []
        for item in result.get("preferred_chunk_ids") or []:
            canonical = allowed_ids.get(str(item).strip().lower())
            if canonical and canonical not in preferred:
                preferred.append(canonical)
        result["preferred_chunk_ids"] = preferred
        return result

    async def _call_model(self, payload: dict[str, Any]) -> dict[str, Any]:
        """调用冲突处理模型并解析结构化结果。"""
        request = ChatModelRequest(
            model=self.model_id,
            messages=[
                ModelMessage(
                    role=ModelRole.SYSTEM,
                    content=[ModelTextPart(text=self.system_prompt)],
                ),
                ModelMessage(
                    role=ModelRole.USER,
                    content=[
                        ModelTextPart(
                            text=json.dumps(
                                payload,
                                ensure_ascii=False,
                                default=str,
                            )
                        )
                    ],
                ),
            ],
            options=ModelGenerationOptions(
                temperature=0.0,
                max_output_tokens=self.max_output_tokens,
            ),
        )
        response = await asyncio.wait_for(
            self.model.complete(request),
            timeout=self.timeout_s,
        )
        data = parse_json_object(response_text(response))
        answer = str(data.get("answer") or "").strip()
        if not answer:
            raise ValueError("冲突处理结果缺少 answer")
        preferred = data.get("preferred_chunk_ids")
        return {
            "answer": answer[:2000],
            "resolution": str(data.get("resolution") or "synthesized")[:60],
            "preferred_chunk_ids": [
                str(item) for item in preferred or [] if str(item).strip()
            ][:4],
            "conflict_summary": str(data.get("conflict_summary") or "")[:500],
        }

    async def _fetch_chunks(
        self,
        chunk_ids: list[str],
        *,
        user_id: UUID,
        permissions: frozenset[str],
    ) -> list[dict[str, Any]]:
        """按 id 读取 chunk 与文档元数据；过滤无权 KB 与非 completed 文档。"""
        collection = self.store.doc_chunks
        results: list[dict[str, Any]] = []
        # 同一文档可能贡献多个冲突片段，文档元数据只查一次。
        doc_cache: dict[str, Any | None] = {}
        for chunk_id in chunk_ids:
            try:
                obj = await self.store.maybe_to_thread(
                    collection.query.fetch_object_by_id,
                    chunk_id,
                    return_properties=_CHUNK_PROPERTIES,
                )
            except Exception:  # noqa: BLE001 - 单个坏 id 不影响其它片段
                obj = None
            if obj is None:
                continue
            props = dict(getattr(obj, "properties", {}) or {})
            kb_id = str(props.get("kb_id") or "")
            if not await self._read_allowed(
                user_id=user_id,
                permissions=permissions,
                kb_id=kb_id,
            ):
                continue
            doc_id = str(props.get("doc_id") or "")
            if doc_id not in doc_cache:
                doc_cache[doc_id] = await self._get_document(doc_id)
            doc = doc_cache[doc_id]
            # 检索流水线只返回 completed 文档；按 id 直读时必须保持同一安全
            # 边界，避免删除/重切片期间的旧 chunk 被冲突工具重新带回答。
            status = getattr(doc, "status", None)
            status_value = getattr(status, "value", status)
            if doc is None or status_value != DocumentStatus.COMPLETED.value:
                continue
            results.append(
                {
                    "chunk_id": str(getattr(obj, "uuid", chunk_id)),
                    "doc_id": doc_id,
                    "doc_name": getattr(doc, "original_filename", None) or doc_id,
                    "doc_description": getattr(doc, "description", None),
                    "doc_created_at": _iso(getattr(doc, "created_at", None)),
                    "doc_updated_at": _iso(getattr(doc, "updated_at", None)),
                    "index_version": props.get("index_version"),
                    "doc_type": props.get("doc_type"),
                    "heading": props.get("heading"),
                    "chunk_index": props.get("chunk_index"),
                    "quality_score": props.get("quality_score"),
                    "created_at": _iso(props.get("created_at")),
                    "content": props.get("content_zh")
                    or props.get("content_en")
                    or "",
                }
            )
        return results

    async def _read_allowed(
        self,
        *,
        user_id: UUID,
        permissions: frozenset[str],
        kb_id: str,
    ) -> bool:
        """按 KB 的实际权限事实源判断可读性。

        生产注入 ``kb_catalog.can_read``，与检索流水线共用
        ``knowledge_bases.required_permission_code``；未注入时退回约定码，
        仅用于离线/单测。权限源异常按拒绝处理（fail-closed）。
        """
        if not kb_id:
            return False
        checker = getattr(self.kb_catalog, "can_read", None)
        if not callable(checker):
            return _can_read(permissions, kb_id)
        try:
            return bool(
                await checker(
                    user_id=user_id,
                    permissions=permissions,
                    kb_id=UUID(kb_id),
                )
            )
        except (TypeError, ValueError):
            return False
        except Exception:  # noqa: BLE001 - 权限源失败不能放行
            logger.warning("知识库权限校验失败，按不可读处理: %s", kb_id, exc_info=True)
            return False

    async def _get_document(self, doc_id: str) -> Any | None:
        """读取未删除文档元数据；失败不阻断冲突处理。"""
        if not doc_id:
            return None
        try:
            return await self.documents.get(UUID(doc_id), include_deleted=False)
        except Exception:  # noqa: BLE001 - 元数据缺失按不可用跳过
            return None

    async def _fetch_neighbors(self, chunk: dict[str, Any]) -> list[dict[str, str]]:
        """取同文档相邻 chunk，供观点冲突时补充上下文。"""
        if self.neighbor_window <= 0 or not chunk.get("doc_id"):
            return []
        index = chunk.get("chunk_index")
        if not isinstance(index, int):
            return []
        try:
            from weaviate.classes.query import Filter

            collection = self.store.doc_chunks
            # 同时限定文档、索引版本和 chunk_index 区间：文档重灌后同一
            # doc_id 可能残留多个版本的 chunk，不能把旧版邻居当上下文。
            where = Filter.by_property("doc_id").equal(str(chunk["doc_id"]))
            version = chunk.get("index_version")
            if version is not None:
                try:
                    where = where & Filter.by_property("index_version").equal(
                        int(version)
                    )
                except (TypeError, ValueError):
                    pass
            where = where & Filter.by_property("chunk_index").greater_or_equal(
                index - self.neighbor_window
            )
            where = where & Filter.by_property("chunk_index").less_or_equal(
                index + self.neighbor_window
            )
            response = await self.store.maybe_to_thread(
                collection.query.fetch_objects,
                filters=where,
                limit=max(4, self.neighbor_window * 2 + 2),
                return_properties=_CHUNK_PROPERTIES,
            )
        except Exception:  # noqa: BLE001 - 邻居失败不影响主片段
            return []
        neighbors: list[dict[str, str]] = []
        for obj in getattr(response, "objects", []) or []:
            props = dict(getattr(obj, "properties", {}) or {})
            item_index = props.get("chunk_index")
            if not isinstance(item_index, int):
                continue
            if not (
                index - self.neighbor_window
                <= item_index
                <= index + self.neighbor_window
            ):
                continue
            if item_index == index:
                continue
            content = props.get("content_zh") or props.get("content_en") or ""
            if not content:
                continue
            neighbors.append(
                {
                    "chunk_id": str(getattr(obj, "uuid", "")),
                    "chunk_index": str(item_index),
                    "heading": str(props.get("heading") or ""),
                    "content": str(content)[:1200],
                }
            )
        neighbors.sort(key=lambda item: int(item["chunk_index"]))
        return neighbors[:_MAX_NEIGHBORS]

    @staticmethod
    def _fallback(
        chunks: list[dict[str, Any]],
        conflict_type: str,
    ) -> dict[str, Any]:
        """LLM 不可用时的确定性摘要：列出冲突，并按类型给出保守结论。"""
        def rank(item: dict[str, Any]) -> tuple:
            stamp = item.get("doc_updated_at") or item.get("doc_created_at") or ""
            quality = float(item.get("quality_score") or 0.0)
            return (stamp, quality)

        preferred = max(chunks, key=rank)
        lines = [
            "检测到知识库文档片段冲突：",
        ]
        for item in chunks:
            excerpt = str(item.get("content") or "").strip().replace("\n", " ")[:180]
            lines.append(f"- 《{item.get('doc_name')}》：{excerpt}")
        if conflict_type == "ambiguous":
            lines.append(
                "现有片段信息不足以可靠判断哪一方正确；请补充适用范围、"
                "数据口径或时间信息后再确认。"
            )
            resolution = "clarify"
            preferred_ids: list[str] = []
        elif conflict_type == "viewpoint":
            lines.append(
                "两种说法可能来自不同观点或适用背景，建议分别说明其来源，"
                "不要在没有依据时强行合并。"
            )
            resolution = "synthesized"
            preferred_ids = []
        else:
            lines.append(
                f"按文档更新时间/质量分优先，建议以《{preferred.get('doc_name')}》"
                f"的记载为准；如冲突仍影响结论，请提供更多上下文以便进一步核实。"
            )
            resolution = "factual_newer"
            preferred_ids = [str(preferred.get("chunk_id") or "")]
        return {
            "answer": "\n".join(lines),
            "resolution": resolution,
            "preferred_chunk_ids": preferred_ids,
            "conflict_summary": "文档片段之间存在冲突，已按元数据降级判断",
        }


def _can_read(permissions: frozenset[str], kb_id: str) -> bool:
    """检查 KB 读取权限；空 kb_id 视为无效。"""
    if not kb_id:
        return False
    return any(
        code in permissions
        for code in ("kb:read", "kb:read:*", f"kb:read:{kb_id}")
    )


def _iso(value: Any) -> str | None:
    """datetime -> ISO 字符串；其它值原样转字符串。"""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    text = str(value).strip()
    if not text:
        return None
    try:
        return parse_datetime(text).isoformat()
    except ValueError:
        return text
