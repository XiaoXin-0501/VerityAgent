"""RAG 索引策略资源：质量扣分、描述标题词表、阶段进度。

这些值属于**索引策略/语言数据**，不是算法实现；集中放在
``config/rag_policy.json``，由本模块加载、校验并提供
``IndexerPolicy`` 只读对象。业务代码只消费策略对象，不再各自维护常量表。

- `QualityPenaltyPolicy`：质量标记 → 扣分；
- `DescriptionPolicy`：可直接作为文档描述的可靠标题词表；
- `AssociationPolicy`：代码/表格与相邻描述文本的关联提示词；
- `GraphPolicy`：通用关键词停用词；
- `StageProgressPolicy`：入库/删除阶段 → 进度（只用于 job view / PG 投影）。

加载期 fail-fast：未知枚举值、越界权重、空标题都会抛
``RagError(INVALID_CONFIG)``，避免坏策略静默生效。
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from config.base import SOURCE_ROOT
from exception import RagError, RagErrorCode
from rag.shared.enums import IndexingStage, QualityFlag

__all__ = [
    "BUILTIN_INDEXER_POLICY_PATH",
    "AssociationPolicy",
    "DescriptionPolicy",
    "GraphPolicy",
    "IndexerPolicy",
    "QualityPenaltyPolicy",
    "StageProgressPolicy",
    "get_indexer_policy",
    "load_indexer_policy",
    "normalize_heading_key",
]

# 默认策略文件放在 config 包，和 IndexerConfig 一起管理；
# 部署可通过 IndexerConfig.indexer_policy_path 指向外部文件。
BUILTIN_INDEXER_POLICY_PATH = SOURCE_ROOT / "config" / "rag_policy.json"

# 标题归一化：去空白与常见分隔标点，英文小写。
# JSON 中的可靠标题也走同一函数，保证词表与运行时判断口径一致。
_HEADING_PUNCTUATION = re.compile(r"[\s:：\-—_.]+")


def normalize_heading_key(text: str) -> str:
    """把标题归一成可比较的 key（去空白/标点、英文小写）。"""
    return _HEADING_PUNCTUATION.sub("", text or "").lower()


class _PolicyModel(BaseModel):
    """策略模型基类：拒绝未知字段，加载后只读。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
    )


class QualityPenaltyPolicy(_PolicyModel):
    """质量标记扣分表。"""

    # QualityFlag → 扣分；未声明的 flag 按 0 分处理（向前兼容）
    penalties: dict[QualityFlag, float] = Field(default_factory=dict)

    @field_validator("penalties")
    @classmethod
    def _check_penalties(
        cls, value: dict[QualityFlag, float]
    ) -> dict[QualityFlag, float]:
        """每个扣分必须在 [0,1]，否则一个标记就会把分数打到 0。"""
        for flag, penalty in value.items():
            if not 0.0 <= float(penalty) <= 1.0:
                raise ValueError(f"{flag} 的扣分必须在 0~1: {penalty}")
        return {flag: float(penalty) for flag, penalty in value.items()}

    def score(self, flags: set[QualityFlag]) -> float:
        """把质量标记折算成 0~1 分：``clamp(1 - Σ扣分, 0, 1)``。

        多个标记叠加扣分，不取最大值；结果保留 4 位小数，保证稳定。
        """
        total = sum(self.penalties.get(flag, 0.0) for flag in flags)
        return round(max(0.0, min(1.0, 1.0 - total)), 4)


class DescriptionPolicy(_PolicyModel):
    """文档描述提取的标题词表。"""

    # 可直接采信为文档描述的“文档级”标题；加载时统一归一化
    reliable_headings: frozenset[str] = Field(default_factory=frozenset)

    @field_validator("reliable_headings")
    @classmethod
    def _normalize_headings(
        cls, value: frozenset[str]
    ) -> frozenset[str]:
        """统一归一化，拒绝空标题。"""
        normalized = frozenset(normalize_heading_key(item) for item in value)
        if "" in normalized:
            raise ValueError("reliable_headings 不能包含空标题")
        return normalized


class AssociationPolicy(_PolicyModel):
    """代码/表格与相邻描述文本的关联提示词。"""

    # 前导说明通常以这些词结尾（“如下：”“下表列出……”）
    lead_in_hints: frozenset[str] = Field(default_factory=frozenset)
    # 后续说明通常以这些词开头（“上述代码……”“该表……”）
    follow_up_hints: frozenset[str] = Field(default_factory=frozenset)

    @field_validator("lead_in_hints", "follow_up_hints")
    @classmethod
    def _check_hints(cls, value: frozenset[str]) -> frozenset[str]:
        """提示词不能为空，否则关联逻辑会退化成猜测。"""
        items = [item.strip().lower() for item in value]
        if any(not item for item in items):
            raise ValueError("association hints 不能包含空字符串")
        return frozenset(items)


class GraphPolicy(_PolicyModel):
    """通用关键词抽取策略。"""

    # 通用停用词（语言级）；领域停用词仍由 DomainProfile 单独维护
    generic_stopwords: frozenset[str] = Field(default_factory=frozenset)

    @field_validator("generic_stopwords")
    @classmethod
    def _normalize_stopwords(
        cls, value: frozenset[str]
    ) -> frozenset[str]:
        """统一去空白/小写，与关键词归一化口径一致。"""
        items = [item.strip().lower() for item in value]
        if any(not item for item in items):
            raise ValueError("generic_stopwords 不能包含空字符串")
        return frozenset(items)


class StageProgressPolicy(_PolicyModel):
    """索引阶段进度映射（仅展示/投影，不参与业务判断）。"""

    stage_progress: dict[IndexingStage, float] = Field(default_factory=dict)

    @field_validator("stage_progress")
    @classmethod
    def _check_progress(
        cls, value: dict[IndexingStage, float]
    ) -> dict[IndexingStage, float]:
        """进度必须在 [0,1]；缺失阶段由流水线保留当前进度。"""
        for stage, progress in value.items():
            if not 0.0 <= float(progress) <= 1.0:
                raise ValueError(f"{stage} 的进度必须在 0~1: {progress}")
        return {stage: float(progress) for stage, progress in value.items()}


class IndexerPolicy(_PolicyModel):
    """一个完整的 RAG 索引策略包。"""

    version: int = Field(default=1, ge=1)
    quality: QualityPenaltyPolicy = Field(
        default_factory=QualityPenaltyPolicy
    )
    description: DescriptionPolicy = Field(default_factory=DescriptionPolicy)
    association: AssociationPolicy = Field(
        default_factory=AssociationPolicy
    )
    graph: GraphPolicy = Field(default_factory=GraphPolicy)
    staging: StageProgressPolicy = Field(default_factory=StageProgressPolicy)


def load_indexer_policy(path: str | Path | None = None) -> IndexerPolicy:
    """加载并校验策略文件；``path`` 为空时使用内置 config/rag_policy.json。"""
    file = _resolve_policy_path(path)
    try:
        payload = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise _error(
            f"索引策略文件读取失败: {file}: {exc}",
            path=str(file),
        ) from exc
    if not isinstance(payload, dict):
        raise _error(f"索引策略文件必须是 JSON 对象: {file}", path=str(file))
    try:
        return IndexerPolicy.model_validate(payload)
    except ValidationError as exc:
        raise _error(
            f"索引策略不合法: {file}: {exc}",
            path=str(file),
        ) from exc


@lru_cache(maxsize=8)
def _load_cached(path: str) -> IndexerPolicy:
    """按文件路径缓存；策略文件变更需重启进程。"""
    return load_indexer_policy(path)


def get_indexer_policy(path: str | Path | None = None) -> IndexerPolicy:
    """获取策略单例（进程级缓存）；``path`` 为空用内置策略。"""
    return _load_cached(str(_resolve_policy_path(path)))


def _resolve_policy_path(path: str | Path | None) -> Path:
    """解析策略路径；相对路径按 ``verity-agent`` 源码根目录解析。"""
    if not path:
        return BUILTIN_INDEXER_POLICY_PATH
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = SOURCE_ROOT / candidate
    return candidate


def _error(message: str, *, path: str | None = None) -> RagError:
    """统一收敛为 RagError(INVALID_CONFIG)。"""
    details = {"path": path} if path else {}
    return RagError(
        RagErrorCode.INVALID_CONFIG,
        message,
        details=details,
    )
