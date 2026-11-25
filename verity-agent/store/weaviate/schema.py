"""Weaviate schema 加载与校验（D-97）。

- ``weaviate_schema_import.json`` 是唯一手写 schema 源；
- 应用启动时读取它，与线上 collection 配置比对，产出 ``SchemaCheckResult``；
- collection/property 的创建由根目录 ``init.py`` / 启动期
  ``ensure_initialized`` 显式执行，运行期不自动改结构。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field

from config import WeaviateConfig
from config.base import PROJECT_ROOT
from exception import ConfigurationError, WeaviateOperationError, WeaviateSchemaError

__all__ = [
    "SchemaCheckResult",
    "SchemaManager",
    "load_schema_import",
]


class SchemaCheckResult(BaseModel):
    """schema 校验结果；``ok=False`` 时 prod 应 fail-fast。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # 是否全部通过
    ok: bool = True
    # 缺失的 collection
    missing_collections: list[str] = Field(default_factory=list)
    # 缺失的属性，格式 ``Collection.property``
    missing_properties: list[str] = Field(default_factory=list)
    # 类型不一致：{collection, property, expected, actual}
    type_mismatches: list[dict[str, Any]] = Field(default_factory=list)
    # 分词不一致：{collection, property, expected, actual}
    tokenization_mismatches: list[dict[str, Any]] = Field(default_factory=list)
    # 向量维度不一致：{collection, expected, actual}
    dimension_mismatches: list[dict[str, Any]] = Field(default_factory=list)
    # 非阻塞提示
    warnings: list[str] = Field(default_factory=list)

    def raise_if_bad(self) -> None:
        """校验失败时抛出 WeaviateSchemaError。"""
        if self.ok:
            return
        raise WeaviateSchemaError(
            "Weaviate schema 校验失败",
            details=self.model_dump(mode="json"),
        )


def load_schema_import(path: str | Path) -> dict[str, Any]:
    """读取手写 schema import JSON。"""
    file_path = Path(path)
    if not file_path.is_absolute() and not file_path.exists():
        # 从 verity-agent 子目录启动时，相对路径按仓库根目录解析
        candidate = PROJECT_ROOT / file_path
        if candidate.exists():
            file_path = candidate
    if not file_path.exists():
        raise ConfigurationError(
            "Weaviate schema import 文件不存在",
            details={"path": str(file_path)},
        )
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigurationError(
            "Weaviate schema import 文件不合法",
            details={"path": str(file_path), "cause": str(exc)},
        ) from exc
    if not isinstance(data, dict) or not isinstance(data.get("classes"), list):
        raise ConfigurationError(
            "Weaviate schema import 缺少 classes 数组",
            details={"path": str(file_path)},
        )
    return data


def _value(value: Any) -> Any:
    """取枚举值；非枚举值原样返回。"""
    return getattr(value, "value", value)


class SchemaManager:
    """基于 import JSON 的 schema 校验/创建。"""

    def __init__(self, config: WeaviateConfig, schema: Mapping[str, Any] | None = None):
        """保存配置并加载手写 schema（未传入时读文件）。"""
        self.config = config
        self._schema = dict(schema) if schema is not None else load_schema_import(
            config.schema_import_path
        )

    @property
    def schema(self) -> dict[str, Any]:
        """返回 schema 的浅拷贝。"""
        return dict(self._schema)

    @property
    def expected_collections(self) -> dict[str, Mapping[str, Any]]:
        """按 class 名索引期望的 collection 定义。"""
        return {item["class"]: item for item in self._schema.get("classes", [])}

    def _schema_version_warning(self) -> str | None:
        """校验 ``WeaviateConfig.expected_schema_version`` 与 manifest 版本。

        版本不一致只写 warning，不阻断启动：自定义 schema 文件可能没有
        manifest；prod 的强校验仍由 schema 属性/分词/维度检查负责。
        """
        expected = (self.config.expected_schema_version or "").strip()
        if not expected:
            return None
        path = Path(self.config.schema_import_path)
        if not path.is_absolute() and not path.exists():
            path = PROJECT_ROOT / path
        manifest = path.with_name("weaviate_schema_manifest.json")
        if not manifest.is_file():
            return f"schema manifest 不存在，跳过版本校验: {manifest}"
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return f"schema manifest 读取失败，跳过版本校验: {exc}"
        actual = str(payload.get("schema_version") or "").strip()
        if actual and actual != expected:
            return (
                f"schema_version 不一致（expected={expected}, actual={actual}）"
            )
        return None

    async def check(
        self,
        client,
        *,
        expected_dimensions: int | None = None,
    ) -> SchemaCheckResult:
        """比对线上 collection 与 schema，返回差异结果。"""
        try:
            existing = await asyncio.to_thread(client.collections.list_all, False)
        except Exception as exc:
            raise WeaviateOperationError(
                "读取 Weaviate collection 配置失败", details={"cause": str(exc)}
            ) from exc
        missing_collections: list[str] = []
        missing_properties: list[str] = []
        type_mismatches: list[dict[str, Any]] = []
        tokenization_mismatches: list[dict[str, Any]] = []
        dimension_mismatches: list[dict[str, Any]] = []
        warnings: list[str] = []
        version_warning = self._schema_version_warning()
        if version_warning:
            warnings.append(version_warning)

        for name, expected in self.expected_collections.items():
            actual = existing.get(name)
            if actual is None:
                missing_collections.append(name)
                continue

            actual_props = {
                getattr(prop, "name", ""): prop
                for prop in getattr(actual, "properties", []) or []
            }
            for prop in expected.get("properties", []):
                prop_name = prop.get("name")
                actual_prop = actual_props.get(prop_name)
                if actual_prop is None:
                    missing_properties.append(f"{name}.{prop_name}")
                    continue
                expected_type = (prop.get("dataType") or [None])[0]
                actual_type = _value(getattr(actual_prop, "data_type", None))
                if expected_type and actual_type != expected_type:
                    type_mismatches.append(
                        {
                            "collection": name,
                            "property": prop_name,
                            "expected": expected_type,
                            "actual": actual_type,
                        }
                    )
                expected_tok = prop.get("tokenization")
                if expected_tok:
                    actual_tok = _value(getattr(actual_prop, "tokenization", None))
                    if actual_tok != expected_tok:
                        tokenization_mismatches.append(
                            {
                                "collection": name,
                                "property": prop_name,
                                "expected": expected_tok,
                                "actual": actual_tok,
                            }
                        )

            if expected_dimensions is not None:
                actual_dim = getattr(actual, "vector_dimensions", None)
                if actual_dim is None:
                    warnings.append(
                        f"{name}: 无法从 collection 配置读取向量维度，跳过维度校验"
                    )
                elif int(actual_dim) != int(expected_dimensions):
                    dimension_mismatches.append(
                        {
                            "collection": name,
                            "expected": expected_dimensions,
                            "actual": actual_dim,
                        }
                    )

            expected_vector = bool(expected.get("vectorConfig"))
            actual_vector = getattr(actual, "vector_config", None) is not None
            if expected_vector != actual_vector:
                warnings.append(
                    f"{name}: 向量配置可能不一致（expected={expected_vector}, "
                    f"actual={actual_vector}）"
                )

        ok = not (
            missing_collections
            or missing_properties
            or type_mismatches
            or tokenization_mismatches
            or dimension_mismatches
        )
        return SchemaCheckResult(
            ok=ok,
            missing_collections=missing_collections,
            missing_properties=missing_properties,
            type_mismatches=type_mismatches,
            tokenization_mismatches=tokenization_mismatches,
            dimension_mismatches=dimension_mismatches,
            warnings=warnings,
        )

    async def ensure(
        self,
        client,
        *,
        expected_dimensions: int | None = None,
    ) -> SchemaCheckResult:
        """创建缺失 collection、补齐缺失 property，然后重新校验。

        只应在 dev/test 调用；prod 的 property 变更走独立迁移流程。
        """
        try:
            existing = await asyncio.to_thread(client.collections.list_all, False)
        except Exception as exc:
            raise WeaviateOperationError(
                "读取 Weaviate collection 配置失败", details={"cause": str(exc)}
            ) from exc

        # 1) 创建缺失 collection
        for name, config in self.expected_collections.items():
            if name not in existing:
                try:
                    await asyncio.to_thread(
                        client.collections.create_from_dict, dict(config)
                    )
                except Exception as exc:
                    raise WeaviateSchemaError(
                        f"创建 Weaviate collection 失败: {name}",
                        details={"cause": str(exc)},
                    ) from exc

        # 2) 给已有 collection 补缺失 property（只加不删）
        try:
            existing = await asyncio.to_thread(client.collections.list_all, False)
        except Exception as exc:
            raise WeaviateOperationError(
                "读取 Weaviate collection 配置失败", details={"cause": str(exc)}
            ) from exc
        for name, expected in self.expected_collections.items():
            actual = existing.get(name)
            if actual is None:
                continue
            actual_names = {
                getattr(prop, "name", "")
                for prop in getattr(actual, "properties", []) or []
            }
            missing = [
                prop
                for prop in expected.get("properties", [])
                if prop.get("name") not in actual_names
            ]
            if not missing:
                continue
            collection = client.collections.get(name)
            for prop in missing:
                try:
                    await asyncio.to_thread(
                        collection.config.add_property,
                        self._property_from_schema(prop),
                    )
                except Exception as exc:
                    raise WeaviateSchemaError(
                        f"为 {name} 添加 property 失败: {prop.get('name')}",
                        details={"cause": str(exc)},
                    ) from exc

        result = await self.check(
            client,
            expected_dimensions=expected_dimensions,
        )
        if not result.ok:
            result.raise_if_bad()
        return result

    @staticmethod
    def _property_from_schema(prop: Mapping[str, Any]):
        """把 import JSON 的 property 定义转成 Weaviate SDK Property。"""
        from weaviate.collections.classes.config import DataType, Property, Tokenization

        data_type = (prop.get("dataType") or ["text"])[0]
        kwargs: dict[str, Any] = {}
        tokenization = prop.get("tokenization")
        if tokenization:
            kwargs["tokenization"] = Tokenization(tokenization)
        return Property(
            name=prop["name"],
            data_type=DataType(data_type),
            **kwargs,
        )
