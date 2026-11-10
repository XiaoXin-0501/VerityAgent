"""数据库行模型的公共基类。

行模型只负责“字段 + 类型校验”，不携带任何行为；读写统一由仓库层完成。
``extra="ignore"`` 保证表结构以后扩列时旧代码不会因为多出来的列报错。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from utils.time import utcnow

__all__ = ["DBModel", "DBJsonModel", "utcnow"]


class DBModel(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
        from_attributes=True,
        populate_by_name=True,
    )


class DBJsonModel(BaseModel):
    """JSONB 结构模型基类。

    与行模型不同，JSONB 结构是“固定协议”，用 ``extra="forbid"`` 挡住未知
    字段；需要扩展时必须升 version。
    """

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )

    def to_json_dict(self) -> dict[str, Any]:
        """转成可直接写入 JSONB 的字典（别名 + JSON 原生类型）。"""
        return self.model_dump(mode="json", by_alias=True)
