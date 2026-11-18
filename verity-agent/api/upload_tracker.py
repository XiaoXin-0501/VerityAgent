"""知识库上传的在途任务登记与取消。

上传请求是单实例内的长任务；删除知识库时需要先发取消信号，让读文件循环
在下一个分块边界退出。跨实例场景无法直接发信号，因此上传在写 MinIO 之后、
登记 PG 文档之前还会原子校验知识库是否仍为 active，作为最终兜底。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from uuid import UUID

__all__ = ["UploadToken", "KbUploadTracker", "kb_uploads"]


@dataclass(slots=True, eq=False)
class UploadToken:
    """一次上传请求的取消令牌；按对象身份登记，不做值比较。"""

    # 所属知识库
    kb_id: UUID
    # 置位后上传循环应尽快退出
    cancelled: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def is_cancelled(self) -> bool:
        """是否已收到取消信号。"""
        return self.cancelled.is_set()

    def cancel(self) -> None:
        """置位取消信号；幂等。"""
        self.cancelled.set()


class KbUploadTracker:
    """进程内按知识库登记在途上传。"""

    def __init__(self) -> None:
        """初始化空的登记表。"""
        self._tokens: dict[UUID, set[UploadToken]] = {}

    def begin(self, kb_id: UUID) -> UploadToken:
        """登记一次上传并返回取消令牌。"""
        token = UploadToken(kb_id=kb_id)
        self._tokens.setdefault(kb_id, set()).add(token)
        return token

    def end(self, token: UploadToken) -> None:
        """上传结束时注销令牌；幂等。"""
        bucket = self._tokens.get(token.kb_id)
        if bucket is None:
            return
        bucket.discard(token)
        if not bucket:
            self._tokens.pop(token.kb_id, None)

    def cancel_kb(self, kb_id: UUID) -> int:
        """向该知识库的全部在途上传发取消信号，返回数量。"""
        bucket = self._tokens.get(kb_id)
        if not bucket:
            return 0
        for token in list(bucket):
            token.cancel()
        return len(bucket)

    def active_count(self, kb_id: UUID) -> int:
        """当前仍在途的上传数。"""
        return len(self._tokens.get(kb_id, ()))

    async def wait_settled(self, kb_id: UUID, timeout_s: float) -> bool:
        """等待该知识库的在途上传退出。

        返回 True 表示已全部退出；超时返回 False，调用方应继续删除并由
        上传侧的原子校验兜底。``timeout_s <= 0`` 时不等待。
        """
        if timeout_s <= 0:
            return self.active_count(kb_id) == 0
        deadline = asyncio.get_running_loop().time() + timeout_s
        while self.active_count(kb_id) > 0:
            if asyncio.get_running_loop().time() >= deadline:
                return False
            await asyncio.sleep(0.05)
        return True


# 进程级登记表；上传路由与删除路由共享。
kb_uploads = KbUploadTracker()
