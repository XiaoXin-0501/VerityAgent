"""Hook 注册表：名称唯一、按事件索引、注册顺序稳定。"""

from __future__ import annotations

from threading import RLock

from exception import HookRegistrationError

from .enum import HookEvent
from .models import HookHandler, HookSpec, RegisteredHook

__all__ = ["HookRegistry"]


class HookRegistry:
    def __init__(self) -> None:
        """初始化 Hook 表与注册顺序计数。"""
        self._hooks: dict[str, RegisteredHook] = {}
        self._sequence = 0
        self._lock = RLock()

    def register(
        self,
        spec: HookSpec,
        handler: HookHandler,
        *,
        source: str | None = None,
    ) -> RegisteredHook:
        """注册 Hook；名称唯一，sequence 决定同优先级顺序。"""
        with self._lock:
            if spec.name in self._hooks:
                raise HookRegistrationError(f"Hook 重复注册: {spec.name}")
            self._sequence += 1
            registered = RegisteredHook(
                spec=spec,
                handler=handler,
                source=source,
                sequence=self._sequence,
            )
            self._hooks[spec.name] = registered
            return registered

    def unregister(self, name: str) -> bool:
        """移除 Hook，返回是否存在。"""
        with self._lock:
            return self._hooks.pop(name, None) is not None

    def get(self, name: str) -> RegisteredHook | None:
        """按名称取 Hook。"""
        return self._hooks.get(name)

    def list(self, event: HookEvent | None = None) -> list[RegisteredHook]:
        """列出全部 Hook，可按事件过滤。"""
        hooks = list(self._hooks.values())
        if event is not None:
            hooks = [hook for hook in hooks if event in hook.spec.events]
        return hooks

    def clear(self) -> None:
        """清空注册表并重置顺序。"""
        with self._lock:
            self._hooks.clear()
            self._sequence = 0

    def __len__(self) -> int:
        """已注册 Hook 数量。"""
        return len(self._hooks)
