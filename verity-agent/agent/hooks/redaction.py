"""日志/审计共用的脱敏策略。"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Mapping

from config.hooks import RedactionConfig
from utils.serialization import to_jsonable

__all__ = ["RedactionPolicy"]


class RedactionPolicy:
    def __init__(self, config: RedactionConfig) -> None:
        self.config = config
        self._fields = {item.lower() for item in config.fields}
        self._patterns = [re.compile(item) for item in config.patterns]

    def redact_text(self, text: str) -> str:
        if not self.config.enabled:
            return text
        value = text
        for pattern in self._patterns:
            value = pattern.sub(self.config.replacement, value)
        if len(value) > self.config.max_text_length:
            value = value[: self.config.max_text_length] + "…"
        return value

    def redact_value(self, value: Any) -> Any:
        if not self.config.enabled:
            return value
        if isinstance(value, str):
            return self.redact_text(value)
        if isinstance(value, Mapping):
            return {
                str(key): (
                    self.config.replacement
                    if str(key).lower() in self._fields
                    else self.redact_value(item)
                )
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple, set, frozenset)):
            return [self.redact_value(item) for item in value]
        return value

    def payload_summary(self, payload: Any, *, max_bytes: int = 2048) -> str:
        data = self.redact_value(to_jsonable(payload))
        text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        if len(text.encode("utf-8")) > max_bytes:
            text = text.encode("utf-8")[:max_bytes].decode("utf-8", "ignore") + "…"
        return text

    def redact_mapping(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return self.redact_value(value)  # type: ignore[return-value]
