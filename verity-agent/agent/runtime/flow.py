"""单次请求的结构化流程详情日志（JSONL）。

与人类可读的 ``verity-agent.log`` 不同，本日志面向“完整还原一次用户请求”：

- 每次 run 一个 ``run_id``，所有事件按 ``seq`` 递增；
- 记录上下文装配、query 规划、每次 LLM 请求的完整 prompt/工具清单、
  LLM 响应与 usage、每次工具调用参数/结果/耗时/成功失败、反思、RAG
  事件与 run 终态；
- 每行一个 JSON 对象，便于 grep/jq/导入分析；敏感字段不做额外裁剪，
  但 prompt/工具结果按配置截断，避免单行过大。

日志文件由 ``LoggingConfig.flow_file_name`` 指定，与 app/access/error/rag
日志分开轮转。
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any
from uuid import UUID

from agent.llm import (
    FilePart,
    ImagePart,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    TextPart,
    ThinkingPart,
)
from config.app import LoggingConfig
from config.base import SOURCE_ROOT
from utils.time import utcnow

__all__ = [
    "JsonlFlowLog",
    "RunFlowLog",
    "build_flow_log",
    "serialize_llm_request",
    "serialize_llm_response",
    "serialize_messages",
]


class JsonlFlowLog:
    """JSONL 流程日志；按大小轮转，写入失败静默降级。"""

    def __init__(
        self,
        path: str | Path,
        *,
        include_prompts: bool = True,
        max_prompt_chars: int = 20000,
        max_result_chars: int = 4000,
        max_bytes: int = 64 * 1024 * 1024,
        backup_count: int = 10,
        encoding: str = "utf-8",
    ) -> None:
        """安装独立 logger/handler；不建立业务依赖。"""
        self.path = Path(path)
        self.include_prompts = bool(include_prompts)
        self.max_prompt_chars = max(500, int(max_prompt_chars))
        self.max_result_chars = max(100, int(max_result_chars))
        # 独立 logger + 不传播：流程日志不会混入 app.log/error.log。
        self._logger = logging.getLogger(f"verity.flow.{id(self)}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        self._handler: logging.Handler | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            handler = RotatingFileHandler(
                self.path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding=encoding,
                delay=True,
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)
            self._handler = handler
        except OSError:
            # 目录不可写时流程日志降级为 no-op，不阻断业务。
            self._handler = None

    @property
    def enabled(self) -> bool:
        """是否成功打开日志文件。"""
        return self._handler is not None

    def bind_run(
        self,
        *,
        run_id: str,
        conversation_id: UUID | str | None,
        turn_id: str | None,
        user_id: UUID | str | None,
    ) -> "RunFlowLog":
        """创建一次 run 的写入器。"""
        return RunFlowLog(
            self,
            run_id=run_id,
            conversation_id=conversation_id,
            turn_id=turn_id,
            user_id=user_id,
        )

    def write(self, record: dict[str, Any]) -> None:
        """写入一行 JSON；失败只丢弃本条，不影响业务。"""
        if self._handler is None:
            return
        try:
            line = json.dumps(record, ensure_ascii=False, default=_json_default)
            self._logger.info(line)
        except Exception:  # noqa: BLE001 - 日志失败不能影响业务
            return

    def close(self) -> None:
        """关闭 handler；测试/优雅退出使用。"""
        if self._handler is not None:
            self._logger.removeHandler(self._handler)
            self._handler.close()
            self._handler = None


class RunFlowLog:
    """绑定单个 run 的流程日志写入器。"""

    def __init__(
        self,
        sink: JsonlFlowLog,
        *,
        run_id: str,
        conversation_id: UUID | str | None,
        turn_id: str | None,
        user_id: UUID | str | None,
    ) -> None:
        """保存公共上下文字段。"""
        self.sink = sink
        self.run_id = run_id
        self.conversation_id = (
            str(conversation_id) if conversation_id is not None else None
        )
        self.turn_id = turn_id
        self.user_id = str(user_id) if user_id is not None else None
        self._seq = 0
        self._lock = threading.Lock()

    def event(self, name: str, **fields: Any) -> None:
        """写入一个流程事件；``name`` 是稳定的事件名。"""
        with self._lock:
            self._seq += 1
            seq = self._seq
        record: dict[str, Any] = {
            "ts": utcnow().isoformat(),
            "seq": seq,
            "run_id": self.run_id,
            "conversation_id": self.conversation_id,
            "turn_id": self.turn_id,
            "user_id": self.user_id,
            "event": name,
        }
        for key, value in fields.items():
            if value is not None:
                record[key] = value
        self.sink.write(record)


def build_flow_log(
    config: LoggingConfig | None = None,
    *,
    enabled: bool | None = None,
) -> JsonlFlowLog | None:
    """按日志配置构建流程日志；未启用返回 None。"""
    cfg = config or LoggingConfig()
    if enabled is None:
        # 总文件日志关闭时流程日志也关闭，避免出现“只写 flow 不写 app”。
        enabled = cfg.flow_enabled and cfg.to_file
    if not enabled:
        return None
    directory = Path(cfg.directory).expanduser()
    if not directory.is_absolute():
        directory = SOURCE_ROOT / directory
    return JsonlFlowLog(
        directory / cfg.flow_file_name,
        include_prompts=cfg.flow_include_prompts,
        max_prompt_chars=cfg.flow_max_prompt_chars,
        max_result_chars=cfg.flow_max_result_chars,
        max_bytes=cfg.max_bytes,
        backup_count=cfg.backup_count,
        encoding=cfg.encoding,
    )


def serialize_llm_request(
    request: LLMRequest,
    *,
    include_prompts: bool = True,
    max_chars: int = 20000,
) -> dict[str, Any]:
    """把 LLMRequest 投影成流程日志字段；prompt 可整体关闭。"""
    data: dict[str, Any] = {
        "request_id": request.request_id,
        "model_id": request.model_id,
        "purpose": request.context.purpose.value,
        "message_count": len(request.messages),
        "tool_count": len(request.tools),
        "tools": [
            {
                "name": item.name,
                "description": item.description,
                "parameters": item.parameters,
            }
            for item in request.tools
        ],
        "tool_choice": request.tool_choice.model_dump(mode="json"),
        "stream": request.stream,
        "deadline": _json_default(request.context.deadline)
        if request.context.deadline is not None
        else None,
    }
    if include_prompts:
        data["messages"] = serialize_messages(
            request.messages,
            max_chars=max_chars,
        )
    return data


def serialize_llm_response(response: LLMResponse) -> dict[str, Any]:
    """把 LLMResponse 投影成流程日志字段。"""
    message = response.message
    return {
        "request_id": response.request_id,
        "provider": response.provider,
        "model": response.model,
        "response_id": response.response_id,
        "finish_reason": response.finish_reason.value,
        "latency_ms": response.latency_ms,
        "first_token_ms": response.first_token_ms,
        "usage": response.usage.model_dump(mode="json"),
        "text": _truncate(
            "".join(
                part.text
                for part in message.content
                if isinstance(part, TextPart)
            ),
            20000,
        ),
        "thinking": [
            part.text
            for part in message.content
            if isinstance(part, ThinkingPart)
        ],
        "tool_calls": [
            {
                "call_id": call.call_id,
                "name": call.name,
                "arguments": call.arguments,
            }
            for call in message.tool_calls
        ],
    }


def serialize_messages(
    messages: list[LLMMessage],
    *,
    max_chars: int = 20000,
) -> list[dict[str, Any]]:
    """投影消息列表；文本拼成一个整体后按 max_chars 截断，保留结构。"""
    result: list[dict[str, Any]] = []
    for message in messages:
        texts: list[str] = []
        attachments: list[dict[str, Any]] = []
        for part in message.content:
            if isinstance(part, TextPart):
                texts.append(part.text)
            elif isinstance(part, ThinkingPart):
                texts.append(f"[thinking] {part.text}")
            elif isinstance(part, ImagePart):
                attachments.append(
                    {
                        "type": "image",
                        "url": part.url,
                        "file_id": str(part.file_id) if part.file_id else None,
                        "mime_type": part.mime_type,
                    }
                )
            elif isinstance(part, FilePart):
                attachments.append(
                    {
                        "type": "file",
                        "file_id": str(part.file_id) if part.file_id else None,
                        "filename": part.filename,
                        "mime_type": part.mime_type,
                        "url": part.url,
                        "text_chars": len(part.text or ""),
                    }
                )
        text = "\n".join(texts)
        result.append(
            {
                "role": message.role.value,
                "text": _truncate(text, max_chars),
                "text_chars": len(text),
                "truncated": len(text) > max_chars,
                "attachments": attachments or None,
                "tool_calls": [
                    {
                        "call_id": call.call_id,
                        "name": call.name,
                        "arguments": call.arguments,
                    }
                    for call in message.tool_calls
                ]
                or None,
                "tool_call_id": message.tool_call_id,
            }
        )
    return result


def _truncate(value: str, limit: int) -> str:
    """按字符数截断；保留尾部标记，便于区分“本来就这么长”。"""
    if limit <= 0 or len(value) <= limit:
        return value
    return value[:limit] + f"…[truncated {len(value) - limit} chars]"


def _json_default(value: Any) -> Any:
    """JSON 序列化兜底：datetime/UUID/Pydantic/Path/异常。"""
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BaseException):
        return str(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return str(value)
