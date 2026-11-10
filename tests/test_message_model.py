"""消息领域模型与存储编解码测试（不需要数据库）。"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError as PydanticValidationError

_ROOT = Path(__file__).resolve().parents[1]
_AGENT_ROOT = _ROOT / "verity-agent"
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

from agent.message import (  # noqa: E402
    Artifact,
    ArtifactKind,
    AssistantMessage,
    FileRef,
    ImageDetail,
    ImageRef,
    MessageRole,
    MessageStatus,
    StoredMessage,
    SystemMessage,
    SystemOrigin,
    Thinking,
    ToolCall,
    ToolCallStatus,
    ToolFile,
    ToolImage,
    ToolMessage,
    UserMessage,
    decode_content,
    decode_row,
    decode_stored_row,
    encode_content,
)
from exception import (  # noqa: E402
    MessageDecodeError,
    UnknownMessageRoleError,
    UnsupportedMessageVersionError,
)

CONVERSATION_ID = uuid4()
NOW = datetime(2026, 11, 4, 10, 0, tzinfo=timezone.utc)


def _decode(role, payload, *, status=MessageStatus.COMPLETED, message_id=None, created_at=NOW):
    return decode_content(
        role,
        payload,
        message_id=message_id or uuid4(),
        conversation_id=CONVERSATION_ID,
        status=status,
        created_at=created_at,
    )


def test_user_message_roundtrip() -> None:
    message = UserMessage(
        conversation_id=CONVERSATION_ID,
        content="这张图里是什么？",
        images=[ImageRef(url="https://cdn.example.com/a.png", detail=ImageDetail.HIGH)],
        files=[FileRef(file_id=uuid4(), filename="report.pdf")],
    )
    payload = encode_content(message)

    assert payload["v"] == 2
    assert payload["text"] == message.content
    assert payload["images"][0]["url"] == "https://cdn.example.com/a.png"
    assert "id" not in payload
    assert "conversation_id" not in payload
    assert "role" not in payload
    assert "status" not in payload
    assert "created_at" not in payload

    decoded = _decode(
        message.role,
        payload,
        message_id=message.id,
        created_at=message.created_at,
    )
    assert decoded == message


def test_system_message_roundtrip() -> None:
    message = SystemMessage(
        conversation_id=CONVERSATION_ID,
        content="你是 VerityAgent。",
        name="default",
        origin=SystemOrigin.BASE,
    )
    payload = encode_content(message)

    assert payload["name"] == "default"
    assert payload["origin"] == "base"

    decoded = _decode(
        message.role,
        payload,
        message_id=message.id,
        created_at=message.created_at,
    )
    assert decoded == message


def test_assistant_message_roundtrip() -> None:
    message = AssistantMessage(
        conversation_id=CONVERSATION_ID,
        content="我先查一下。",
        thinking=[Thinking(text="需要检索知识库", signature="sig-1")],
        tool_calls=[
            ToolCall(
                call_id="call_1",
                name="search_kb",
                arguments={"query": "Weaviate"},
                status=ToolCallStatus.RUNNING,
            )
        ],
        artifacts=[
            Artifact(
                kind=ArtifactKind.CHART,
                url="https://cdn.example.com/chart.png",
                filename="chart.png",
            )
        ],
        is_prefix=True,
    )
    payload = encode_content(message)

    assert payload["is_prefix"] is True
    assert payload["tool_calls"][0]["call_id"] == "call_1"
    assert payload["thinking"][0]["signature"] == "sig-1"

    decoded = _decode(
        message.role,
        payload,
        message_id=message.id,
        created_at=message.created_at,
    )
    assert decoded == message


def test_tool_message_roundtrip() -> None:
    message = ToolMessage(
        conversation_id=CONVERSATION_ID,
        content="命中 5 个 chunk。",
        tool_call_id="call_1",
        name="search_kb",
        json={"chunks": [{"chunk_id": "c1"}]},
        files=[ToolFile(filename="result.json", content='{"ok": true}')],
        images=[ToolImage(url="https://cdn.example.com/tool.png")],
    )
    payload = encode_content(message)

    assert payload["tool_call_id"] == "call_1"
    assert payload["json"] == {"chunks": [{"chunk_id": "c1"}]}
    assert "json_data" not in payload

    decoded = _decode(
        message.role,
        payload,
        message_id=message.id,
        created_at=message.created_at,
    )
    assert decoded == message


def test_decode_row_and_stored_row() -> None:
    message = UserMessage(
        conversation_id=CONVERSATION_ID,
        content="你好",
    )
    payload = encode_content(message)
    row = {
        "id": str(message.id),
        "conversation_id": str(message.conversation_id),
        "seq_no": 3,
        "role": "user",
        "status": "completed",
        "created_at": message.created_at.isoformat(),
        "content": payload,
    }

    decoded = decode_row(row)
    assert decoded == message

    stored = decode_stored_row(row)
    assert isinstance(stored, StoredMessage)
    assert stored.seq_no == 3
    assert stored.message == message


def test_decode_row_accepts_json_string_content() -> None:
    message = UserMessage(conversation_id=CONVERSATION_ID, content="你好")
    row = {
        "id": str(message.id),
        "conversation_id": str(message.conversation_id),
        "role": "user",
        "status": "completed",
        "created_at": message.created_at.isoformat(),
        "content": json.dumps(encode_content(message), ensure_ascii=False),
    }

    assert decode_row(row) == message


def test_unsupported_version_raises() -> None:
    with pytest.raises(UnsupportedMessageVersionError):
        _decode(MessageRole.USER, {"v": 99, "text": "hi"})


def test_unknown_role_raises() -> None:
    with pytest.raises(UnknownMessageRoleError):
        _decode("developer", {"v": 1, "text": "hi"})


def test_decode_invalid_content_raises_message_decode_error() -> None:
    with pytest.raises(MessageDecodeError):
        _decode(MessageRole.TOOL, {"v": 1, "text": "ok", "name": "search_kb"})


def test_user_message_requires_payload() -> None:
    with pytest.raises(PydanticValidationError):
        UserMessage(conversation_id=CONVERSATION_ID, content="")


def test_user_message_rejects_duplicate_file_ids() -> None:
    file_id = uuid4()
    with pytest.raises(PydanticValidationError):
        UserMessage(
            conversation_id=CONVERSATION_ID,
            content="两个文件",
            files=[FileRef(file_id=file_id), FileRef(file_id=file_id)],
        )


def test_assistant_rejects_duplicate_call_ids() -> None:
    with pytest.raises(PydanticValidationError):
        AssistantMessage(
            conversation_id=CONVERSATION_ID,
            content="调用两次",
            tool_calls=[
                ToolCall(call_id="call_1", name="a"),
                ToolCall(call_id="call_1", name="b"),
            ],
        )


def test_streaming_and_failed_assistant_can_be_empty() -> None:
    streaming = AssistantMessage(
        conversation_id=CONVERSATION_ID,
        status=MessageStatus.STREAMING,
    )
    failed = AssistantMessage(
        conversation_id=CONVERSATION_ID,
        status=MessageStatus.FAILED,
    )
    assert streaming.content == ""
    assert failed.content == ""


def test_tool_message_requires_result() -> None:
    with pytest.raises(PydanticValidationError):
        ToolMessage(
            conversation_id=CONVERSATION_ID,
            content="",
            tool_call_id="call_1",
            name="search_kb",
        )


def test_tool_file_requires_url_or_content() -> None:
    with pytest.raises(PydanticValidationError):
        ToolFile(filename="empty.bin")


def test_tool_json_must_be_serializable() -> None:
    with pytest.raises(PydanticValidationError):
        ToolMessage(
            conversation_id=CONVERSATION_ID,
            content="ok",
            tool_call_id="call_1",
            name="search_kb",
            json=object(),
        )


def test_stored_message_requires_positive_seq() -> None:
    message = UserMessage(conversation_id=CONVERSATION_ID, content="hi")
    with pytest.raises(PydanticValidationError):
        StoredMessage(seq_no=0, message=message)
