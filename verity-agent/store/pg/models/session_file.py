"""session_files 行模型。"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from .base import DBModel
from .enums import SessionFileStatus


class SessionFile(DBModel):
    id: UUID
    conversation_id: UUID
    message_id: UUID | None = None
    uploaded_by: UUID
    original_filename: str
    extension: str
    size_bytes: int
    sha256: str | None = None
    source_object_key: str
    parse_manifest_key: str | None = None
    status: SessionFileStatus
    token_count: int | None = None
    created_at: datetime
