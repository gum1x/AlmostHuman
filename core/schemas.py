from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from core.constants import EventType, MessageType


class ForwardMetadata(BaseModel):
    from_id: int | None = None
    from_name: str | None = None
    from_chat_id: int | None = None
    from_message_id: int | None = None
    date: datetime | None = None


class MediaMetadata(BaseModel):
    media_type: str | None = None
    file_id: str | None = None
    file_size: int | None = None
    mime_type: str | None = None
    duration: int | None = None
    width: int | None = None
    height: int | None = None


class SenderInfo(BaseModel):
    sender_id: int
    username: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    is_bot: bool = False
    is_premium: bool = False


class RawTelegramEvent(BaseModel):
    event_type: EventType
    message_id: int
    chat_id: int
    sender_id: int | None = None
    timestamp: datetime
    text: str | None = None
    reply_to_message_id: int | None = None
    forward: ForwardMetadata | None = None
    media: MediaMetadata | None = None
    entities: list[dict[str, Any]] = Field(default_factory=list)
    grouped_id: int | None = None
    sender_info: SenderInfo | None = None
    deleted_message_ids: list[int] = Field(default_factory=list)
    reactions: list[dict[str, Any]] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


class CanonicalMessage(BaseModel):
    message_id: int
    chat_id: int
    sender_id: int | None = None
    timestamp: datetime
    message_type: MessageType = MessageType.TEXT
    text_raw: str | None = None
    text_cleaned: str | None = None
    reply_to_message_id: int | None = None
    forward_from_id: int | None = None
    forward_from_chat_id: int | None = None
    forward_from_message_id: int | None = None
    forward_from_name: str | None = None
    forward_date: datetime | None = None
    media_type: str | None = None
    media_file_id: str | None = None
    media_file_size: int | None = None
    media_mime_type: str | None = None
    media_duration: int | None = None
    media_width: int | None = None
    media_height: int | None = None
    mention_list: list[str] = Field(default_factory=list)
    entity_list: list[dict[str, Any]] = Field(default_factory=list)
    grouped_id: int | None = None
    raw_event: dict[str, Any] = Field(default_factory=dict)


class MessageResponse(BaseModel):
    id: int
    message_id: int
    chat_id: int
    sender_id: int | None
    timestamp: datetime
    message_type: str
    text_raw: str | None
    text_cleaned: str | None
    reply_to_message_id: int | None
    is_deleted: bool
    edit_history: list[dict[str, Any]]
    mention_list: list[str]
    created_at: datetime
    updated_at: datetime


class PaginatedResponse(BaseModel):
    items: list[MessageResponse]
    total: int
    limit: int
    offset: int


class HealthResponse(BaseModel):
    status: str
    postgres: str
    redis: str
    stream_lag: int | None = None
