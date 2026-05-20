from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    message_id: Mapped[int] = mapped_column(Integer, nullable=False)
    chat_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    sender_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    message_type: Mapped[str] = mapped_column(String(20), nullable=False, server_default="text")

    text_raw: Mapped[str | None] = mapped_column(Text, nullable=True)
    text_cleaned: Mapped[str | None] = mapped_column(Text, nullable=True)

    reply_to_message_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    forward_from_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    forward_from_chat_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    forward_from_message_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    forward_from_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    forward_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    media_type: Mapped[str | None] = mapped_column(String(30), nullable=True)
    media_file_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    media_file_size: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    media_mime_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    media_duration: Mapped[int | None] = mapped_column(Integer, nullable=True)
    media_width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    media_height: Mapped[int | None] = mapped_column(Integer, nullable=True)

    mention_list: Mapped[dict] = mapped_column(JSONB, server_default="[]")
    entity_list: Mapped[dict] = mapped_column(JSONB, server_default="[]")
    edit_history: Mapped[dict] = mapped_column(JSONB, server_default="[]")
    grouped_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)

    is_deleted: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    raw_event: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        UniqueConstraint("chat_id", "message_id", name="uq_chat_message"),
        Index("ix_messages_chat_id_timestamp", "chat_id", timestamp.desc()),
        Index("ix_messages_sender_id_timestamp", "sender_id", timestamp.desc()),
        Index(
            "ix_messages_chat_id_reply_to",
            "chat_id",
            "reply_to_message_id",
            postgresql_where=(reply_to_message_id.isnot(None)),
        ),
        Index(
            "ix_messages_grouped_id",
            "grouped_id",
            postgresql_where=(grouped_id.isnot(None)),
        ),
        Index(
            "ix_messages_active",
            "chat_id",
            timestamp.desc(),
            postgresql_where=(is_deleted.is_(False)),
        ),
    )


class Chat(Base):
    __tablename__ = "chats"

    chat_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    username: Mapped[str | None] = mapped_column(String(255), nullable=True)
    chat_type: Mapped[str | None] = mapped_column(String(20), nullable=True)
    member_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    last_message_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, server_default="{}")


class Sender(Base):
    __tablename__ = "senders"

    sender_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    username: Mapped[str | None] = mapped_column(String(255), nullable=True)
    first_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    last_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    is_bot: Mapped[bool] = mapped_column(Boolean, server_default="false")
    is_premium: Mapped[bool] = mapped_column(Boolean, server_default="false")
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, server_default="{}")

    __table_args__ = (
        Index("ix_senders_username", "username", postgresql_where=(username.isnot(None))),
    )


class ChatMember(Base):
    __tablename__ = "chat_members"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    chat_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    sender_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    message_count: Mapped[int] = mapped_column(Integer, server_default="0")

    __table_args__ = (
        UniqueConstraint("chat_id", "sender_id", name="uq_chat_sender"),
    )


class DeletionEvent(Base):
    __tablename__ = "deletion_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    chat_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    message_ids: Mapped[list[int]] = mapped_column(ARRAY(Integer), nullable=False)
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
