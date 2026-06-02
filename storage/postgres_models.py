from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import UserDefinedType

try:
    from pgvector.sqlalchemy import Vector
except ImportError:  # pragma: no cover - only used before optional dependency install
    class Vector(UserDefinedType):
        cache_ok = True

        def __init__(self, dimensions: int):
            self.dimensions = dimensions

        def get_col_spec(self, **kw):
            return f"vector({self.dimensions})"


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


class BotMemory(Base):
    __tablename__ = "bot_memory"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chat_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    sent_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    sent_message_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    response_text: Mapped[str] = mapped_column(Text, nullable=False)
    reply_to_user_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    reply_to_message_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    tone_calibration: Mapped[str | None] = mapped_column(Text, nullable=True)
    brief_snapshot: Mapped[dict] = mapped_column(JSONB, server_default="{}")
    stances: Mapped[dict] = mapped_column(JSONB, server_default="{}")
    prompt_version: Mapped[str] = mapped_column(Text, nullable=False)
    cycle_snapshot_message_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    current_posture: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_bot_memory_chat_sent_at", "chat_id", sent_at.desc()),
    )


class ConversationSummary(Base):
    __tablename__ = "conversation_summaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chat_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    chunk_start_message_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    chunk_end_message_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")


class BriefCache(Base):
    __tablename__ = "brief_cache"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chat_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    snapshot_message_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    brief_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")


class AiDecision(Base):
    __tablename__ = "ai_decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chat_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    evaluated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    prompt_version: Mapped[str] = mapped_column(Text, nullable=False)
    snapshot_message_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    new_message_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    should_respond: Mapped[bool] = mapped_column(Boolean, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, server_default="0")
    response_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    reply_to_message_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    sent_message_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    request1_latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    request2_latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    request1_tokens_used: Mapped[int | None] = mapped_column(Integer, nullable=True)
    request2_tokens_used: Mapped[int | None] = mapped_column(Integer, nullable=True)
    gate_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    gate_factors: Mapped[dict] = mapped_column(JSONB, server_default="{}")


class FailedCycle(Base):
    __tablename__ = "failed_cycles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chat_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    failed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    stage: Mapped[str] = mapped_column(Text, nullable=False)
    error_message: Mapped[str] = mapped_column(Text, nullable=False)
    raw_context_sent: Mapped[str | None] = mapped_column(Text, nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(Text, nullable=True)


class ReplyDistribution(Base):
    __tablename__ = "reply_distribution"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chat_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    reply_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    last_replied_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        UniqueConstraint("chat_id", "user_id", name="uq_reply_distribution_chat_user"),
    )


class StanceTracker(Base):
    __tablename__ = "stance_tracker"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chat_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    user_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    topic: Mapped[str] = mapped_column(Text, nullable=False)
    stance: Mapped[str] = mapped_column(Text, nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class PromptVersion(Base):
    __tablename__ = "prompt_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    perception_system_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    decision_system_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


class CircuitBreakerState(Base):
    __tablename__ = "circuit_breaker_state"

    chat_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    failure_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    last_failure_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    paused_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class BotPersonaCore(Base):
    __tablename__ = "bot_persona_core"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    identity_summary: Mapped[str] = mapped_column(Text, nullable=False)
    core_beliefs: Mapped[list] = mapped_column(JSONB, nullable=False)
    speaking_style: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(384), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")


class BotVectorMemory(Base):
    __tablename__ = "bot_vector_memories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chat_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    memory_type: Mapped[str] = mapped_column(Text, nullable=False)
    user_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(384), nullable=True)
    importance_score: Mapped[float] = mapped_column(Float, nullable=False, server_default="0")


class BotSelfReflection(Base):
    __tablename__ = "bot_self_reflections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chat_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    trigger: Mapped[str] = mapped_column(Text, nullable=False)
    messages_since_last: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    reflection_text: Mapped[str] = mapped_column(Text, nullable=False)
    updated_summary: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(384), nullable=True)
    drift_score: Mapped[float] = mapped_column(Float, nullable=False, server_default="0")


class UserRelationshipProfile(Base):
    __tablename__ = "user_relationship_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chat_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    first_interaction_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    last_interaction_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    total_exchanges: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    relationship_strength: Mapped[float] = mapped_column(Float, nullable=False, server_default="0")
    sentiment_trend: Mapped[float] = mapped_column(Float, nullable=False, server_default="0")
    receptiveness_score: Mapped[float] = mapped_column(Float, nullable=False, server_default="0")
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(384), nullable=True)

    __table_args__ = (
        UniqueConstraint("chat_id", "user_id", name="uq_relationship_chat_user"),
    )


class ResponseFeedback(Base):
    __tablename__ = "response_feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chat_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    bot_memory_id: Mapped[int] = mapped_column(Integer, ForeignKey("bot_memory.id"), nullable=False)
    sent_message_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    observation_window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    reply_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    reaction_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    reaction_types: Mapped[list] = mapped_column(JSONB, server_default="[]")
    quote_reply_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    follow_up_sentiment: Mapped[float] = mapped_column(Float, nullable=False, server_default="0")
    outcome: Mapped[str] = mapped_column(Text, nullable=False)
    outcome_score: Mapped[float] = mapped_column(Float, nullable=False, server_default="0")
    scored_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    meta_reflected: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")


class ChatActivityPattern(Base):
    __tablename__ = "chat_activity_patterns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chat_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    hour_of_day: Mapped[int] = mapped_column(Integer, nullable=False)
    day_of_week: Mapped[int] = mapped_column(Integer, nullable=False)
    avg_message_velocity: Mapped[float] = mapped_column(Float, nullable=False, server_default="0")
    avg_tension: Mapped[float] = mapped_column(Float, nullable=False, server_default="0")
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    last_updated: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("chat_id", "hour_of_day", "day_of_week", name="uq_activity_chat_hour_day"),
    )
