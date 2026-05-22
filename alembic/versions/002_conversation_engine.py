"""conversation engine schema

Revision ID: 002
Revises: 001
Create Date: 2026-05-22
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.types import UserDefinedType

try:
    from pgvector.sqlalchemy import Vector
except ImportError:  # pragma: no cover
    class Vector(UserDefinedType):
        cache_ok = True

        def __init__(self, dimensions: int):
            self.dimensions = dimensions

        def get_col_spec(self, **kw):
            return f"vector({self.dimensions})"


revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def _vector_column(name: str):
    return sa.Column(name, Vector(384), nullable=True)


def upgrade():
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "bot_memory",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("chat_id", sa.BigInteger(), nullable=False),
        sa.Column("sent_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("sent_message_id", sa.BigInteger(), nullable=True),
        sa.Column("response_text", sa.Text(), nullable=False),
        sa.Column("reply_to_user_id", sa.BigInteger(), nullable=True),
        sa.Column("reply_to_message_id", sa.BigInteger(), nullable=True),
        sa.Column("reasoning", sa.Text(), nullable=True),
        sa.Column("tone_calibration", sa.Text(), nullable=True),
        sa.Column("brief_snapshot", postgresql.JSONB(), server_default="{}", nullable=True),
        sa.Column("stances", postgresql.JSONB(), server_default="{}", nullable=True),
        sa.Column("prompt_version", sa.Text(), nullable=False),
        sa.Column("cycle_snapshot_message_id", sa.BigInteger(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_bot_memory_chat_id", "bot_memory", ["chat_id"])
    op.create_index("ix_bot_memory_chat_sent_at", "bot_memory", ["chat_id", sa.text("sent_at DESC")])

    op.create_table(
        "conversation_summaries",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("chat_id", sa.BigInteger(), nullable=False),
        sa.Column("chunk_start_message_id", sa.BigInteger(), nullable=False),
        sa.Column("chunk_end_message_id", sa.BigInteger(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), server_default="0", nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_conversation_summaries_chat_id", "conversation_summaries", ["chat_id"])

    op.create_table(
        "brief_cache",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("chat_id", sa.BigInteger(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("snapshot_message_id", sa.BigInteger(), nullable=False),
        sa.Column("brief_json", postgresql.JSONB(), nullable=False),
        sa.Column("token_count", sa.Integer(), server_default="0", nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_brief_cache_chat_id", "brief_cache", ["chat_id"])

    op.create_table(
        "ai_decisions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("chat_id", sa.BigInteger(), nullable=False),
        sa.Column("evaluated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("prompt_version", sa.Text(), nullable=False),
        sa.Column("snapshot_message_id", sa.BigInteger(), nullable=True),
        sa.Column("new_message_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("should_respond", sa.Boolean(), nullable=False),
        sa.Column("confidence", sa.Float(), server_default="0", nullable=False),
        sa.Column("response_text", sa.Text(), nullable=True),
        sa.Column("reply_to_message_id", sa.BigInteger(), nullable=True),
        sa.Column("sent_message_id", sa.BigInteger(), nullable=True),
        sa.Column("reasoning", sa.Text(), nullable=True),
        sa.Column("request1_latency_ms", sa.Integer(), nullable=True),
        sa.Column("request2_latency_ms", sa.Integer(), nullable=True),
        sa.Column("request1_tokens_used", sa.Integer(), nullable=True),
        sa.Column("request2_tokens_used", sa.Integer(), nullable=True),
        sa.Column("gate_score", sa.Float(), nullable=True),
        sa.Column("gate_factors", postgresql.JSONB(), server_default="{}", nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_ai_decisions_chat_id", "ai_decisions", ["chat_id"])

    op.create_table(
        "failed_cycles",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("chat_id", sa.BigInteger(), nullable=False),
        sa.Column("failed_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("stage", sa.Text(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=False),
        sa.Column("raw_context_sent", sa.Text(), nullable=True),
        sa.Column("prompt_version", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_failed_cycles_chat_id", "failed_cycles", ["chat_id"])

    op.create_table(
        "reply_distribution",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("chat_id", sa.BigInteger(), nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("reply_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("last_replied_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("chat_id", "user_id", name="uq_reply_distribution_chat_user"),
    )

    op.create_table(
        "stance_tracker",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("chat_id", sa.BigInteger(), nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=True),
        sa.Column("topic", sa.Text(), nullable=False),
        sa.Column("stance", sa.Text(), nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "prompt_versions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("version", sa.Text(), nullable=False),
        sa.Column("perception_system_prompt", sa.Text(), nullable=False),
        sa.Column("decision_system_prompt", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("version"),
    )

    op.create_table(
        "circuit_breaker_state",
        sa.Column("chat_id", sa.BigInteger(), nullable=False),
        sa.Column("failure_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("last_failure_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("paused_until", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("chat_id"),
    )

    op.create_table(
        "bot_persona_core",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("identity_summary", sa.Text(), nullable=False),
        sa.Column("core_beliefs", postgresql.JSONB(), nullable=False),
        sa.Column("speaking_style", sa.Text(), nullable=False),
        _vector_column("embedding"),
        sa.Column("version", sa.Integer(), server_default="1", nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "bot_vector_memories",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("chat_id", sa.BigInteger(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("memory_type", sa.Text(), nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        _vector_column("embedding"),
        sa.Column("importance_score", sa.Float(), server_default="0", nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_bot_vector_memories_chat_id", "bot_vector_memories", ["chat_id"])

    op.create_table(
        "bot_self_reflections",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("chat_id", sa.BigInteger(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("trigger", sa.Text(), nullable=False),
        sa.Column("messages_since_last", sa.Integer(), server_default="0", nullable=False),
        sa.Column("reflection_text", sa.Text(), nullable=False),
        sa.Column("updated_summary", sa.Text(), nullable=False),
        _vector_column("embedding"),
        sa.Column("drift_score", sa.Float(), server_default="0", nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_bot_self_reflections_chat_id", "bot_self_reflections", ["chat_id"])

    op.create_table(
        "user_relationship_profiles",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("chat_id", sa.BigInteger(), nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("first_interaction_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("last_interaction_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("total_exchanges", sa.Integer(), server_default="0", nullable=False),
        sa.Column("relationship_strength", sa.Float(), server_default="0", nullable=False),
        sa.Column("sentiment_trend", sa.Float(), server_default="0", nullable=False),
        sa.Column("receptiveness_score", sa.Float(), server_default="0", nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        _vector_column("embedding"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("chat_id", "user_id", name="uq_relationship_chat_user"),
    )

    op.create_table(
        "response_feedback",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("chat_id", sa.BigInteger(), nullable=False),
        sa.Column("bot_memory_id", sa.Integer(), nullable=False),
        sa.Column("sent_message_id", sa.BigInteger(), nullable=False),
        sa.Column("observation_window_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("reply_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("reaction_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("reaction_types", postgresql.JSONB(), server_default="[]", nullable=True),
        sa.Column("quote_reply_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("follow_up_sentiment", sa.Float(), server_default="0", nullable=False),
        sa.Column("outcome", sa.Text(), nullable=False),
        sa.Column("outcome_score", sa.Float(), server_default="0", nullable=False),
        sa.Column("scored_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("meta_reflected", sa.Boolean(), server_default="false", nullable=False),
        sa.ForeignKeyConstraint(["bot_memory_id"], ["bot_memory.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_response_feedback_chat_id", "response_feedback", ["chat_id"])

    op.create_table(
        "chat_activity_patterns",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("chat_id", sa.BigInteger(), nullable=False),
        sa.Column("hour_of_day", sa.Integer(), nullable=False),
        sa.Column("day_of_week", sa.Integer(), nullable=False),
        sa.Column("avg_message_velocity", sa.Float(), server_default="0", nullable=False),
        sa.Column("avg_tension", sa.Float(), server_default="0", nullable=False),
        sa.Column("sample_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("last_updated", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("chat_id", "hour_of_day", "day_of_week", name="uq_activity_chat_hour_day"),
    )

    op.execute(
        "CREATE INDEX ix_bot_vector_memories_embedding ON bot_vector_memories "
        "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
    )
    op.execute(
        "CREATE INDEX ix_bot_self_reflections_embedding ON bot_self_reflections "
        "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
    )
    op.execute(
        "CREATE INDEX ix_user_relationship_profiles_embedding ON user_relationship_profiles "
        "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
    )


def downgrade():
    op.execute("DROP INDEX IF EXISTS ix_user_relationship_profiles_embedding")
    op.execute("DROP INDEX IF EXISTS ix_bot_self_reflections_embedding")
    op.execute("DROP INDEX IF EXISTS ix_bot_vector_memories_embedding")
    op.drop_table("chat_activity_patterns")
    op.drop_index("ix_response_feedback_chat_id", table_name="response_feedback")
    op.drop_table("response_feedback")
    op.drop_table("user_relationship_profiles")
    op.drop_index("ix_bot_self_reflections_chat_id", table_name="bot_self_reflections")
    op.drop_table("bot_self_reflections")
    op.drop_index("ix_bot_vector_memories_chat_id", table_name="bot_vector_memories")
    op.drop_table("bot_vector_memories")
    op.drop_table("bot_persona_core")
    op.drop_table("circuit_breaker_state")
    op.drop_table("prompt_versions")
    op.drop_table("stance_tracker")
    op.drop_table("reply_distribution")
    op.drop_index("ix_failed_cycles_chat_id", table_name="failed_cycles")
    op.drop_table("failed_cycles")
    op.drop_index("ix_ai_decisions_chat_id", table_name="ai_decisions")
    op.drop_table("ai_decisions")
    op.drop_index("ix_brief_cache_chat_id", table_name="brief_cache")
    op.drop_table("brief_cache")
    op.drop_index("ix_conversation_summaries_chat_id", table_name="conversation_summaries")
    op.drop_table("conversation_summaries")
    op.drop_index("ix_bot_memory_chat_sent_at", table_name="bot_memory")
    op.drop_index("ix_bot_memory_chat_id", table_name="bot_memory")
    op.drop_table("bot_memory")
