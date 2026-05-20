"""initial schema

Revision ID: 001
Revises:
Create Date: 2026-05-20
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "chats",
        sa.Column("chat_id", sa.BigInteger(), nullable=False),
        sa.Column("title", sa.String(255), nullable=True),
        sa.Column("username", sa.String(255), nullable=True),
        sa.Column("chat_type", sa.String(20), nullable=True),
        sa.Column("member_count", sa.Integer(), nullable=True),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("last_message_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), server_default="{}"),
        sa.PrimaryKeyConstraint("chat_id"),
    )

    op.create_table(
        "senders",
        sa.Column("sender_id", sa.BigInteger(), nullable=False),
        sa.Column("username", sa.String(255), nullable=True),
        sa.Column("first_name", sa.String(255), nullable=True),
        sa.Column("last_name", sa.String(255), nullable=True),
        sa.Column("is_bot", sa.Boolean(), server_default="false"),
        sa.Column("is_premium", sa.Boolean(), server_default="false"),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), server_default="{}"),
        sa.PrimaryKeyConstraint("sender_id"),
    )
    op.create_index(
        "ix_senders_username", "senders", ["username"],
        postgresql_where=sa.text("username IS NOT NULL"),
    )

    op.create_table(
        "chat_members",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("chat_id", sa.BigInteger(), nullable=False),
        sa.Column("sender_id", sa.BigInteger(), nullable=False),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("message_count", sa.Integer(), server_default="0"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("chat_id", "sender_id", name="uq_chat_sender"),
    )

    op.create_table(
        "messages",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("message_id", sa.Integer(), nullable=False),
        sa.Column("chat_id", sa.BigInteger(), nullable=False),
        sa.Column("sender_id", sa.BigInteger(), nullable=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("message_type", sa.String(20), nullable=False, server_default="text"),
        sa.Column("text_raw", sa.Text(), nullable=True),
        sa.Column("text_cleaned", sa.Text(), nullable=True),
        sa.Column("reply_to_message_id", sa.Integer(), nullable=True),
        sa.Column("forward_from_id", sa.BigInteger(), nullable=True),
        sa.Column("forward_from_chat_id", sa.BigInteger(), nullable=True),
        sa.Column("forward_from_message_id", sa.Integer(), nullable=True),
        sa.Column("forward_from_name", sa.String(255), nullable=True),
        sa.Column("forward_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("media_type", sa.String(30), nullable=True),
        sa.Column("media_file_id", sa.String(255), nullable=True),
        sa.Column("media_file_size", sa.BigInteger(), nullable=True),
        sa.Column("media_mime_type", sa.String(100), nullable=True),
        sa.Column("media_duration", sa.Integer(), nullable=True),
        sa.Column("media_width", sa.Integer(), nullable=True),
        sa.Column("media_height", sa.Integer(), nullable=True),
        sa.Column("mention_list", postgresql.JSONB(), server_default="[]"),
        sa.Column("entity_list", postgresql.JSONB(), server_default="[]"),
        sa.Column("edit_history", postgresql.JSONB(), server_default="[]"),
        sa.Column("grouped_id", sa.BigInteger(), nullable=True),
        sa.Column("is_deleted", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("raw_event", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("chat_id", "message_id", name="uq_chat_message"),
    )

    op.create_index("ix_messages_chat_id_timestamp", "messages", ["chat_id", sa.text("timestamp DESC")])
    op.create_index("ix_messages_sender_id_timestamp", "messages", ["sender_id", sa.text("timestamp DESC")])
    op.create_index(
        "ix_messages_chat_id_reply_to", "messages", ["chat_id", "reply_to_message_id"],
        postgresql_where=sa.text("reply_to_message_id IS NOT NULL"),
    )
    op.create_index(
        "ix_messages_grouped_id", "messages", ["grouped_id"],
        postgresql_where=sa.text("grouped_id IS NOT NULL"),
    )
    op.create_index(
        "ix_messages_active", "messages", ["chat_id", sa.text("timestamp DESC")],
        postgresql_where=sa.text("is_deleted = false"),
    )

    op.create_table(
        "deletion_events",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("chat_id", sa.BigInteger(), nullable=False),
        sa.Column("message_ids", postgresql.ARRAY(sa.Integer()), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    op.execute(
        "CREATE OR REPLACE FUNCTION update_updated_at() RETURNS TRIGGER AS $$ "
        "BEGIN NEW.updated_at = NOW(); RETURN NEW; END; $$ LANGUAGE plpgsql"
    )
    op.execute(
        "CREATE TRIGGER trg_messages_updated_at BEFORE UPDATE ON messages "
        "FOR EACH ROW EXECUTE FUNCTION update_updated_at()"
    )


def downgrade():
    op.execute("DROP TRIGGER IF EXISTS trg_messages_updated_at ON messages")
    op.execute("DROP FUNCTION IF EXISTS update_updated_at()")
    op.drop_table("deletion_events")
    op.drop_table("messages")
    op.drop_table("chat_members")
    op.drop_table("senders")
    op.drop_table("chats")
