"""drop the unused reply_distribution table

The reply_distribution table had no live readers or writers in the engine; the
model and all references were removed. This migration drops the table; the
downgrade recreates it with its original schema (mirrors 002_conversation_engine).

Revision ID: 007
Revises: 006
Create Date: 2026-06-26
"""

import sqlalchemy as sa

from alembic import op

revision = "007"
down_revision = "006"
branch_labels = None
depends_on = None


def upgrade():
    op.drop_table("reply_distribution")


def downgrade():
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
