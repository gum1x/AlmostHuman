"""add pending_observations table for DB-backed due-at feedback

Revision ID: 006
Revises: 005
Create Date: 2026-06-13
"""

from alembic import op
import sqlalchemy as sa

revision = "006"
down_revision = "005"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "pending_observations",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("chat_id", sa.BigInteger(), nullable=False),
        sa.Column("bot_memory_id", sa.Integer(), nullable=False),
        sa.Column("sent_message_id", sa.BigInteger(), nullable=False),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("due_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_pending_observations_chat_id", "pending_observations", ["chat_id"])
    op.create_index("ix_pending_observations_due_at", "pending_observations", ["due_at"])


def downgrade():
    op.drop_index("ix_pending_observations_due_at", table_name="pending_observations")
    op.drop_index("ix_pending_observations_chat_id", table_name="pending_observations")
    op.drop_table("pending_observations")
