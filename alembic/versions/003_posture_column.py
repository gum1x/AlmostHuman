"""add current_posture to bot_memory

Revision ID: 003
Revises: 002
Create Date: 2026-06-02
"""

from alembic import op
import sqlalchemy as sa

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("bot_memory", sa.Column("current_posture", sa.Text(), nullable=True))


def downgrade():
    op.drop_column("bot_memory", "current_posture")
