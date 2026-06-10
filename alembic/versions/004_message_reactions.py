"""add reactions snapshot to messages

Revision ID: 004
Revises: 003
Create Date: 2026-06-10
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("messages", sa.Column("reactions", postgresql.JSONB(), server_default="[]"))
    op.add_column("messages", sa.Column("reaction_count", sa.Integer(), server_default="0", nullable=False))


def downgrade():
    op.drop_column("messages", "reaction_count")
    op.drop_column("messages", "reactions")
