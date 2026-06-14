"""add per-person dossier/tone/aliases to relationship profiles

Revision ID: 005
Revises: 004
Create Date: 2026-06-12
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "005"
down_revision = "004"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "user_relationship_profiles",
        sa.Column("dossier", postgresql.JSONB(), server_default="{}", nullable=True),
    )
    op.add_column(
        "user_relationship_profiles",
        sa.Column("tone", postgresql.JSONB(), server_default="{}", nullable=True),
    )
    op.add_column(
        "user_relationship_profiles",
        sa.Column("aliases", postgresql.JSONB(), server_default="[]", nullable=True),
    )


def downgrade():
    op.drop_column("user_relationship_profiles", "aliases")
    op.drop_column("user_relationship_profiles", "tone")
    op.drop_column("user_relationship_profiles", "dossier")
