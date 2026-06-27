"""drop the three unused ivfflat (vector_cosine_ops) embedding indexes

None of the ivfflat embedding indexes created in 002 are usable by any query in
the engine, so they are pure write-side overhead (ivfflat maintenance on every
insert/update) with zero read benefit:

* ``bot_vector_memories.embedding`` IS queried (ConversationMemoryManager
  .get_relevant_vector_memories), but only via a recency-decayed *composite*
  score (similarity * importance * exp(-decay)) -- an ``ORDER BY <expr> DESC``
  that ivfflat cannot accelerate (ivfflat only serves a pure
  ``ORDER BY embedding <=> q LIMIT k``). The ``WHERE chat_id`` scan is served by
  ``ix_bot_vector_memories_chat_id``, so dropping the ivfflat index does not
  change the read plan.
* ``bot_self_reflections.embedding`` and ``user_relationship_profiles.embedding``
  are written but never similarity-queried anywhere in the codebase, so their
  ivfflat indexes serve nothing.

Dropping all three is a no-op for reads. The downgrade recreates them exactly as
002_conversation_engine did.

Revision ID: 008
Revises: 007
Create Date: 2026-06-27
"""

from alembic import op

revision = "008"
down_revision = "007"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("DROP INDEX IF EXISTS ix_user_relationship_profiles_embedding")
    op.execute("DROP INDEX IF EXISTS ix_bot_self_reflections_embedding")
    op.execute("DROP INDEX IF EXISTS ix_bot_vector_memories_embedding")


def downgrade():
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
