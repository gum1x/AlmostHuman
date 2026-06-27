"""Integration coverage for ConversationMemoryManager against a real Postgres.

These are exactly the behaviours the unit suite has to fake:

* pgvector cosine-distance relevance retrieval and its recency-decayed ranking,
* the relationship first-insert race resolving through a SAVEPOINT
  (``begin_nested``) under the real ``uq_relationship_chat_user`` constraint
  instead of a hand-rolled ``IntegrityError`` stub,
* the circuit breaker's atomic ``ON CONFLICT ... CASE`` upsert flipping
  ``paused_until`` only once the failure threshold is crossed.
"""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import select

from conversation_engine.memory_manager import ConversationMemoryManager, utcnow
from storage.postgres_models import CircuitBreakerState, UserRelationshipProfile

# Docker-gating lives in tests/integration/conftest.py's pytest_collection_modifyitems
# hook: with no Docker daemon, every integration-marked test is skipped at collection.
pytestmark = [pytest.mark.integration, pytest.mark.asyncio(loop_scope="session")]

CHAT_ID = -100777


def _unit(*nonzero_dims: int, dim: int = 384) -> list[float]:
    """A 384-dim vector that is 1.0 in the given dims, 0 elsewhere.

    Distinct one-hot-ish vectors give predictable cosine ordering without
    depending on a real embedding model.
    """
    vec = [0.0] * dim
    for d in nonzero_dims:
        vec[d] = 1.0
    return vec


# --------------------------------------------------------------------------- #
# pgvector relevance retrieval
# --------------------------------------------------------------------------- #


async def test_vector_memory_relevance_orders_by_similarity(db_session):
    """Closest-by-cosine memory ranks first; an orthogonal one ranks last."""
    mem = ConversationMemoryManager(db_session)

    # query points along dim 0. near := mostly dim 0, mid := overlaps dim 0,
    # far := orthogonal (dims 5/6, zero overlap with the query).
    await mem.write_vector_memory(CHAT_ID, "fact", "near", _unit(0), importance_score=0.9)
    await mem.write_vector_memory(CHAT_ID, "fact", "mid", _unit(0, 1), importance_score=0.9)
    await mem.write_vector_memory(CHAT_ID, "fact", "far", _unit(5, 6), importance_score=0.9)
    await db_session.commit()

    results = await mem.get_relevant_vector_memories(CHAT_ID, _unit(0), top_k=3)
    contents = [r.content for r in results]

    assert contents[0] == "near"
    assert "far" not in contents[:2]
    # similarity is monotonically non-increasing in rank order.
    sims = [r.similarity for r in results]
    assert sims == sorted(sims, reverse=True)
    assert results[0].similarity > results[-1].similarity


async def test_vector_memory_scoped_per_chat(db_session):
    """Retrieval never crosses chat boundaries."""
    mem = ConversationMemoryManager(db_session)
    await mem.write_vector_memory(CHAT_ID, "fact", "mine", _unit(0), importance_score=0.5)
    await mem.write_vector_memory(999000, "fact", "theirs", _unit(0), importance_score=0.5)
    await db_session.commit()

    results = await mem.get_relevant_vector_memories(CHAT_ID, _unit(0), top_k=10)
    assert [r.content for r in results] == ["mine"]


async def test_vector_memory_no_query_falls_back_to_importance(db_session):
    """With no query embedding, ranking falls back to importance then recency."""
    mem = ConversationMemoryManager(db_session)
    await mem.write_vector_memory(CHAT_ID, "fact", "low", _unit(0), importance_score=0.1)
    await mem.write_vector_memory(CHAT_ID, "fact", "high", _unit(1), importance_score=0.9)
    await db_session.commit()

    results = await mem.get_relevant_vector_memories(CHAT_ID, None, top_k=2)
    assert [r.content for r in results] == ["high", "low"]


# --------------------------------------------------------------------------- #
# relationship first-insert race (SAVEPOINT under the real unique constraint)
# --------------------------------------------------------------------------- #


async def test_relationship_insert_race_resolves_via_savepoint(session_factory):
    """Two concurrent upserts for the same (chat_id, user_id) collide on the
    unique index; the loser rolls back its failed INSERT through the SAVEPOINT,
    re-reads the winner's row and merges into it -- no IntegrityError escapes,
    and exactly one row exists with both notes merged."""
    user_id = 4242

    async def upsert(note: str) -> None:
        async with session_factory() as session:
            mem = ConversationMemoryManager(session)
            await mem.upsert_user_relationship(CHAT_ID, user_id, notes=note)
            await session.commit()

    # Run them concurrently; whichever loses the INSERT race must recover.
    await asyncio.gather(upsert("note-A"), upsert("note-B"))

    async with session_factory() as session:
        rows = (
            (
                await session.execute(
                    select(UserRelationshipProfile).where(
                        UserRelationshipProfile.chat_id == CHAT_ID,
                        UserRelationshipProfile.user_id == user_id,
                    )
                )
            )
            .scalars()
            .all()
        )

    assert len(rows) == 1, "unique constraint must collapse the race to one row"
    notes = rows[0].notes or ""
    # The loser merged its note into the winner's row, so both survive.
    assert "note-A" in notes
    assert "note-B" in notes


async def test_relationship_race_preserves_outer_transaction_work(session_factory):
    """The SAVEPOINT only rolls back the failed insert: a row the caller added in
    the same outer transaction before the losing upsert must still commit."""
    user_id = 9191

    # Pre-seed the row so the second caller is guaranteed to lose the insert race
    # the moment it tries (its nested INSERT violates the unique constraint).
    async with session_factory() as seed:
        await ConversationMemoryManager(seed).upsert_user_relationship(
            CHAT_ID, user_id, notes="seed"
        )
        await seed.commit()

    async with session_factory() as session:
        # Outer-transaction work that must survive the nested-insert rollback.
        session.add(UserRelationshipProfile(chat_id=CHAT_ID, user_id=user_id + 1, notes="sibling"))
        mem = ConversationMemoryManager(session)
        await mem.upsert_user_relationship(CHAT_ID, user_id, notes="merged-in")
        await session.commit()

    async with session_factory() as check:
        target = (
            await check.execute(
                select(UserRelationshipProfile).where(
                    UserRelationshipProfile.chat_id == CHAT_ID,
                    UserRelationshipProfile.user_id == user_id,
                )
            )
        ).scalar_one()
        sibling = (
            await check.execute(
                select(UserRelationshipProfile).where(
                    UserRelationshipProfile.chat_id == CHAT_ID,
                    UserRelationshipProfile.user_id == user_id + 1,
                )
            )
        ).scalar_one()

    assert "seed" in (target.notes or "")
    assert "merged-in" in (target.notes or "")
    assert sibling.notes == "sibling"


async def test_record_user_exchange_concurrent_first_insert(session_factory):
    """record_user_exchange's first-insert race path also resolves to one row
    with the exchanges accumulated across both callers."""
    user_id = 7373

    async def exchange() -> None:
        async with session_factory() as session:
            await ConversationMemoryManager(session).record_user_exchange(
                CHAT_ID, user_id, outcome_score=0.6, reply_sentiment=0.5
            )
            await session.commit()

    await asyncio.gather(exchange(), exchange())

    async with session_factory() as session:
        rows = (
            (
                await session.execute(
                    select(UserRelationshipProfile).where(
                        UserRelationshipProfile.chat_id == CHAT_ID,
                        UserRelationshipProfile.user_id == user_id,
                    )
                )
            )
            .scalars()
            .all()
        )

    assert len(rows) == 1
    # One insert (total_exchanges=1) + one update path (+1) == 2.
    assert rows[0].total_exchanges == 2


# --------------------------------------------------------------------------- #
# circuit breaker atomic CASE upsert
# --------------------------------------------------------------------------- #


async def test_circuit_breaker_pauses_after_threshold(db_session):
    """record_cycle_failure increments atomically; paused_until stays NULL until
    the failure count reaches the threshold, then is set to NOW()+pause."""
    mem = ConversationMemoryManager(db_session)
    threshold = 3
    pause_minutes = 15

    # Below threshold: failures accumulate but the breaker stays open (not paused).
    for _ in range(threshold - 1):
        await mem.record_cycle_failure(CHAT_ID, threshold, pause_minutes)
        await db_session.commit()

    # record_cycle_failure issues a Core INSERT...ON CONFLICT, which bypasses the
    # ORM identity map; drop any cached instance so reads see committed state
    # (production calls is_circuit_paused on a fresh per-cycle session anyway).
    db_session.expire_all()
    assert await mem.is_circuit_paused(CHAT_ID) is False
    state = (
        await db_session.execute(
            select(CircuitBreakerState).where(CircuitBreakerState.chat_id == CHAT_ID)
        )
    ).scalar_one()
    assert state.failure_count == threshold - 1
    assert state.paused_until is None

    # The threshold-crossing failure flips paused_until into the future.
    before = utcnow()
    await mem.record_cycle_failure(CHAT_ID, threshold, pause_minutes)
    await db_session.commit()
    db_session.expire_all()

    assert await mem.is_circuit_paused(CHAT_ID) is True
    state = (
        await db_session.execute(
            select(CircuitBreakerState).where(CircuitBreakerState.chat_id == CHAT_ID)
        )
    ).scalar_one()
    assert state.failure_count == threshold
    assert state.paused_until is not None
    assert state.paused_until > before


async def test_circuit_breaker_success_resets_state(db_session):
    """record_cycle_success clears the failure count and unpauses."""
    mem = ConversationMemoryManager(db_session)
    for _ in range(3):
        await mem.record_cycle_failure(CHAT_ID, failure_threshold=3, pause_minutes=15)
        await db_session.commit()
    db_session.expire_all()  # Core upsert bypasses the identity map; see test above.
    assert await mem.is_circuit_paused(CHAT_ID) is True

    await mem.record_cycle_success(CHAT_ID)
    await db_session.commit()
    db_session.expire_all()

    assert await mem.is_circuit_paused(CHAT_ID) is False
    state = (
        await db_session.execute(
            select(CircuitBreakerState).where(CircuitBreakerState.chat_id == CHAT_ID)
        )
    ).scalar_one()
    assert state.failure_count == 0
    assert state.paused_until is None
