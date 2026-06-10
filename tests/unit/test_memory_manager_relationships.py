from __future__ import annotations

import pytest
from sqlalchemy.exc import IntegrityError

from conversation_engine.memory_manager import ConversationMemoryManager, merge_relationship_notes
from storage.postgres_models import UserRelationshipProfile


class FakeResult:
    def __init__(self, row):
        self._row = row

    def scalar_one_or_none(self):
        return self._row


class FakeSavepoint:
    """On exception, discards rows added inside the savepoint (rollback) and
    re-raises -- outer session state is preserved, like a real SAVEPOINT."""

    def __init__(self, session):
        self.session = session

    async def __aenter__(self):
        self._added_before = len(self.session.added)
        return self

    async def __aexit__(self, exc_type, *args):
        if exc_type is not None:
            del self.session.added[self._added_before:]
        return False


class FakeSession:
    def __init__(self, existing=None):
        self.existing = existing
        self.added = []

    async def execute(self, stmt):
        return FakeResult(self.existing)

    def add(self, row):
        self.added.append(row)

    async def flush(self):
        pass

    def begin_nested(self):
        return FakeSavepoint(self)


class RaceySession(FakeSession):
    """First select misses, the insert flush raises IntegrityError (a concurrent
    task won the first-insert race), then the re-select finds the winner's row."""

    def __init__(self, row):
        super().__init__(existing=None)
        self.row = row
        self.selects = 0
        self.insert_raised = False

    async def execute(self, stmt):
        self.selects += 1
        return FakeResult(None if self.selects == 1 else self.row)

    async def flush(self):
        if not self.insert_raised:
            self.insert_raised = True
            raise IntegrityError("INSERT", {}, Exception("uq_relationship_chat_user"))


def existing_profile(**overrides):
    row = UserRelationshipProfile(
        chat_id=-100,
        user_id=42,
        total_exchanges=3,
        relationship_strength=0.5,
        sentiment_trend=0.4,
        receptiveness_score=0.6,
        notes="met in chat",
        embedding=[0.1] * 384,
    )
    for key, value in overrides.items():
        setattr(row, key, value)
    return row


# --- merge_relationship_notes ---


def test_merge_notes_appends_distinct():
    assert merge_relationship_notes("a", "b") == "a\nb"


def test_merge_notes_dedupes():
    assert merge_relationship_notes("a\nb", "b") == "a\nb"


def test_merge_notes_from_empty():
    assert merge_relationship_notes(None, "first") == "first"


def test_merge_notes_caps_length_keeping_newest():
    existing = "\n".join(f"note {i} " + "x" * 50 for i in range(30))
    merged = merge_relationship_notes(existing, "newest note", max_length=200)
    assert len(merged) <= 200
    assert merged.endswith("newest note")


# --- upsert_user_relationship ---


async def test_upsert_conflict_preserves_unspecified_fields():
    row = existing_profile()
    memory = ConversationMemoryManager(FakeSession(existing=row))
    await memory.upsert_user_relationship(-100, 42, notes="Preferred tone: dry")
    assert row.sentiment_trend == 0.4
    assert row.receptiveness_score == 0.6
    assert row.embedding == [0.1] * 384
    assert row.total_exchanges == 3
    assert row.relationship_strength == 0.5


async def test_upsert_merges_notes_instead_of_clobbering():
    row = existing_profile(notes="likes to argue about L2s")
    memory = ConversationMemoryManager(FakeSession(existing=row))
    await memory.upsert_user_relationship(-100, 42, notes="Preferred tone: dry")
    assert "likes to argue about L2s" in row.notes
    assert "Preferred tone: dry" in row.notes


async def test_upsert_explicitly_provided_fields_are_updated():
    row = existing_profile()
    memory = ConversationMemoryManager(FakeSession(existing=row))
    await memory.upsert_user_relationship(-100, 42, sentiment_trend=-0.3, receptiveness_score=0.9)
    assert row.sentiment_trend == -0.3
    assert row.receptiveness_score == 0.9
    assert row.notes == "met in chat"


async def test_upsert_inserts_when_missing():
    session = FakeSession(existing=None)
    memory = ConversationMemoryManager(session)
    await memory.upsert_user_relationship(-100, 42, notes="new guy", sentiment_trend=0.2)
    [row] = session.added
    assert row.chat_id == -100
    assert row.user_id == 42
    assert row.notes == "new guy"
    assert row.sentiment_trend == 0.2


# --- record_user_exchange ---


async def test_record_exchange_increments_and_nudges_up():
    row = existing_profile()
    memory = ConversationMemoryManager(FakeSession(existing=row))
    await memory.record_user_exchange(-100, 42, outcome_score=0.6, reply_sentiment=1.0)
    assert row.total_exchanges == 4
    assert row.relationship_strength == pytest.approx(0.55)
    assert row.sentiment_trend == pytest.approx(0.7 * 0.4 + 0.3 * 1.0)


async def test_record_exchange_negative_outcome_nudges_down_bounded():
    row = existing_profile(relationship_strength=0.0, sentiment_trend=0.0)
    memory = ConversationMemoryManager(FakeSession(existing=row))
    await memory.record_user_exchange(-100, 42, outcome_score=-0.5, reply_sentiment=-0.8)
    assert row.total_exchanges == 4
    assert row.relationship_strength == 0.0
    assert row.sentiment_trend == pytest.approx(0.3 * -0.8)


async def test_record_exchange_strength_bounded_at_one():
    row = existing_profile(relationship_strength=0.99)
    memory = ConversationMemoryManager(FakeSession(existing=row))
    await memory.record_user_exchange(-100, 42, outcome_score=0.6, reply_sentiment=0.0)
    assert row.relationship_strength == 1.0


async def test_record_exchange_creates_profile():
    session = FakeSession(existing=None)
    memory = ConversationMemoryManager(session)
    await memory.record_user_exchange(-100, 42, outcome_score=0.6, reply_sentiment=0.5)
    [row] = session.added
    assert row.total_exchanges == 1
    assert row.sentiment_trend == 0.5
    assert row.relationship_strength == pytest.approx(0.15)


# --- concurrent first-insert race (IntegrityError on uq_relationship_chat_user) ---


async def test_record_exchange_insert_race_falls_back_to_update():
    winner = existing_profile()
    session = RaceySession(winner)
    outer_work = object()
    session.added.append(outer_work)  # e.g. the feedback row from observe_response
    memory = ConversationMemoryManager(session)
    await memory.record_user_exchange(-100, 42, outcome_score=0.6, reply_sentiment=1.0)
    # fallback applied the update path to the winner's row
    assert winner.total_exchanges == 4
    assert winner.relationship_strength == pytest.approx(0.55)
    assert winner.sentiment_trend == pytest.approx(0.7 * 0.4 + 0.3 * 1.0)
    # savepoint rollback discarded only the failed insert; outer work survived
    assert session.added == [outer_work]


async def test_upsert_insert_race_falls_back_to_update():
    winner = existing_profile()
    session = RaceySession(winner)
    outer_work = object()
    session.added.append(outer_work)
    memory = ConversationMemoryManager(session)
    await memory.upsert_user_relationship(-100, 42, notes="Preferred tone: dry")
    assert "met in chat" in winner.notes
    assert "Preferred tone: dry" in winner.notes
    assert session.added == [outer_work]
