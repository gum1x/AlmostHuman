"""Unit tests for the reaction_update worker handler and repository update."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pipeline.workers as workers
from core.constants import EventType
from core.schemas import RawTelegramEvent
from pipeline.workers import MessageWorker
from storage.repositories import MessageRepository


class FakeSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def begin(self):
        return self


class RepoRecorder:
    calls: list[dict] = []

    def __init__(self, session):
        self.session = session

    async def apply_reactions(self, **kwargs):
        RepoRecorder.calls.append(kwargs)


def _event(reactions: list[dict]) -> RawTelegramEvent:
    return RawTelegramEvent(
        event_type=EventType.REACTION_UPDATE,
        message_id=42,
        chat_id=-100123,
        timestamp=datetime.now(timezone.utc),
        reactions=reactions,
    )


def _patch(monkeypatch):
    RepoRecorder.calls = []
    monkeypatch.setattr(workers, "async_session_factory", lambda: FakeSession())
    monkeypatch.setattr(workers, "MessageRepository", RepoRecorder)


async def test_new_reactions_persisted(monkeypatch):
    _patch(monkeypatch)
    worker = MessageWorker()

    snapshot = [{"emoji": "🔥", "count": 2}, {"emoji": "😂", "count": 1}]
    await worker.process(_event(snapshot))

    assert RepoRecorder.calls == [{
        "chat_id": -100123,
        "message_id": 42,
        "reactions": snapshot,
        "reaction_count": 3,
    }]


async def test_mixed_reaction_shapes_all_counted(monkeypatch):
    _patch(monkeypatch)
    worker = MessageWorker()

    snapshot = [
        {"emoji": "🔥", "count": 2},
        {"custom_emoji_id": "999", "count": 5},
        {"paid": True, "count": 1},
        {"emoji": None, "count": 4},
    ]
    await worker.process(_event(snapshot))

    assert RepoRecorder.calls == [{
        "chat_id": -100123,
        "message_id": 42,
        "reactions": snapshot,
        "reaction_count": 12,
    }]


async def test_updated_counts_replace_snapshot(monkeypatch):
    _patch(monkeypatch)
    worker = MessageWorker()

    await worker.process(_event([{"emoji": "🔥", "count": 2}]))
    await worker.process(_event([{"emoji": "🔥", "count": 5}]))

    assert len(RepoRecorder.calls) == 2
    assert RepoRecorder.calls[1]["reactions"] == [{"emoji": "🔥", "count": 5}]
    assert RepoRecorder.calls[1]["reaction_count"] == 5


async def test_cleared_reactions(monkeypatch):
    _patch(monkeypatch)
    worker = MessageWorker()

    await worker.process(_event([]))

    assert RepoRecorder.calls == [{
        "chat_id": -100123,
        "message_id": 42,
        "reactions": [],
        "reaction_count": 0,
    }]


async def test_apply_reactions_missing_row_is_silent():
    # apply_reactions issues a plain UPDATE: zero matched rows is a no-op,
    # mirroring how out-of-order edits/deletes are handled.
    session = MagicMock()
    session.execute = AsyncMock(return_value=MagicMock(rowcount=0))
    session.flush = AsyncMock()

    repo = MessageRepository(session)
    await repo.apply_reactions(
        chat_id=-100123, message_id=42,
        reactions=[{"emoji": "🔥", "count": 1}], reaction_count=1,
    )

    assert session.execute.await_count == 1
    stmt = session.execute.await_args.args[0]
    assert str(stmt).startswith("UPDATE messages")
