"""Integration coverage for storage/repositories.py against a real Postgres.

Drives the genuine ingestion write path (``MessageWorker.process`` ->
``MessageRepository.upsert_message`` inside a real transaction) and then reads
back through ``MessageRepository`` to assert pagination, soft-delete filtering,
and recursive-CTE thread reconstruction behave against actual SQL.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from core.constants import EventType
from core.schemas import RawTelegramEvent, SenderInfo
from pipeline.workers import MessageWorker
from storage.repositories import MessageRepository
from tests.integration.conftest import skip_if_no_docker

# No Docker/testcontainers -> skip the whole module cleanly (never a collect error).
skip_if_no_docker()

pytestmark = [pytest.mark.integration, pytest.mark.asyncio(loop_scope="session")]

CHAT_ID = -1001234567890
BASE_TS = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def _new_message_event(
    message_id: int,
    *,
    text: str = "hello",
    sender_id: int = 555,
    reply_to_message_id: int | None = None,
    ts_offset_seconds: int = 0,
) -> RawTelegramEvent:
    return RawTelegramEvent(
        event_type=EventType.NEW_MESSAGE,
        message_id=message_id,
        chat_id=CHAT_ID,
        sender_id=sender_id,
        timestamp=BASE_TS + timedelta(seconds=ts_offset_seconds),
        text=text,
        reply_to_message_id=reply_to_message_id,
        sender_info=SenderInfo(sender_id=sender_id, username=f"user{sender_id}"),
        raw={"chat_type": "supergroup"},
    )


def _delete_event(message_ids: list[int]) -> RawTelegramEvent:
    return RawTelegramEvent(
        event_type=EventType.DELETE,
        message_id=message_ids[0],
        chat_id=CHAT_ID,
        timestamp=BASE_TS,
        deleted_message_ids=message_ids,
    )


async def _ingest(worker: MessageWorker, events: list[RawTelegramEvent]) -> None:
    for event in events:
        await worker.process(event)


async def test_worker_upsert_persists_message_and_related_rows(db_session, session_factory):
    """The full worker path writes Message + Sender + Chat + ChatMember rows."""
    worker = MessageWorker()
    await _ingest(worker, [_new_message_event(1, text="gm degens")])

    repo = MessageRepository(db_session)
    messages, total = await repo.get_messages(CHAT_ID)
    assert total == 1
    assert len(messages) == 1
    msg = messages[0]
    assert msg.message_id == 1
    assert msg.chat_id == CHAT_ID
    assert msg.text_raw == "gm degens"
    assert msg.text_cleaned == "gm degens"
    assert msg.sender_id == 555
    assert msg.is_deleted is False


async def test_upsert_is_idempotent_on_conflict(db_session):
    """Re-ingesting the same (chat_id, message_id) updates text, not row count."""
    worker = MessageWorker()
    await _ingest(worker, [_new_message_event(7, text="first")])
    await _ingest(worker, [_new_message_event(7, text="edited via re-send")])

    repo = MessageRepository(db_session)
    messages, total = await repo.get_messages(CHAT_ID)
    assert total == 1
    assert messages[0].text_raw == "edited via re-send"


async def test_get_messages_pagination_orders_newest_first(db_session):
    """limit/offset page through timestamp-desc ordering deterministically."""
    worker = MessageWorker()
    await _ingest(
        worker,
        [_new_message_event(i, text=f"msg-{i}", ts_offset_seconds=i) for i in range(1, 11)],
    )

    repo = MessageRepository(db_session)

    page1, total = await repo.get_messages(CHAT_ID, limit=3, offset=0)
    assert total == 10
    assert [m.message_id for m in page1] == [10, 9, 8]

    page2, total2 = await repo.get_messages(CHAT_ID, limit=3, offset=3)
    assert total2 == 10
    assert [m.message_id for m in page2] == [7, 6, 5]

    # No overlap between adjacent pages.
    assert not ({m.message_id for m in page1} & {m.message_id for m in page2})


async def test_get_messages_include_deleted_toggle(db_session):
    """Soft-deleted rows are hidden by default and surfaced (with count) on demand."""
    worker = MessageWorker()
    await _ingest(
        worker,
        [
            _new_message_event(1, text="keep", ts_offset_seconds=1),
            _new_message_event(2, text="will delete", ts_offset_seconds=2),
            _new_message_event(3, text="keep too", ts_offset_seconds=3),
        ],
    )
    await _ingest(worker, [_delete_event([2])])

    repo = MessageRepository(db_session)

    visible, visible_total = await repo.get_messages(CHAT_ID)
    assert visible_total == 2
    assert {m.message_id for m in visible} == {1, 3}

    everything, all_total = await repo.get_messages(CHAT_ID, include_deleted=True)
    assert all_total == 3
    deleted = next(m for m in everything if m.message_id == 2)
    assert deleted.is_deleted is True
    assert deleted.deleted_at is not None


async def test_get_thread_walks_reply_chain(db_session):
    """get_thread reconstructs the full root->leaf reply chain in timestamp order."""
    worker = MessageWorker()
    await _ingest(
        worker,
        [
            _new_message_event(100, text="root", ts_offset_seconds=0),
            _new_message_event(101, text="reply-1", reply_to_message_id=100, ts_offset_seconds=1),
            _new_message_event(102, text="reply-2", reply_to_message_id=101, ts_offset_seconds=2),
            # An unrelated message in the same chat must NOT leak into the thread.
            _new_message_event(200, text="unrelated", ts_offset_seconds=3),
        ],
    )

    repo = MessageRepository(db_session)

    # Querying from any node in the chain returns the whole chain from the root.
    thread = await repo.get_thread(CHAT_ID, 102)
    assert [m.message_id for m in thread] == [100, 101, 102]
    assert all(m.message_id != 200 for m in thread)

    from_middle = await repo.get_thread(CHAT_ID, 101)
    assert [m.message_id for m in from_middle] == [100, 101, 102]


async def test_get_messages_empty_chat_returns_zero(db_session):
    repo = MessageRepository(db_session)
    messages, total = await repo.get_messages(CHAT_ID)
    assert messages == []
    assert total == 0
