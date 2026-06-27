"""Poison-event dead-lettering in the queue consumer.

A handler that always raises must be retried up to the delivery cap, after
which the entry is moved to the dead-letter stream and acked so the autoclaim
loop stops re-delivering it forever. Fully offline against a minimal fake Redis
that models consumer-group pending/delivery-count semantics.
"""

from datetime import datetime, timezone

import orjson
import pytest

from core.config import settings
from core.constants import EventType
from core.schemas import RawTelegramEvent
from pipeline.queue_consumer import QueueConsumer


class FakeRedis:
    """Just enough Redis stream + consumer-group behavior for the consumer.

    Tracks a per-entry pending list with a delivery count that increments on
    every (re)delivery, exactly like Redis bumps `times_delivered` on
    XREADGROUP '>' and XAUTOCLAIM.
    """

    def __init__(self):
        self.streams: dict[bytes, list[tuple[bytes, dict]]] = {}
        # stream -> {entry_id: {"fields": dict, "delivered": int}}
        self.pending: dict[bytes, dict[bytes, dict]] = {}
        self._seq = 0

    @staticmethod
    def _b(v):
        return v if isinstance(v, bytes) else str(v).encode()

    def seed(self, stream: bytes, fields: dict) -> bytes:
        self._seq += 1
        entry_id = f"{self._seq}-0".encode()
        self.streams.setdefault(stream, []).append((entry_id, fields))
        return entry_id

    async def xadd(self, stream, fields, **kwargs):
        stream = self._b(stream)
        self._seq += 1
        entry_id = f"{self._seq}-0".encode()
        self.streams.setdefault(stream, []).append((entry_id, dict(fields)))
        return entry_id

    async def xreadgroup(self, group, consumer, streams, count=None, block=None):
        out = []
        for stream_key in streams:
            stream = self._b(stream_key)
            pel = self.pending.setdefault(stream, {})
            delivered_ids = {eid for eid, _ in self.streams.get(stream, [])} & set(pel)
            messages = []
            for entry_id, fields in self.streams.get(stream, []):
                if entry_id in delivered_ids:
                    continue  # only never-before-delivered entries for '>'
                pel[entry_id] = {"fields": fields, "delivered": 1}
                messages.append((entry_id, fields))
            out.append((stream, messages))
        return out

    async def xack(self, stream, group, entry_id):
        stream = self._b(stream)
        self.pending.get(stream, {}).pop(self._b(entry_id), None)
        return 1

    async def xautoclaim(self, stream, group, consumer, min_idle_time, start_id="0-0", count=None):
        stream = self._b(stream)
        pel = self.pending.get(stream, {})
        claimed = []
        for entry_id, meta in pel.items():
            meta["delivered"] += 1  # redelivery bumps the count
            claimed.append((entry_id, meta["fields"]))
        return (b"0-0", claimed, [])

    async def xpending_range(self, stream, group, min, max, count, **kwargs):
        stream = self._b(stream)
        meta = self.pending.get(stream, {}).get(self._b(min))
        if meta is None:
            return []
        return [
            {
                "message_id": self._b(min),
                "consumer": b"c",
                "time_since_delivered": 0,
                "times_delivered": meta["delivered"],
            }
        ]


def _poison_payload() -> dict:
    event = RawTelegramEvent(
        event_type=EventType.NEW_MESSAGE,
        message_id=1,
        chat_id=-100,
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    return {b"payload": orjson.dumps(event.model_dump(mode="json"))}


class _AlwaysRaises:
    async def process(self, event):
        raise RuntimeError("poison")


@pytest.fixture
def consumer():
    c = QueueConsumer()
    c._redis = FakeRedis()
    c._worker = _AlwaysRaises()
    return c


async def test_poison_event_is_dead_lettered_after_cap(consumer):
    fake = consumer._redis
    stream = consumer._stream_key.encode()
    dlq = consumer._dlq_stream_key.encode()
    cap = consumer._dlq_max_delivery

    # First delivery via XREADGROUP '>' (delivered count -> 1).
    entry_id = fake.seed(stream, _poison_payload())
    delivered = await fake.xreadgroup(consumer._group, consumer._consumer_name, {stream: ">"})
    await consumer._process_batch(delivered[0][1])

    # It raised, so it must still be pending and NOT yet dead-lettered.
    assert entry_id in fake.pending[stream]
    assert dlq not in fake.streams

    # Subsequent redeliveries via the autoclaim loop, until we hit the cap.
    safety = 0
    while entry_id in fake.pending[stream] and safety < 50:
        safety += 1
        _, claimed, _ = await fake.xautoclaim(
            stream, consumer._group, consumer._consumer_name, min_idle_time=0
        )
        await consumer._process_batch(claimed)

    # Once the cap is reached: acked (no longer pending) and moved to the DLQ.
    assert entry_id not in fake.pending.get(stream, {})
    assert dlq in fake.streams and len(fake.streams[dlq]) == 1

    _, dlq_fields = fake.streams[dlq][0]
    assert dlq_fields[b"payload"] == _poison_payload()[b"payload"]
    assert dlq_fields[b"original_id"] == entry_id
    assert int(dlq_fields[b"attempts"]) >= cap
    assert b"poison" in dlq_fields[b"error"]


async def test_below_cap_entry_stays_pending_not_dlqd(consumer):
    """A single failure must not dead-letter — it stays pending for retry."""
    fake = consumer._redis
    stream = consumer._stream_key.encode()
    dlq = consumer._dlq_stream_key.encode()

    entry_id = fake.seed(stream, _poison_payload())
    delivered = await fake.xreadgroup(consumer._group, consumer._consumer_name, {stream: ">"})
    await consumer._process_batch(delivered[0][1])

    assert consumer._dlq_max_delivery > 1  # guard: the test needs a >1 cap
    assert entry_id in fake.pending[stream]
    assert dlq not in fake.streams


async def test_successful_processing_acks_and_no_dlq():
    """Happy path is unchanged: process, ack, nothing dead-lettered."""

    class _Ok:
        seen = []

        async def process(self, event):
            _Ok.seen.append(event.message_id)

    c = QueueConsumer()
    c._redis = FakeRedis()
    c._worker = _Ok()
    stream = c._stream_key.encode()
    dlq = c._dlq_stream_key.encode()

    entry_id = c._redis.seed(stream, _poison_payload())
    delivered = await c._redis.xreadgroup(c._group, c._consumer_name, {stream: ">"})
    await c._process_batch(delivered[0][1])

    assert _Ok.seen == [1]
    assert entry_id not in c._redis.pending.get(stream, {})
    assert dlq not in c._redis.streams


def test_dlq_stream_key_derives_from_stream_key():
    assert settings.redis_dlq_stream_key == f"{settings.redis_stream_key}:dlq"
