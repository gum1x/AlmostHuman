"""Unit tests for reaction-update extraction and production in ingestion/event_handlers.py."""

from telethon import types, utils

from core.constants import EventType
from ingestion.event_handlers import _extract_reactions, _produce_reaction_update


class FakeProducer:
    def __init__(self, fail: bool = False):
        self.events = []
        self.fail = fail

    async def produce(self, event):
        if self.fail:
            raise RuntimeError("redis down")
        self.events.append(event)


def _update(peer, msg_id=42, results=None):
    return types.UpdateMessageReactions(
        peer=peer,
        msg_id=msg_id,
        reactions=types.MessageReactions(results=results or []),
    )


def test_extract_reactions_plain_emoji():
    reactions = types.MessageReactions(results=[
        types.ReactionCount(reaction=types.ReactionEmoji(emoticon="🔥"), count=3),
        types.ReactionCount(reaction=types.ReactionEmoji(emoticon="😂"), count=1),
    ])
    out, skipped = _extract_reactions(reactions)
    assert out == [{"emoji": "🔥", "count": 3}, {"emoji": "😂", "count": 1}]
    assert skipped == 0


def test_extract_reactions_includes_custom_and_paid():
    reactions = types.MessageReactions(results=[
        types.ReactionCount(reaction=types.ReactionEmoji(emoticon="🔥"), count=2),
        types.ReactionCount(reaction=types.ReactionCustomEmoji(document_id=999), count=5),
        types.ReactionCount(reaction=types.ReactionPaid(), count=1),
    ])
    out, skipped = _extract_reactions(reactions)
    assert out == [
        {"emoji": "🔥", "count": 2},
        {"custom_emoji_id": "999", "count": 5},
        {"paid": True, "count": 1},
    ]
    assert skipped == 0


def test_extract_reactions_unknown_shape_with_count():
    reactions = types.MessageReactions(results=[
        types.ReactionCount(reaction=types.ReactionEmpty(), count=4),
    ])
    out, skipped = _extract_reactions(reactions)
    assert out == [{"emoji": None, "count": 4}]
    assert skipped == 0


def test_extract_reactions_skips_entry_without_count():
    reactions = types.MessageReactions(results=[
        types.ReactionCount(reaction=types.ReactionEmoji(emoticon="🔥"), count=2),
        types.ReactionCount(reaction=types.ReactionEmoji(emoticon="😂"), count=None),
    ])
    out, skipped = _extract_reactions(reactions)
    assert out == [{"emoji": "🔥", "count": 2}]
    assert skipped == 1


def test_extract_reactions_cleared():
    out, skipped = _extract_reactions(types.MessageReactions(results=[]))
    assert out == []
    assert skipped == 0


def test_extract_reactions_unrecognized_object():
    out, skipped = _extract_reactions(None)
    assert out == []
    assert skipped == 0


async def test_produce_reaction_update_produces_event():
    producer = FakeProducer()
    update = _update(
        peer=types.PeerChannel(channel_id=123),
        msg_id=42,
        results=[types.ReactionCount(reaction=types.ReactionEmoji(emoticon="🔥"), count=3)],
    )

    await _produce_reaction_update(update, producer, None)

    assert len(producer.events) == 1
    event = producer.events[0]
    assert event.event_type == EventType.REACTION_UPDATE
    assert event.chat_id == utils.get_peer_id(types.PeerChannel(channel_id=123))
    assert event.message_id == 42
    assert event.reactions == [{"emoji": "🔥", "count": 3}]
    assert event.raw["reactions"] == [{"emoji": "🔥", "count": 3}]


async def test_produce_reaction_update_cleared_reactions():
    producer = FakeProducer()
    update = _update(peer=types.PeerChannel(channel_id=123), results=[])

    await _produce_reaction_update(update, producer, None)

    assert len(producer.events) == 1
    assert producer.events[0].reactions == []


async def test_produce_reaction_update_respects_chat_filter():
    producer = FakeProducer()
    update = _update(peer=types.PeerChannel(channel_id=123))

    await _produce_reaction_update(update, producer, [999])

    assert producer.events == []


async def test_produce_reaction_update_unrecognized_peer():
    producer = FakeProducer()
    update = _update(peer=None)

    await _produce_reaction_update(update, producer, None)

    assert producer.events == []


async def test_produce_reaction_update_producer_error_swallowed():
    producer = FakeProducer(fail=True)
    update = _update(peer=types.PeerChannel(channel_id=123))

    await _produce_reaction_update(update, producer, None)

    assert producer.events == []
