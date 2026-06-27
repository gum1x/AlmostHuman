from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import conversation_engine.feedback_loop as feedback_loop
from conversation_engine.feedback_loop import FeedbackLoop, Reaction, score_outcome
from conversation_engine.memory_manager import utcnow


def fake_sentiment(text: str) -> float:
    if "HOSTILE" in text:
        return -0.8
    if "FRIENDLY" in text:
        return 0.8
    return 0.0


@pytest.fixture(autouse=True)
def patch_sentiment(monkeypatch):
    monkeypatch.setattr(feedback_loop, "sentiment_score", fake_sentiment)


def msg(message_id: int, text: str, sender_id: int | None = 111, reply_to: int | None = None):
    return SimpleNamespace(
        message_id=message_id,
        sender_id=sender_id,
        text_cleaned=text,
        text_raw=text,
        reply_to_message_id=reply_to,
        timestamp=utcnow(),
    )


# --- score_outcome ---


async def test_hostile_reply_scores_negative():
    outcome, score = await score_outcome([msg(2, "HOSTILE trash take")], [], 0.0)
    assert outcome == "negative"
    assert score < 0


async def test_friendly_reply_scores_positive():
    outcome, score = await score_outcome([msg(2, "FRIENDLY love this")], [], 0.0)
    assert outcome == "positive"
    assert score > 0


async def test_mixed_replies_score_proportional():
    replies = [msg(2, "HOSTILE"), msg(3, "FRIENDLY")]
    outcome, score = await score_outcome(replies, [], 0.0)
    assert outcome == "neutral"
    assert score == pytest.approx(0.0)


async def test_no_engagement_is_ignored():
    outcome, score = await score_outcome([], [], 0.0)
    assert outcome == "ignored"
    assert score == 0.0


async def test_killed_thread_is_negative_outcome():
    outcome, score = await score_outcome([], [], 0.0, chat_killed=True)
    assert outcome == "killed"
    assert score < 0


async def test_backlash_on_negative_reactions_and_hostile_replies():
    outcome, score = await score_outcome([msg(2, "HOSTILE")], [Reaction("👎", 3)], 0.0)
    assert outcome == "backlash"
    assert score == -0.8


async def test_heavy_negative_reactions_only_is_backlash():
    outcome, score = await score_outcome([], [Reaction("👎", 3)], 0.0)
    assert outcome == "backlash"
    assert score == -0.8


async def test_mild_negative_reactions_only_is_negative():
    outcome, score = await score_outcome([], [Reaction("👎", 2)], 0.0)
    assert outcome == "negative"
    assert score == -0.4


async def test_hostile_reply_not_labeled_positive_despite_being_a_reply():
    # Regression: the old cascade labeled ANY text reply "positive" because
    # quote_replies always equaled replies.
    outcome, score = await score_outcome([msg(2, "HOSTILE shut up bot", reply_to=1)], [], 0.0)
    assert outcome != "positive"
    assert score < 0


# --- observe_response wiring ---


class FakeMemory:
    def __init__(self, replies=None, follow_up=None, baseline=None, before_count=0, after_count=0):
        self.replies = replies or []
        self.follow_up = follow_up or []
        self.baseline = baseline or []
        self.before_count = before_count
        self.after_count = after_count
        self.inserted = None
        self.exchanges = []
        self.messages_after_args = None
        self.count_calls = []

    async def get_replies_to(self, chat_id, sent_message_id):
        return self.replies

    async def get_messages_after(self, chat_id, sent_message_id, sent_at, window_minutes):
        self.messages_after_args = (chat_id, sent_message_id, sent_at, window_minutes)
        return self.follow_up

    async def get_messages_before(self, chat_id, sent_message_id, limit=20):
        return self.baseline

    async def count_messages_between(self, chat_id, start, end, exclude_message_id=None):
        self.count_calls.append((start, end, exclude_message_id))
        return self.before_count if len(self.count_calls) == 1 else self.after_count

    async def insert_response_feedback(self, **kwargs):
        self.inserted = kwargs

    async def record_user_exchange(self, **kwargs):
        self.exchanges.append(kwargs)


class FakeBegin:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class FakeSession:
    def begin(self):
        return FakeBegin()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


def make_loop(monkeypatch, fake_memory):
    monkeypatch.setattr(feedback_loop, "async_session_factory", lambda: FakeSession())
    monkeypatch.setattr(feedback_loop, "ConversationMemoryManager", lambda session: fake_memory)
    config = SimpleNamespace(feedback_loop=SimpleNamespace(observation_window_minutes=0))
    return FeedbackLoop(config, ai_client=None, sender=None)


async def test_observe_response_hostile_reply_records_negative(monkeypatch):
    fake = FakeMemory(replies=[msg(2, "HOSTILE garbage", sender_id=42, reply_to=1)])
    loop = make_loop(monkeypatch, fake)
    await loop.observe_response(bot_memory_id=7, sent_message_id=1, chat_id=-100)
    assert fake.inserted["outcome"] == "negative"
    assert fake.inserted["outcome_score"] < 0
    [exchange] = fake.exchanges
    assert exchange["user_id"] == 42
    assert exchange["outcome_score"] < 0
    assert exchange["reply_sentiment"] == pytest.approx(-0.8)


async def test_observe_response_friendly_reply_records_positive(monkeypatch):
    fake = FakeMemory(replies=[msg(2, "FRIENDLY nice one", sender_id=42, reply_to=1)])
    loop = make_loop(monkeypatch, fake)
    await loop.observe_response(bot_memory_id=7, sent_message_id=1, chat_id=-100)
    assert fake.inserted["outcome"] == "positive"
    assert fake.inserted["outcome_score"] > 0
    [exchange] = fake.exchanges
    assert exchange["outcome_score"] > 0
    assert exchange["reply_sentiment"] == pytest.approx(0.8)


async def test_observe_response_killed_thread(monkeypatch):
    fake = FakeMemory(replies=[], before_count=6, after_count=0)
    loop = make_loop(monkeypatch, fake)
    await loop.observe_response(bot_memory_id=7, sent_message_id=1, chat_id=-100)
    assert fake.inserted["outcome"] == "killed"
    assert fake.inserted["outcome_score"] < 0
    # before-window count then after-window count, both excluding the bot's own message
    assert len(fake.count_calls) == 2
    assert all(call[2] == 1 for call in fake.count_calls)
    assert fake.exchanges == []


async def test_observe_response_quiet_chat_not_killed(monkeypatch):
    # Chat was already quiet before the send: ignored, not killed.
    fake = FakeMemory(replies=[], before_count=2, after_count=0)
    loop = make_loop(monkeypatch, fake)
    await loop.observe_response(bot_memory_id=7, sent_message_id=1, chat_id=-100)
    assert fake.inserted["outcome"] == "ignored"
    assert fake.inserted["outcome_score"] == 0.0


async def test_observe_response_velocity_drop_boundary_killed(monkeypatch):
    # killed when after_count <= 0.2 * before_count: 2 <= 0.2 * 10
    fake = FakeMemory(replies=[], before_count=10, after_count=2)
    loop = make_loop(monkeypatch, fake)
    await loop.observe_response(bot_memory_id=7, sent_message_id=1, chat_id=-100)
    assert fake.inserted["outcome"] == "killed"


async def test_observe_response_velocity_drop_above_boundary_not_killed(monkeypatch):
    # 3 > 0.2 * 10: chat slowed but did not die
    fake = FakeMemory(replies=[], before_count=10, after_count=3)
    loop = make_loop(monkeypatch, fake)
    await loop.observe_response(bot_memory_id=7, sent_message_id=1, chat_id=-100)
    assert fake.inserted["outcome"] == "ignored"


async def test_observe_response_sentiment_shift_uses_baseline(monkeypatch):
    fake = FakeMemory(
        replies=[msg(2, "FRIENDLY", sender_id=42, reply_to=1)],
        follow_up=[msg(3, "FRIENDLY")],
        baseline=[msg(0, "HOSTILE")],
    )
    loop = make_loop(monkeypatch, fake)
    await loop.observe_response(bot_memory_id=7, sent_message_id=1, chat_id=-100)
    assert fake.inserted["follow_up_sentiment"] == pytest.approx(0.8 - (-0.8))


async def test_observe_response_window_anchored_to_send_time(monkeypatch):
    fake = FakeMemory()
    loop = make_loop(monkeypatch, fake)
    before = utcnow()
    await loop.observe_response(bot_memory_id=7, sent_message_id=1, chat_id=-100)
    after = utcnow()
    _, _, sent_at, window_minutes = fake.messages_after_args
    assert before <= sent_at <= after
    assert window_minutes == 0


# --- background-task GC safety ---


async def test_observation_task_tracked_then_discarded(monkeypatch):
    # Regression: the fire-and-forget observe_response task was created without
    # keeping a strong reference, so asyncio could GC it mid-flight. The loop must
    # hold it in self._bg_tasks while pending and drop it once complete.
    config = SimpleNamespace(feedback_loop=SimpleNamespace(observation_window_minutes=0))
    loop = FeedbackLoop(config, ai_client=None, sender=None)

    gate = asyncio.Event()
    seen_args = []

    async def fake_observe(bot_memory_id, sent_message_id, chat_id):
        seen_args.append((bot_memory_id, sent_message_id, chat_id))
        await gate.wait()

    monkeypatch.setattr(loop, "observe_response", fake_observe)

    await loop._queue.put((7, 1, -100))
    drainer = asyncio.create_task(loop.run_observation_tasks())
    try:
        # Let the loop pull the item and spawn the observation task.
        while not seen_args:
            await asyncio.sleep(0)
        assert seen_args == [(7, 1, -100)]
        # Task is in flight (blocked on the gate) and held by a strong ref.
        assert len(loop._bg_tasks) == 1
        [obs_task] = loop._bg_tasks
        assert not obs_task.done()

        # Release the coroutine and let the done-callback fire.
        gate.set()
        await obs_task
        await asyncio.sleep(0)
        assert loop._bg_tasks == set()
    finally:
        loop.shutdown()
        await drainer
