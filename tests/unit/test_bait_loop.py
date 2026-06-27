"""Bait-loop hard caps (TASK A3).

A user who keeps replying to the bot must not be able to make it respond every cycle:
(a) max_group_responses_per_10min is absolute — the direct-reply override cannot cross it;
(b) after max_consecutive_replies_per_user consecutive replies to the same user with no
    other human turn in between, that user's direct replies stop force-proceeding;
(c) the anti-flame bypass (tension >= threshold) works at most once per
    same_thread_cooldown_minutes.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("ALLOW_FAKE_EMBEDDER", "true")

from conversation_engine.ai_client import ResponseDecision
from conversation_engine.config import (
    AiConfig,
    CircuitBreakerConfig,
    EngagementGateConfig,
    EngineConfig,
    FeedbackLoopConfig,
    PersonaConfig,
    PersonaEngineConfig,
    PromptConfig,
    SchedulerConfig,
)
from conversation_engine.engagement_gate import GateResult
from conversation_engine.enrichment import Brief
from conversation_engine.persona_engine import load_embedder
from conversation_engine.scheduler import ConversationScheduler

load_embedder("test")

BOT_ID = 9999
BAITER = 666
OTHER = 777


@dataclass
class Msg:
    message_id: int
    sender_id: int | None
    reply_to_message_id: int | None
    text: str
    chat_id: int = -100
    timestamp: datetime = datetime(2026, 6, 10, tzinfo=timezone.utc)

    @property
    def text_cleaned(self):
        return self.text

    @property
    def text_raw(self):
        return self.text

    @property
    def cleaned_text(self):
        return self.text

    @property
    def raw_text(self):
        return self.text


def bot_mem_entry(sent_message_id, reply_to_user_id, sent_at=None, tension=0.0):
    return SimpleNamespace(
        sent_message_id=sent_message_id,
        reply_to_user_id=reply_to_user_id,
        reply_to_message_id=None,
        response_text="ok",
        reasoning=None,
        current_posture=None,
        sent_at=sent_at or datetime.now(timezone.utc),
        created_at=sent_at or datetime.now(timezone.utc),
        brief_snapshot={"tension_level": tension},
    )


def make_config(**gate_overrides) -> EngineConfig:
    return EngineConfig(
        active_chat_ids=[],
        xai_api_key="",
        xai_base_url="",
        conversation_tg_session_name="test",
        persona=PersonaConfig(),
        ai=AiConfig(),
        prompt=PromptConfig(),
        scheduler=SchedulerConfig(),
        circuit_breaker=CircuitBreakerConfig(),
        persona_engine=PersonaEngineConfig(),
        feedback_loop=FeedbackLoopConfig(),
        engagement_gate=EngagementGateConfig(**gate_overrides),
    )


def make_scheduler(config=None) -> ConversationScheduler:
    return ConversationScheduler(
        config or make_config(),
        ai_client=SimpleNamespace(),
        sender=SimpleNamespace(),
        feedback_loop=SimpleNamespace(),
        bot_user_id=BOT_ID,
        bot_username="thebot",
    )


class FakeMemory:
    def __init__(self, messages, bot_mem=None, responses_10min=0, responses_60min=0):
        self.messages = list(messages)  # chronological
        self.bot_mem = list(bot_mem or [])  # newest first
        self.responses_10min = responses_10min
        self.responses_60min = responses_60min
        self.decisions = []

    async def get_latest_ai_decision(self, chat_id):
        return None

    async def count_messages_after_snapshot(self, chat_id, snapshot_message_id):
        return len(self.messages)

    async def get_recent_bot_memory(self, chat_id, limit=50):
        return self.bot_mem[:limit]

    async def get_recent_messages(self, chat_id, limit=100):
        return self.messages[-limit:]

    async def seed_persona_if_empty(self, **kwargs):
        return None

    async def get_avg_feedback_score(self, chat_id, window_hours=24):
        return 0.0

    async def latest_message_id(self, chat_id):
        return self.messages[-1].message_id if self.messages else None

    async def upsert_activity_pattern(self, chat_id, hour_of_day, day_of_week, velocity, tension):
        return None

    async def count_messages_in_window(self, chat_id, minutes):
        return len(self.messages)

    async def count_bot_responses(self, chat_id, window_minutes):
        return self.responses_10min if window_minutes == 10 else self.responses_60min

    async def avg_relationship_strength(self, chat_id, user_ids):
        return 0.0

    async def get_activity_pattern(self, chat_id, hour, day=None):
        return None

    async def count_bot_responses_in_threads(self, chat_id, thread_message_ids, window_minutes):
        return 0

    async def insert_ai_decision(self, **kwargs):
        self.decisions.append(kwargs)
        return SimpleNamespace(id=len(self.decisions))

    async def record_cycle_success(self, chat_id):
        return None


def bait_thread():
    """101: baiter pokes; 102/104: bot replies; 103/105: baiter replies straight back."""
    return [
        Msg(101, BAITER, None, "oi @thebot say something"),
        Msg(102, BOT_ID, 101, "what"),
        Msg(103, BAITER, 102, "ur so dumb lol"),
        Msg(104, BOT_ID, 103, "sure"),
        Msg(105, BAITER, 104, "no answer? pathetic"),
    ]


async def test_rate_cap_is_absolute_even_for_direct_replies():
    scheduler = make_scheduler()
    # 3 responses in the last 10 min == max_group_responses_per_10min -> the gate sets
    # rate_limit_10min and should_proceed=False; the direct reply must NOT override it.
    memory = FakeMemory(
        messages=bait_thread(),
        bot_mem=[bot_mem_entry(104, BAITER), bot_mem_entry(102, BAITER)],
        responses_10min=3,
        responses_60min=3,
    )
    result = await scheduler._prepare_cycle(memory, chat_id=-100, is_private_dm=False)

    assert result is None  # blocked: no LLM call, no send
    assert len(memory.decisions) == 1
    row = memory.decisions[0]
    assert row["should_respond"] is False
    assert row["gate_factors"]["rate_limit_10min"] == 3.0
    assert row["gate_factors"]["direct_override_suppressed"] == "rate_limit_10min"


async def test_consecutive_replies_to_same_user_stop_force_proceed():
    # Gate blocked on score (min set impossibly high), rate cap NOT hit; the bot already
    # replied to the baiter twice in a row with no other human turn in between.
    scheduler = make_scheduler(make_config(min_gate_score_to_send=0.99, max_consecutive_replies_per_user=2))
    memory = FakeMemory(
        messages=bait_thread(),
        bot_mem=[bot_mem_entry(104, BAITER), bot_mem_entry(102, BAITER)],
        responses_10min=1,
        responses_60min=2,
    )
    result = await scheduler._prepare_cycle(memory, chat_id=-100, is_private_dm=False)

    assert result is None
    row = memory.decisions[0]
    assert row["should_respond"] is False
    assert row["gate_factors"]["direct_override_suppressed"] == "consecutive_user_cap"


def test_other_human_turn_resets_consecutive_cap():
    scheduler = make_scheduler()
    bot_mem = [bot_mem_entry(104, BAITER), bot_mem_entry(102, BAITER)]
    thread = bait_thread() + [Msg(106, OTHER, None, "lmao this thread"), Msg(107, BAITER, 104, "well?")]
    gate = GateResult(gate_score=0.1, gate_factors={"velocity": 0.2}, should_proceed=False)
    target = thread[-1]
    # Another human spoke after the bot's replies -> cap no longer applies.
    assert scheduler._hit_consecutive_user_cap(target, bot_mem, thread) is False
    assert scheduler._direct_override_suppression(gate, target, bot_mem, thread) is None
    # Without that turn the cap holds.
    assert scheduler._hit_consecutive_user_cap(bait_thread()[-1], bot_mem, bait_thread()) is True


def test_under_cap_direct_override_still_allowed():
    scheduler = make_scheduler()
    gate = GateResult(gate_score=0.1, gate_factors={"velocity": 0.2}, should_proceed=False)
    thread = bait_thread()[:3]  # only one bot reply so far
    assert scheduler._direct_override_suppression(gate, thread[-1], [bot_mem_entry(102, BAITER)], thread) is None


def test_bait_sequence_simulation_bot_stops_engaging():
    """User replies to the bot every cycle. The bot may engage at most
    max_consecutive_replies_per_user (2) times in a row, then suppression kicks in
    and stays; the 10-min rate cap (3) is never crossed."""
    scheduler = make_scheduler()
    messages = [Msg(100, BAITER, None, "oi @thebot")]
    bot_mem = []  # newest first
    next_id = 101
    bot_replies = 0
    suppressions = []

    for _ in range(8):
        last_bot_id = bot_mem[0].sent_message_id if bot_mem else None
        messages.append(Msg(next_id, BAITER, last_bot_id, "answer me @thebot"))
        target = messages[-1]
        next_id += 1

        factors = {"velocity": 0.2}
        if bot_replies >= scheduler.config.engagement_gate.max_group_responses_per_10min:
            factors["rate_limit_10min"] = float(bot_replies)
        gate = GateResult(gate_score=0.1, gate_factors=factors, should_proceed=False)

        suppression = scheduler._direct_override_suppression(gate, target, bot_mem, messages)
        suppressions.append(suppression)
        if suppression is None:  # direct override force-proceeds -> bot replies
            messages.append(Msg(next_id, BOT_ID, target.message_id, "reply"))
            bot_mem.insert(0, bot_mem_entry(next_id, BAITER))
            next_id += 1
            bot_replies += 1

    assert bot_replies == 2  # stopped at the consecutive cap
    assert suppressions[:2] == [None, None]
    assert all(s == "consecutive_user_cap" for s in suppressions[2:])
    assert bot_replies <= scheduler.config.engagement_gate.max_group_responses_per_10min  # rate cap never crossed


def test_anti_flame_bypass_at_most_once_per_cooldown():
    scheduler = make_scheduler()
    now = datetime.now(timezone.utc)
    thread = bait_thread()
    brief = Brief(tension_level=0.8, topic_drift=False, active_threads=[], summary="")

    def check(recent_bot_mem):
        decision = ResponseDecision(should_respond=True, confidence=0.9, response_text="x", reply_to_message_id=105)
        return scheduler._passes_social_safety(
            is_private_dm=False,
            active_bot_thread=True,
            enriched=thread,
            brief=brief,
            decision=decision,
            bot_sent_ids={102, 104},
            recent_bot_mem=recent_bot_mem,
        )

    # Bypass spent 5 minutes ago (a reply sent while tension was high) -> blocked.
    spent = [bot_mem_entry(104, BAITER, sent_at=now - timedelta(minutes=5), tension=0.8)]
    assert check(spent) == (False, "anti_flame_protection")
    # Last high-tension reply outside the cooldown window -> bypass available again.
    old = [bot_mem_entry(104, BAITER, sent_at=now - timedelta(minutes=31), tension=0.8)]
    assert check(old) == (True, None)
    # Recent reply but sent at low tension -> bypass not spent.
    calm = [bot_mem_entry(104, BAITER, sent_at=now - timedelta(minutes=5), tension=0.1)]
    assert check(calm) == (True, None)
    # No bot activity at all -> first bypass allowed.
    assert check([]) == (True, None)


def test_anti_flame_still_blocks_non_direct_messages():
    scheduler = make_scheduler()
    brief = Brief(tension_level=0.8, topic_drift=False, active_threads=[], summary="")
    thread = [Msg(201, OTHER, None, "everyone here is a scammer fuck off")]
    decision = ResponseDecision(should_respond=True, confidence=0.9, response_text="x", reply_to_message_id=201)
    ok, reason = scheduler._passes_social_safety(
        is_private_dm=False,
        active_bot_thread=False,
        enriched=thread,
        brief=brief,
        decision=decision,
        bot_sent_ids=set(),
        recent_bot_mem=[],
    )
    assert (ok, reason) == (False, "anti_flame_protection")
