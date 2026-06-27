"""Default-path (behavioral_layer_enabled=False) send/transaction restructure.

Regression for the bug where the physical Telegram send ran INSIDE the open DB
transaction that also held the post-send recording writes: a failure in any of
those writes rolled back an already-DELIVERED message and made the outer handler
record a cycle FAILURE (tripping the circuit breaker for a successful send).

_finalize_cycle now mirrors the behavioral path: it sends OUTSIDE any
transaction, then records in a SEPARATE short transaction wrapped in try/except.
A post-send recording failure is logged ('post_send_record_failed') and the
cycle is treated as a SUCCESS without re-raising (the message is out).
"""
from __future__ import annotations

import os
import sys
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
from conversation_engine.scheduler import ConversationScheduler, _CycleLlmOutcome, _CyclePrep

load_embedder("test")

GATE_FACTOR_KEYS = [
    "velocity",
    "emotional_trend",
    "fatigue",
    "relationship_strength",
    "topic_alignment",
    "topic_drift_penalty",
    "historical_activity",
    "thread_repeat",
    "feedback_signal",
]


def make_config() -> EngineConfig:
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
        engagement_gate=EngagementGateConfig(),
    )


class FakeSender:
    def __init__(self):
        self.calls = []

    async def send_message(self, chat_id, text, reply_to_message_id):
        self.calls.append((chat_id, text, reply_to_message_id))
        return 555


class FakeFeedbackLoop:
    def __init__(self):
        self.observations = []

    async def schedule_observation(self, bot_memory_id, sent_message_id, chat_id):
        self.observations.append((bot_memory_id, sent_message_id, chat_id))


class FakeMemory:
    """Records every persistence call _finalize_cycle makes. Any method name
    listed in ``raise_on`` raises when invoked, simulating a post-send DB error."""

    def __init__(self, raise_on: set[str] | None = None):
        self.raise_on = raise_on or set()
        self.decisions = []
        self.bot_memories = []
        self.vector_memories = []
        self.stances = []
        self.success_calls = []
        self.failure_calls = []
        self.sent_message_updates = []

    def _maybe_raise(self, name):
        if name in self.raise_on:
            raise RuntimeError(f"boom in {name}")

    async def insert_ai_decision(self, **kwargs):
        self._maybe_raise("insert_ai_decision")
        self.decisions.append(kwargs)
        return SimpleNamespace(id=len(self.decisions))

    async def update_ai_decision_sent_message(self, decision_id, sent_message_id):
        self._maybe_raise("update_ai_decision_sent_message")
        self.sent_message_updates.append((decision_id, sent_message_id))

    async def insert_bot_memory(self, **kwargs):
        self._maybe_raise("insert_bot_memory")
        self.bot_memories.append(kwargs)
        return SimpleNamespace(id=len(self.bot_memories))

    async def write_vector_memory(self, **kwargs):
        self._maybe_raise("write_vector_memory")
        self.vector_memories.append(kwargs)

    async def upsert_stance(self, chat_id, topic, stance, user_id=None):
        self._maybe_raise("upsert_stance")
        self.stances.append((chat_id, topic, stance, user_id))

    async def record_cycle_success(self, chat_id):
        self._maybe_raise("record_cycle_success")
        self.success_calls.append(chat_id)

    async def record_cycle_failure(self, *args, **kwargs):
        # If _finalize_cycle ever calls this, the test must fail: a delivered
        # message must never be recorded as a cycle failure.
        self.failure_calls.append((args, kwargs))


class _FakeSession:
    """No-op stand-in for the AsyncSession context managers _finalize_cycle opens
    internally (`async with async_session_factory()` / `async with session.begin()`)."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def begin(self):
        return self


def patch_finalize_memory(monkeypatch, memory: FakeMemory) -> None:
    """Route the short transactions _finalize_cycle opens internally to ``memory``."""
    monkeypatch.setattr(
        "conversation_engine.scheduler.async_session_factory", lambda: _FakeSession()
    )
    monkeypatch.setattr(
        "conversation_engine.scheduler.ConversationMemoryManager", lambda session: memory
    )


def make_scheduler(sender: FakeSender, feedback_loop: FakeFeedbackLoop) -> ConversationScheduler:
    config = make_config()
    assert config.behavioral_layer_enabled is False  # exercising the DEFAULT path
    return ConversationScheduler(
        config,
        ai_client=SimpleNamespace(),
        sender=sender,
        feedback_loop=feedback_loop,
        bot_user_id=9999,
        bot_username="thebot",
    )


def make_prep() -> _CyclePrep:
    gate = GateResult(
        gate_score=0.42,
        gate_factors={key: 0.5 for key in GATE_FACTOR_KEYS},
        should_proceed=True,
    )
    return _CyclePrep(
        chat_id=-100,
        is_private_dm=False,
        active_bot_thread=False,
        new_message_count=4,
        snapshot_message_id=15,
        gate=gate,
        visible_numeric_controls={"tension_level": 0.3, "outcome_score_24h": 0.1},
        brief=Brief(tension_level=0.3, topic_drift=False, active_threads=[], summary=""),
        enriched=[],
        context=None,
        raw_context="",
        high_level_enriched=[],
        recent_enriched_for_summary=[],
        recent_bot_mem=[],
        bot_sent_ids=set(),
        recent_bot_activity="",
        posture="watching",
        responses_last_hour=0,
    )


def make_llm_out(decision: ResponseDecision, ok: bool, reason: str | None) -> _CycleLlmOutcome:
    zero = SimpleNamespace(latency_ms=10, tokens_used=20)
    return _CycleLlmOutcome(
        decision=decision,
        request1=zero,
        request2=zero,
        posture="watching",
        ok=ok,
        reason=reason,
    )


def _send_decision() -> ResponseDecision:
    return ResponseDecision(
        should_respond=True,
        confidence=0.9,
        response_text="lol ok",
        reply_to_message_id=15,
        reply_to_user_id=2,
        reasoning="funny moment, jumping in",
    )


async def test_happy_path_sends_once_records_success_and_schedules_observation(monkeypatch):
    sender = FakeSender()
    feedback_loop = FakeFeedbackLoop()
    scheduler = make_scheduler(sender, feedback_loop)
    memory = FakeMemory()
    patch_finalize_memory(monkeypatch, memory)

    interval = await scheduler._finalize_cycle(make_prep(), make_llm_out(_send_decision(), ok=True, reason=None))

    assert interval == scheduler.config.scheduler.initial_interval_seconds
    # (a) exactly one physical send.
    assert sender.calls == [(-100, "lol ok", 15)]
    # (c) success recorded and the delayed observation scheduled, both on the happy path.
    assert memory.success_calls == [-100]
    assert memory.bot_memories  # the response was actually recorded
    assert feedback_loop.observations == [(1, 555, -100)]
    # A successful cycle never records a failure.
    assert memory.failure_calls == []


async def test_post_send_write_failure_is_swallowed_not_a_cycle_failure(monkeypatch):
    sender = FakeSender()
    feedback_loop = FakeFeedbackLoop()
    scheduler = make_scheduler(sender, feedback_loop)
    # insert_bot_memory raises AFTER the message is already sent.
    memory = FakeMemory(raise_on={"insert_bot_memory"})
    patch_finalize_memory(monkeypatch, memory)

    logged = []

    async def fake_aexception(event, **kwargs):
        logged.append((event, kwargs))

    monkeypatch.setattr("conversation_engine.scheduler.log.aexception", fake_aexception)

    # (b) must NOT propagate even though a post-send write raised.
    interval = await scheduler._finalize_cycle(make_prep(), make_llm_out(_send_decision(), ok=True, reason=None))

    # The message was delivered exactly once...
    assert sender.calls == [(-100, "lol ok", 15)]
    # ...and the failure was logged, not raised.
    assert logged and logged[0][0] == "post_send_record_failed"
    assert logged[0][1]["sent_message_id"] == 555
    # The cycle returns the normal interval (not a backoff) and never records a failure
    # (a delivered send must not trip the circuit breaker).
    assert interval == scheduler.config.scheduler.initial_interval_seconds
    assert memory.failure_calls == []


async def test_decline_records_success_without_sending(monkeypatch):
    sender = FakeSender()
    feedback_loop = FakeFeedbackLoop()
    scheduler = make_scheduler(sender, feedback_loop)
    memory = FakeMemory()
    patch_finalize_memory(monkeypatch, memory)
    decision = ResponseDecision(
        should_respond=False,
        confidence=0.2,
        reasoning="not my moment",
    )

    interval = await scheduler._finalize_cycle(
        make_prep(), make_llm_out(decision, ok=False, reason="decision_should_not_respond")
    )

    assert interval == scheduler.config.scheduler.initial_interval_seconds
    # No send, no observation; the decline is still recorded as a successful cycle.
    assert sender.calls == []
    assert feedback_loop.observations == []
    assert len(memory.decisions) == 1
    assert memory.decisions[0]["should_respond"] is False
    assert memory.success_calls == [-100]
    assert memory.failure_calls == []
