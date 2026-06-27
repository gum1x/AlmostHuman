"""Gate telemetry on the LLM path (TASK A2).

_finalize_cycle must persist the FULL 9-factor gate dict (merged with the visible
numeric controls, not replaced by them) on every cycle that reached the LLM, and on
declines must store the model's actual reasoning instead of the constant
'decision_should_not_respond'.
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
from conversation_engine.scheduler import ConversationScheduler, _CyclePrep, _CycleLlmOutcome, _decline_reasoning

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
    async def send_message(self, chat_id, text, reply_to_message_id):
        return 555


class FakeFeedbackLoop:
    async def schedule_observation(self, bot_memory_id, sent_message_id, chat_id):
        return None


def make_scheduler() -> ConversationScheduler:
    return ConversationScheduler(
        make_config(),
        ai_client=SimpleNamespace(),
        sender=FakeSender(),
        feedback_loop=FakeFeedbackLoop(),
        bot_user_id=9999,
        bot_username="thebot",
    )


class FakeMemory:
    def __init__(self):
        self.decisions = []
        self.bot_memories = []
        self.vector_memories = []

    async def insert_ai_decision(self, **kwargs):
        self.decisions.append(kwargs)
        return SimpleNamespace(id=len(self.decisions))

    async def record_cycle_success(self, chat_id):
        return None

    async def update_ai_decision_sent_message(self, decision_id, sent_message_id):
        return None

    async def insert_bot_memory(self, **kwargs):
        self.bot_memories.append(kwargs)
        return SimpleNamespace(id=len(self.bot_memories))

    async def write_vector_memory(self, **kwargs):
        self.vector_memories.append(kwargs)
        return None

    async def upsert_stance(self, chat_id, topic, stance, user_id=None):
        return None


class _FakeSession:
    """No-op stand-in for the SQLAlchemy AsyncSession context managers opened
    inside _finalize_cycle (`async with async_session_factory()` /
    `async with session.begin()`)."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def begin(self):
        return self


def patch_finalize_memory(monkeypatch, memory: FakeMemory) -> None:
    """Route the sessions _finalize_cycle now opens internally to ``memory``.

    _finalize_cycle manages its own short transactions (so the send happens
    outside any open txn), so the fake memory is injected by stubbing the
    session factory + ConversationMemoryManager rather than passed as an arg.
    """
    monkeypatch.setattr(
        "conversation_engine.scheduler.async_session_factory", lambda: _FakeSession()
    )
    monkeypatch.setattr(
        "conversation_engine.scheduler.ConversationMemoryManager", lambda session: memory
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


async def test_declined_cycle_persists_full_gate_factors_and_real_reasoning(monkeypatch):
    scheduler = make_scheduler()
    memory = FakeMemory()
    patch_finalize_memory(monkeypatch, memory)
    decision = ResponseDecision(
        should_respond=False,
        confidence=0.2,
        reasoning="this convo is between two people settling a deal, not my moment",
        annoying_reason="butting in would read as forced",
    )
    await scheduler._finalize_cycle(make_prep(), make_llm_out(decision, ok=False, reason="decision_should_not_respond"))

    assert len(memory.decisions) == 1
    row = memory.decisions[0]
    # FULL gate factors persisted (positive-class training data), merged not replaced.
    for key in GATE_FACTOR_KEYS:
        assert key in row["gate_factors"], f"gate factor {key} destroyed on LLM path"
    assert row["gate_factors"]["tension_level"] == 0.3
    assert row["gate_factors"]["outcome_score_24h"] == 0.1
    # Real model reasoning persisted, not just the constant.
    assert row["reasoning"] != "decision_should_not_respond"
    assert "decision_should_not_respond" in row["reasoning"]
    assert "not my moment" in row["reasoning"]
    assert "butting in would read as forced" in row["reasoning"]


async def test_sent_cycle_persists_full_gate_factors(monkeypatch):
    scheduler = make_scheduler()
    memory = FakeMemory()
    patch_finalize_memory(monkeypatch, memory)
    decision = ResponseDecision(
        should_respond=True,
        confidence=0.9,
        response_text="lol ok",
        reply_to_message_id=15,
        reply_to_user_id=2,
        reasoning="funny moment, jumping in",
    )
    await scheduler._finalize_cycle(make_prep(), make_llm_out(decision, ok=True, reason=None))

    row = memory.decisions[0]
    for key in GATE_FACTOR_KEYS:
        assert key in row["gate_factors"]
    assert row["gate_factors"]["tension_level"] == 0.3
    assert row["reasoning"].startswith("funny moment, jumping in")
    assert memory.bot_memories  # response actually recorded


def test_decline_reasoning_concatenates_and_truncates():
    decision = ResponseDecision(should_respond=False, reasoning="r" * 5000, annoying_reason="a" * 100)
    text = _decline_reasoning("low_confidence:0.2", decision)
    assert text.startswith("low_confidence:0.2 | r")
    assert len(text) <= 2000
    # No model reasoning -> just the validator reason.
    assert _decline_reasoning("empty_response", ResponseDecision(should_respond=False)) == "empty_response"
