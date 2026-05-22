from __future__ import annotations

from pathlib import Path

import pytest

from conversation_engine.ai_client import ResponseDecision
from conversation_engine.config import load_engine_config
from conversation_engine.context_builder import build_context
from conversation_engine.engagement_gate import GateResult, compute_gate_score, score_velocity
from conversation_engine.enrichment import Brief, EnrichedMessage
from conversation_engine.feedback_loop import Reaction, score_outcome
from conversation_engine.memory_manager import RetrievedMemory
from conversation_engine.validators import validate
from storage.postgres_models import BotPersonaCore, UserRelationshipProfile


def test_load_engine_config_reads_toml_and_env(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[persona]
identity = "identity"
core_beliefs = ["belief"]
speaking_style = "style"

[prompt]
topics_of_interest = ["crypto"]
""".strip()
    )
    monkeypatch.setenv("ACTIVE_CHAT_IDS", "-1001,-1002")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
    monkeypatch.setenv("WORKER_POOL_SIZE", "7")

    config = load_engine_config(config_path)

    assert config.active_chat_ids == [-1001, -1002]
    assert config.anthropic_api_key == "key"
    assert config.scheduler.worker_pool_size == 7
    assert config.persona.identity == "identity"
    assert config.prompt.topics_of_interest == ["crypto"]


def test_score_velocity_sweet_spot_and_extremes():
    assert score_velocity(0.1) == 0.2
    assert score_velocity(1.0) == 1.0
    assert score_velocity(11.0) == 0.3


class FakeGateMemory:
    async def count_messages_in_window(self, chat_id, minutes):
        return 20

    async def count_bot_responses(self, chat_id, window_minutes):
        return 0

    async def avg_relationship_strength(self, chat_id, user_ids):
        return 0.5

    async def get_activity_pattern(self, chat_id, hour, day):
        return None

    async def count_bot_responses_in_threads(self, chat_id, thread_message_ids, window_minutes):
        return 0

    async def get_avg_feedback_score(self, chat_id, window_hours):
        return 0.2


@pytest.mark.asyncio
async def test_gate_hard_blocks_high_tension(default_engine_config):
    messages = [
        EnrichedMessage(1, -100, 1, "angry", None, -0.8, 1.0, ["crypto"]),
    ]
    brief = Brief(tension_level=0.9, topic_drift=False, active_threads=[], summary="tense")

    result = await compute_gate_score(-100, messages, brief, FakeGateMemory(), default_engine_config)

    assert result.should_proceed is False
    assert result.gate_score == 0.0
    assert result.gate_factors["blocked"] == "anti_flame_protection"


def test_validate_rejects_persona_misalignment(default_engine_config):
    decision = ResponseDecision(
        should_respond=True,
        confidence=0.9,
        response_text="hello",
        persona_alignment_score=0.4,
    )

    ok, reason = validate(decision, default_engine_config)

    assert ok is False
    assert reason == "persona_misalignment:0.4"


@pytest.mark.asyncio
async def test_score_outcome_fast_paths():
    assert await score_outcome([], [], [], 0.0) == ("ignored", 0.0)
    assert await score_outcome([object()], [Reaction("👎", 2)], [], -0.5) == ("backlash", -0.8)
    assert await score_outcome([], [Reaction("👍", 3)], [], 0.1) == ("positive", 0.5)


class FakeContextMemory:
    async def get_relationship_profiles(self, chat_id, user_ids):
        return [
            UserRelationshipProfile(
                chat_id=chat_id,
                user_id=user_ids[0],
                relationship_strength=0.7,
                receptiveness_score=0.6,
                total_exchanges=4,
            )
        ]

    async def get_avg_feedback_score(self, chat_id, window_hours):
        return 0.25


@pytest.mark.asyncio
async def test_context_builder_injects_persona_and_gate(default_engine_config):
    messages = [
        EnrichedMessage(1, -100, 42, "crypto infra looks good", None, 0.2, 1.0, ["crypto"]),
    ]
    brief = Brief(tension_level=0.1, topic_drift=False, active_threads=[], summary="summary")
    gate = GateResult(gate_score=0.77, gate_factors={"velocity": 1.0}, should_proceed=True)
    persona = BotPersonaCore(
        identity_summary="identity",
        core_beliefs=["belief"],
        speaking_style="style",
        version=1,
    )

    bundle = await build_context(
        -100,
        messages,
        brief,
        gate,
        FakeContextMemory(),
        [RetrievedMemory("remember this", "interaction", 0.5, 0.8)],
        None,
        persona,
    )

    assert "=== VECTOR PERSONA MEMORIES" in bundle.context
    assert "[interaction] remember this (relevance: 0.80)" in bundle.context
    assert "gate_score: 0.77" in bundle.context
    assert "user_42: relationship_strength=0.70" in bundle.context
