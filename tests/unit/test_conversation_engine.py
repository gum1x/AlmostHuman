from __future__ import annotations

from pathlib import Path

import pytest

from conversation_engine.ai_client import ResponseDecision
from conversation_engine.config import load_engine_config
from conversation_engine.context_builder import build_context, build_target_message_block
from conversation_engine.engagement_gate import GateResult, compute_gate_score, score_velocity
from conversation_engine.enrichment import Brief, EnrichedMessage
from conversation_engine.feedback_loop import Reaction, score_outcome
from conversation_engine.memory_manager import RetrievedMemory
from conversation_engine.prompts import build_decide_and_draft_prompt, build_reflection_prompt
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


def test_validate_accepts_confident_nonempty_response(default_engine_config):
    decision = ResponseDecision(
        should_respond=True,
        confidence=0.9,
        response_text="hello",
    )

    ok, reason = validate(decision, default_engine_config)

    assert ok is True
    assert reason is None


def test_validate_rejects_avoided_users(default_engine_config):
    default_engine_config.prompt.avoid_users.append(42)
    decision = ResponseDecision(
        should_respond=True,
        confidence=0.9,
        response_text="hello",
        reply_to_user_id=42,
    )

    ok, reason = validate(decision, default_engine_config)

    assert ok is False
    assert reason == "avoided_user:42"


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
    assert "[interaction] remember this" in bundle.context
    assert "gate_score: 0.77" in bundle.context
    assert "tension_level: 0.10" in bundle.context
    assert "outcome_score_24h: 0.25" in bundle.context
    assert "candidate_users: user_42" in bundle.context
    assert "gate_factors" not in bundle.context
    assert "relationship_strength" not in bundle.context
    assert "receptiveness" not in bundle.context
    assert "relevance:" not in bundle.context


def test_target_message_block_includes_exact_message_and_thread():
    messages = [
        EnrichedMessage(
            10,
            -100,
            1,
            "parent cleaned",
            None,
            0.0,
            0.0,
            [],
            raw_text="parent raw",
            cleaned_text="parent cleaned",
        ),
        EnrichedMessage(
            11,
            -100,
            42,
            "target cleaned",
            10,
            0.1,
            1.0,
            ["crypto"],
            raw_text="target raw",
            cleaned_text="target cleaned",
        ),
    ]

    block = build_target_message_block(messages, [11])

    assert "=== TARGET MESSAGE ===" in block
    assert "message_id: 11" in block
    assert "sender: user_42" in block
    assert "reply_to: 10" in block
    assert "raw_text: target raw" in block
    assert "cleaned_text: target cleaned" in block
    assert "10 user_1 reply_to=None: parent cleaned" in block


def test_decide_and_draft_prompt_combines_gate_target_and_response(default_engine_config):
    bundle = type(
        "Bundle",
        (),
        {
            "context": "=== RECENT CHAT ===\nmessage_id=1 sender=user_42 text=btc?",
        },
    )()

    prompt, system = build_decide_and_draft_prompt(bundle, default_engine_config)

    assert "DECISION PROMPT: DECIDE_AND_DRAFT" in prompt
    assert "should_respond" in prompt
    assert "response_text" in prompt
    assert "reply_to_message_id" in prompt
    assert "Return only valid JSON" in system


def test_reflection_prompt_handles_learning_tasks():
    prompt, system = build_reflection_prompt(
        "meta_reflection",
        {"feedback_count": 10, "aggregated_feedback": {"overall_trend": 0.2}},
    )

    assert "REFLECTION PROMPT" in prompt
    assert "Task: meta_reflection" in prompt
    assert "what_works" in prompt
    assert "updated_stance_recommendations" in prompt
    assert "Return only valid JSON" in system
