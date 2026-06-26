"""Pure, instance-free support pieces extracted from ``scheduler.py``.

Holds module-level dataclasses and pure helper functions that carry no
``ConversationScheduler`` state. Kept here to slim the scheduler god-module;
behavior is identical and ``scheduler.py`` re-imports these names so its public
surface is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from conversation_engine.ai_client import ResponseDecision
from conversation_engine.context_builder import ContextBundle
from conversation_engine.engagement_gate import GateResult
from conversation_engine.enrichment import Brief


@dataclass(frozen=True)
class _CyclePrep:
    chat_id: int
    is_private_dm: bool
    active_bot_thread: bool
    new_message_count: int
    snapshot_message_id: int | None
    gate: GateResult
    visible_numeric_controls: dict[str, Any]
    brief: Brief
    enriched: list
    context: ContextBundle
    raw_context: str
    high_level_enriched: list
    recent_enriched_for_summary: list
    recent_bot_mem: list
    bot_sent_ids: set[int]
    recent_bot_activity: str
    posture: str
    responses_last_hour: int
    # Behavioral-layer counts (only populated when behavioral_layer_enabled; else 0).
    group_msgs_last_hour: int = 0
    bot_sends_last_10min: int = 0
    # Persona + latest self-reflection carried into the decision-time constraints
    # block (how recent replies landed + per-user tone + persona alignment).
    current_persona: Any = None
    latest_reflection: Any = None


@dataclass(frozen=True)
class _CycleLlmOutcome:
    decision: ResponseDecision
    request1: Any
    request2: Any
    posture: str
    ok: bool
    reason: str | None


def _decline_reasoning(reason: str | None, decision: ResponseDecision) -> str:
    """Reasoning text persisted when a cycle doesn't send: the validator/decline
    reason plus the model's actual reasoning (not just the constant)."""
    parts = [reason or "declined"]
    if decision.reasoning:
        parts.append(decision.reasoning)
    if decision.annoying_reason:
        parts.append(f"annoying_reason: {decision.annoying_reason}")
    return " | ".join(parts)[:2000]


def _append_context_block(context, title: str, body: str):
    if not body.strip():
        return context
    return type(context)(
        context=f"{context.context}\n\n=== {title} ===\n{body.strip()}",
        candidate_user_ids=context.candidate_user_ids,
        relationship_profiles=context.relationship_profiles,
        avg_feedback_score=context.avg_feedback_score,
    )
