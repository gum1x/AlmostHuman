from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from conversation_engine.enrichment import Brief, EnrichedMessage
from conversation_engine.engagement_gate import GateResult, get_candidate_user_ids
from conversation_engine.memory_manager import ConversationMemoryManager, RetrievedMemory
from storage.postgres_models import BotPersonaCore, BotSelfReflection, UserRelationshipProfile


@dataclass(frozen=True)
class ContextBundle:
    context: str
    candidate_user_ids: list[int]
    relationship_profiles: list[UserRelationshipProfile]
    avg_feedback_score: float


def format_vector_memories(memories: list[RetrievedMemory]) -> str:
    if not memories:
        return "No vector persona memories yet."
    return "\n".join(
        f"[{memory.memory_type}] {memory.content} (relevance: {memory.similarity:.2f})"
        for memory in memories
    )


def format_latest_reflection(
    latest_reflection: BotSelfReflection | None,
    current_persona: BotPersonaCore | None,
) -> str:
    if not latest_reflection:
        return "No self-reflection has been recorded yet."
    lines = [
        latest_reflection.reflection_text,
        f"Self-summary: {latest_reflection.updated_summary}",
        f"Identity drift score: {latest_reflection.drift_score:.2f}",
    ]
    if latest_reflection.drift_score > 0.4 and current_persona:
        lines.append(
            "WARNING: recent behavior has drifted from core identity. "
            f"Recalibrate toward: {current_persona.identity_summary}"
        )
    return "\n".join(lines)


def _relationships_by_user(profiles: list[UserRelationshipProfile]) -> dict[int, UserRelationshipProfile]:
    return {profile.user_id: profile for profile in profiles}


async def build_context(
    chat_id: int,
    enriched_messages: list[EnrichedMessage],
    brief: Brief,
    gate: GateResult,
    memory: ConversationMemoryManager,
    persona_memories: list[RetrievedMemory],
    latest_reflection: BotSelfReflection | None,
    current_persona: BotPersonaCore | None,
) -> ContextBundle:
    candidate_users = get_candidate_user_ids(enriched_messages)
    profiles = await memory.get_relationship_profiles(chat_id, candidate_users)
    avg_feedback = await memory.get_avg_feedback_score(chat_id, window_hours=24)
    profile_map = _relationships_by_user(profiles)

    messages = "\n".join(
        f"{message.message_id} user_{message.sender_id}: {message.text}"
        for message in enriched_messages[-100:]
    )
    receptiveness = {
        str(profile.user_id): profile.receptiveness_score
        for profile in profiles
    }
    candidate_lines = []
    for user_id in candidate_users:
        profile = profile_map.get(user_id)
        if profile:
            candidate_lines.append(
                f"user_{user_id}: relationship_strength={profile.relationship_strength:.2f}, "
                f"preferred_tone=unknown, total_exchanges={profile.total_exchanges}"
            )
        else:
            candidate_lines.append(
                f"user_{user_id}: relationship_strength=0.00, preferred_tone=unknown, total_exchanges=0"
            )

    context = f"""
=== RECENT CHAT ===
{messages}

=== CURRENT BRIEF ===
{brief.summary}

=== BOT SELF MEMORY ===
Recent feedback trend is {avg_feedback:.2f}.

=== VECTOR PERSONA MEMORIES (most relevant to current context) ===
{format_vector_memories(persona_memories)}

=== LATEST SELF REFLECTION ===
{format_latest_reflection(latest_reflection, current_persona)}

=== HARD CONSTRAINTS ===
gate_score: {gate.gate_score:.2f}
gate_factors: {json.dumps(gate.gate_factors, sort_keys=True)}
feedback_trend_24h: {avg_feedback:.2f}
receptiveness_by_user: {json.dumps(receptiveness, sort_keys=True)}
{chr(10).join(candidate_lines)}
""".strip()
    return ContextBundle(
        context=context,
        candidate_user_ids=candidate_users,
        relationship_profiles=profiles,
        avg_feedback_score=avg_feedback,
    )


def build_request2_constraints(
    current_persona: BotPersonaCore | None,
    latest_reflection: BotSelfReflection | None,
    meta_reflection: dict[str, Any] | None,
    relationship_profiles: list[UserRelationshipProfile],
) -> str:
    meta = meta_reflection or {}
    lines = [
        "=== FEEDBACK LEARNING ===",
        f"What has worked recently: {meta.get('what_works', 'unknown')}",
        f"What has not worked: {meta.get('what_doesnt', 'unknown')}",
    ]
    for profile in relationship_profiles:
        lines.append(f"For user_{profile.user_id} specifically: preferred_tone=unknown")
    lines.extend(
        [
            "",
            "=== PERSONA ALIGNMENT CHECK ===",
            f"Core identity: {current_persona.identity_summary if current_persona else 'unknown'}",
            f"Latest self-reflection: {latest_reflection.updated_summary if latest_reflection else 'none'}",
            "If your drafted response contradicts your core identity, revise it.",
        ]
    )
    return "\n".join(lines)
