from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from conversation_engine.enrichment import Brief, EnrichedMessage
from conversation_engine.engagement_gate import GateResult, get_candidate_user_ids
from conversation_engine.memory_manager import ConversationMemoryManager, RetrievedMemory
from storage.postgres_models import BotPersonaCore, BotSelfReflection, UserRelationshipProfile


def _message_label(message: EnrichedMessage) -> str:
    timestamp = message.timestamp.isoformat() if message.timestamp else "unknown"
    return (
        f"message_id: {message.message_id}\n"
        f"timestamp: {timestamp}\n"
        f"sender: user_{message.sender_id}\n"
        f"reply_to: {message.reply_to_message_id if message.reply_to_message_id is not None else 'none'}\n"
        f"raw_text: {message.raw_text if message.raw_text is not None else message.text}\n"
        f"cleaned_text: {message.cleaned_text if message.cleaned_text is not None else message.text}"
    )


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
        f"[{memory.memory_type}] {memory.content}"
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
    ]
    return "\n".join(lines)


def select_target_message(
    enriched_messages: list[EnrichedMessage],
    entry_points: list[int] | None = None,
) -> EnrichedMessage | None:
    by_id = {message.message_id: message for message in enriched_messages}
    for message_id in entry_points or []:
        if message_id in by_id:
            return by_id[message_id]
    for message in reversed(enriched_messages):
        if message.text.strip():
            return message
    return enriched_messages[-1] if enriched_messages else None


def build_thread_context(enriched_messages: list[EnrichedMessage], target: EnrichedMessage | None) -> str:
    if not target:
        return "No target message selected."
    related_ids = {target.message_id}
    if target.reply_to_message_id is not None:
        related_ids.add(target.reply_to_message_id)
    thread_messages = [
        message
        for message in enriched_messages
        if message.message_id in related_ids
        or message.reply_to_message_id in related_ids
        or (
            target.reply_to_message_id is not None
            and message.reply_to_message_id == target.reply_to_message_id
        )
    ]
    if not thread_messages:
        thread_messages = [target]
    return "\n".join(
        f"{message.message_id} user_{message.sender_id} reply_to={message.reply_to_message_id}: {message.text}"
        for message in thread_messages[-20:]
    )


def build_target_message_block(
    enriched_messages: list[EnrichedMessage],
    entry_points: list[int] | None = None,
) -> str:
    target = select_target_message(enriched_messages, entry_points)
    if not target:
        return "=== TARGET MESSAGE ===\nNo target message selected."
    return f"""
=== TARGET MESSAGE ===
{_message_label(target)}
thread_context:
{build_thread_context(enriched_messages, target)}
""".strip()


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

    messages = "\n".join(
        (
            f"message_id={message.message_id} "
            f"timestamp={message.timestamp.isoformat() if message.timestamp else 'unknown'} "
            f"sender=user_{message.sender_id} "
            f"reply_to={message.reply_to_message_id if message.reply_to_message_id is not None else 'none'} "
            f"text={message.text}"
        )
        for message in enriched_messages[-100:]
    )
    candidate_line = ", ".join(f"user_{user_id}" for user_id in candidate_users) or "none"

    context = f"""
=== RECENT CHAT ===
{messages}

=== CURRENT BRIEF ===
{brief.summary}

=== BOT SELF MEMORY ===
Recent outcome score is {avg_feedback:.2f}.

=== VECTOR PERSONA MEMORIES (most relevant to current context) ===
{format_vector_memories(persona_memories)}

=== LATEST SELF REFLECTION ===
{format_latest_reflection(latest_reflection, current_persona)}

=== HARD CONSTRAINTS ===
gate_score: {gate.gate_score:.2f}
tension_level: {brief.tension_level:.2f}
outcome_score_24h: {avg_feedback:.2f}
candidate_users: {candidate_line}
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
    target_message_block: str = "",
) -> str:
    meta = meta_reflection or {}
    lines = [
        target_message_block,
        "",
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
            "",
            "=== SEMANTIC JUDGMENT ===",
            "Answer these before drafting: Is this worth replying to? What exact message are you replying to? "
            "Why? What are the risks? What would make this annoying?",
            "Use the exact target message and thread context above. Numeric controls are only operational hints; "
            "the visible conversation is authoritative.",
        ]
    )
    return "\n".join(lines)
