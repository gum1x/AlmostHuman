from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from conversation_engine.enrichment import Brief, EnrichedMessage
from conversation_engine.engagement_gate import GateResult, get_candidate_user_ids
from conversation_engine.memory_manager import ConversationMemoryManager, RetrievedMemory
from storage.postgres_models import BotPersonaCore, BotSelfReflection, UserRelationshipProfile


_AVERAGE_CHARS_PER_TOKEN = 4
_NEARBY_NOISE_TERMS = (
    "sold ✅",
    "price:",
    "rent. gifts",
    "marketapp",
    "off-chain ➡️ off-chain",
)


def _clip(text: str | None, max_chars: int) -> str:
    value = (text or "").strip()
    if len(value) <= max_chars:
        return value
    return f"{value[: max_chars - 3].rstrip()}..."


def _is_noisy_nearby(text: str) -> bool:
    lowered = text.lower()
    if any(term in lowered for term in _NEARBY_NOISE_TERMS):
        return True
    return len(text) > 360


def _script_hint(text: str) -> str:
    if not text.strip():
        return "empty"
    ranges = {
        "cyrillic": (0x0400, 0x04FF),
        "arabic": (0x0600, 0x06FF),
        "devanagari": (0x0900, 0x097F),
        "cjk": (0x4E00, 0x9FFF),
        "hangul": (0xAC00, 0xD7AF),
        "kana": (0x3040, 0x30FF),
    }
    counts = {name: 0 for name in ranges}
    latin = 0
    letters = 0
    for char in text:
        if char.isalpha():
            letters += 1
            codepoint = ord(char)
            if "a" <= char.lower() <= "z":
                latin += 1
            for name, (start, end) in ranges.items():
                if start <= codepoint <= end:
                    counts[name] += 1
    if not letters:
        return "symbols_or_numbers"
    dominant = max(counts.items(), key=lambda item: item[1])
    if dominant[1] > 0 and dominant[1] >= latin:
        return dominant[0]
    if latin / max(1, letters) > 0.8:
        return "latin/english_or_romanized"
    return "mixed_or_unknown"


def _message_label(message: EnrichedMessage) -> str:
    return (
        f"{message.message_id} user_{message.sender_id} "
        f"reply_to={message.reply_to_message_id if message.reply_to_message_id is not None else 'none'}: "
        f"{_clip(message.cleaned_text if message.cleaned_text is not None else message.text, 500)}"
    )


@dataclass(frozen=True)
class ContextBundle:
    context: str
    candidate_user_ids: list[int]
    relationship_profiles: list[UserRelationshipProfile]
    avg_feedback_score: float


def extract_target_line(context: str) -> str:
    for line in context.splitlines():
        if line.startswith("target:"):
            return line
    return "target: none"


def build_response_context(source_context: str, relevant_context_summary: str = "") -> str:
    lines: list[str] = []
    summary = relevant_context_summary.strip()
    if summary:
        lines.extend(["context:", _clip(summary, 260)])
    lines.extend(["respond to this message:", extract_target_line(source_context)])
    return "\n".join(lines)


def format_vector_memories(memories: list[RetrievedMemory]) -> str:
    if not memories:
        return ""
    return "\n".join(
        (
            f"{memory.memory_type}: {_clip(memory.content, 180)}"
        )
        for memory in sorted(memories, key=lambda item: (item.similarity * item.importance_score), reverse=True)
    )


def format_latest_reflection(
    latest_reflection: BotSelfReflection | None,
    current_persona: BotPersonaCore | None,
) -> str:
    if not latest_reflection:
        return ""
    lines = [
        _clip(latest_reflection.updated_summary, 220),
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


def _format_context_message(message: EnrichedMessage, max_chars: int = 260) -> str:
    return (
        f"{message.message_id} user_{message.sender_id} "
        f"reply_to={message.reply_to_message_id if message.reply_to_message_id is not None else 'none'}: "
        f"{_clip(message.cleaned_text if message.cleaned_text is not None else message.text, max_chars)}"
    )


def build_thread_context(enriched_messages: list[EnrichedMessage], target: EnrichedMessage | None) -> str:
    if not target:
        return "No target message selected."
    thread_messages: list[EnrichedMessage] = []
    if target.reply_to_message_id is not None:
        thread_messages.extend(
            message
            for message in enriched_messages
            if message.message_id == target.reply_to_message_id
        )
    thread_messages.extend(
        message
        for message in enriched_messages
        if message.reply_to_message_id == target.message_id
    )
    seen: set[int] = {target.message_id}
    unique_thread = []
    for message in thread_messages:
        if message.message_id in seen:
            continue
        seen.add(message.message_id)
        unique_thread.append(message)
    if not unique_thread:
        return ""
    return "\n".join(
        _format_context_message(message)
        for message in unique_thread[-3:]
    )


def build_target_message_block(
    enriched_messages: list[EnrichedMessage],
    entry_points: list[int] | None = None,
) -> str:
    target = select_target_message(enriched_messages, entry_points)
    if not target:
        return "target: none"
    thread = build_thread_context(enriched_messages, target)
    if not thread:
        return f"target: {_message_label(target)}"
    return f"target: {_message_label(target)}\nreply_context:\n{thread}"


def _format_local_window(enriched_messages: list[EnrichedMessage], target: EnrichedMessage | None, radius: int = 2) -> str:
    if not target:
        return ""
    target_index = next(
        (index for index, message in enumerate(enriched_messages) if message.message_id == target.message_id),
        len(enriched_messages) - 1,
    )
    start = max(0, target_index - radius)
    end = min(len(enriched_messages), target_index + radius + 1)
    thread_ids = {target.message_id}
    if target.reply_to_message_id is not None:
        thread_ids.add(target.reply_to_message_id)
    return "\n".join(
        _format_context_message(message, 180)
        for message in enriched_messages[start:end]
        if message.message_id not in thread_ids and message.reply_to_message_id != target.message_id
        and not _is_noisy_nearby(message.text)
    )


def _format_relationship_profiles(profiles: list[UserRelationshipProfile]) -> str:
    if not profiles:
        return ""
    lines = []
    for profile in profiles[:4]:
        if not profile.notes:
            continue
        lines.append(
            (
                f"user_{profile.user_id}: {_clip(profile.notes, 120)}"
            )
        )
    return "\n".join(lines)


def _add_optional_section(lines: list[str], title: str, value: str) -> None:
    if value.strip():
        lines.extend([title, value.strip()])


async def build_context(
    chat_id: int,
    enriched_messages: list[EnrichedMessage],
    brief: Brief,
    gate: GateResult,
    memory: ConversationMemoryManager,
    persona_memories: list[RetrievedMemory],
    latest_reflection: BotSelfReflection | None,
    current_persona: BotPersonaCore | None,
    token_budget: int = 6_000,
) -> ContextBundle:
    candidate_users = get_candidate_user_ids(enriched_messages)
    profiles = await memory.get_relationship_profiles(chat_id, candidate_users)
    avg_feedback = await memory.get_avg_feedback_score(chat_id, window_hours=24)
    chat_mode = "private_dm" if chat_id > 0 else "group_chat"
    target = select_target_message(enriched_messages)
    latest_text = enriched_messages[-1].text if enriched_messages else ""
    script_hint = _script_hint(latest_text)
    lines = [build_target_message_block(enriched_messages)]
    if chat_mode == "private_dm":
        lines.insert(0, "mode: private_dm")
    _add_optional_section(lines, "nearby:", _format_local_window(enriched_messages, target))
    if script_hint not in {"latin/english_or_romanized", "empty", "symbols_or_numbers"}:
        lines.append(f"lang: {script_hint}")
    if brief.tension_level >= 0.7 or avg_feedback <= -0.4:
        lines.append(f"signals: tension={brief.tension_level:.2f} feedback={avg_feedback:.2f}")
    _add_optional_section(lines, "memory:", format_vector_memories(persona_memories))
    _add_optional_section(lines, "users:", _format_relationship_profiles(profiles))
    context = "\n".join(lines).strip()
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
