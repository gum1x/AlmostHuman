from __future__ import annotations

from dataclasses import dataclass

from conversation_engine.engagement_gate import GateResult, get_candidate_user_ids
from conversation_engine.enrichment import Brief, EnrichedMessage
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
        (f"{memory.memory_type}: {_clip(memory.content, 180)}")
        for memory in sorted(
            memories, key=lambda item: item.similarity * item.importance_score, reverse=True
        )
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


def build_thread_context(
    enriched_messages: list[EnrichedMessage], target: EnrichedMessage | None
) -> str:
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
        message for message in enriched_messages if message.reply_to_message_id == target.message_id
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
    return "\n".join(_format_context_message(message) for message in unique_thread[-3:])


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


def _format_local_window(
    enriched_messages: list[EnrichedMessage], target: EnrichedMessage | None, radius: int = 2
) -> str:
    if not target:
        return ""
    target_index = next(
        (
            index
            for index, message in enumerate(enriched_messages)
            if message.message_id == target.message_id
        ),
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
        if message.message_id not in thread_ids
        and message.reply_to_message_id != target.message_id
        and not _is_noisy_nearby(message.text)
    )


def _extract_preferred_tone(notes: str | None) -> str | None:
    """Pull 'Preferred tone: X' out of relationship notes if present."""
    if not notes:
        return None
    for line in notes.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("preferred tone:"):
            return stripped[len("preferred tone:") :].strip()
    return None


def _format_relationship_profiles(profiles: list[UserRelationshipProfile]) -> str:
    if not profiles:
        return ""
    lines = []
    for profile in profiles[:4]:
        if not profile.notes:
            continue
        lines.append((f"user_{profile.user_id}: {_clip(profile.notes, 120)}"))
    return "\n".join(lines)


def _add_optional_section(lines: list[str], title: str, value: str) -> None:
    if value.strip():
        lines.extend([title, value.strip()])


def compute_quantitative_signals(
    enriched_messages: list[EnrichedMessage],
    bot_user_id: int | None = None,
    bot_sent_message_ids: set[int] | None = None,
    time_since_last_bot_msg_min: float | None = None,
    responses_last_hour: int = 0,
    bot_username: str | None = None,
) -> dict[str, str | float | int | bool]:
    """Compute high-value raw activity numbers from message history and bot state.

    Only the counts and checks that track the bot's own recent messages sent
    (responses_last_hour, time since, is_reply_to_bot via bot_sent_ids scan)
    plus direct_mention (for the 3Q "never ignore" rule) are retained.
    These are the "numbers" the character uses as memory of its output rate,
    direct threads, and explicit address/continuation. No derived scores, no
    emotional/unresolved/velocity/tension/direct modeling values for quantitative
    decision reasoning. The qualitative model (three questions) uses the raw
    facts + injected persona memory instead.
    """
    bot_ids = bot_sent_message_ids or set()
    target = select_target_message(enriched_messages)

    # --- is_reply_to_bot: the check "throughout the whole chat" whether the
    # target directly replies to one of our previously sent messages (from BotMemory).
    is_reply_to_bot = False
    if target and target.reply_to_message_id is not None:
        is_reply_to_bot = target.reply_to_message_id in bot_ids

    # direct_mention: high-value flag for the qualitative rule. Covers explicit
    # @username address, reply to one of our messages, or (when caller sets it
    # via active_bot_thread) continuation of a thread we are part of.
    direct_mention = bool(is_reply_to_bot)
    if target:
        txt = (target.cleaned_text or target.text or "").lower()
        if bot_username and f"@{bot_username.lower().strip()}" in txt:
            direct_mention = True
        if "temp3289" in txt:
            direct_mention = True

    return {
        "is_reply_to_bot": is_reply_to_bot,
        "time_since_last_bot_msg_min": round(time_since_last_bot_msg_min, 1)
        if time_since_last_bot_msg_min is not None
        else -1,
        "responses_last_hour": responses_last_hour,
        "direct_mention": direct_mention,
    }


def format_quantitative_signals(signals: dict[str, str | float | int | bool]) -> str:
    """Format pre-computed signals as a compact block for the context."""
    lines = []
    for key, val in signals.items():
        lines.append(f"{key}={val}")
    return " | ".join(lines)


def format_enriched_for_context(
    enriched_messages: list[EnrichedMessage], max_chars: int = 180
) -> str:
    """Public helper to format a list of messages for the high/recent context
    summarizer prompt (and similar). Uses the same clipping/label style as
    the internal target/nearby formatters so the summarizer sees consistent ids.
    """
    if not enriched_messages:
        return ""
    return "\n".join(_format_context_message(m, max_chars) for m in enriched_messages)


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
    recent_bot_activity: str = "",
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

    # === Self-referential blocks for the smart model as a real participant ===
    # These give the character its own history and current state so it can have
    # natural timing/engagement rhythm instead of pure per-cycle external judgment.
    if current_persona:
        lines.append("=== WHO I AM (my character) ===")
        lines.append(current_persona.identity_summary or "")
        if current_persona.core_beliefs:
            lines.append("Core beliefs: " + " | ".join(current_persona.core_beliefs))
        if current_persona.speaking_style:
            lines.append("How I talk: " + current_persona.speaking_style)

    if latest_reflection:
        lines.append("=== MY LATEST SELF-REFLECTION ===")
        lines.append(latest_reflection.updated_summary or latest_reflection.reflection_text or "")

    # Recent activity as the character (critical for natural timing and "my" rhythm)
    # The smart model uses this to know what it's been doing lately as itself,
    # so it can decide whether something feels like a continuation of *its* energy or not.
    if recent_bot_activity.strip():
        lines.append("=== MY RECENT ACTIVITY AS ME ===")
        lines.append(recent_bot_activity.strip())

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
    relationship_profiles: list[UserRelationshipProfile],
    avg_feedback_score: float = 0.0,
    target_message_block: str = "",
) -> str:
    """Per-decision constraints fed to the decision model so feedback and
    relationships actually influence what gets said.

    Built entirely from already-persisted state: how recent replies landed
    (24h ``avg_feedback_score`` from ResponseFeedback), what the bot has learned
    in this chat (latest self-reflection, which is itself feedback-driven), and
    per-user ``preferred_tone`` (written by meta-reflection into relationship
    notes). No new storage required — it surfaces signals the engine already
    computes but previously dropped before the prompt.
    """
    lines: list[str] = []
    if target_message_block.strip():
        lines.extend([target_message_block.strip(), ""])

    if avg_feedback_score > 0.15:
        landed = (
            f"recent replies have landed well (avg {avg_feedback_score:+.2f} on a -1..1 scale) — "
            "keep doing what's working"
        )
    elif avg_feedback_score < -0.15:
        landed = (
            f"recent replies have landed flat/poorly (avg {avg_feedback_score:+.2f} on a -1..1 scale) — "
            "be sharper or stay silent more"
        )
    else:
        landed = f"recent replies have been roughly neutral (avg {avg_feedback_score:+.2f} on a -1..1 scale)"
    lines.extend(["=== FEEDBACK LEARNING ===", f"How my recent replies landed: {landed}."])
    if latest_reflection and (latest_reflection.updated_summary or "").strip():
        lines.append(
            f"What I've learned in this chat: {_clip(latest_reflection.updated_summary.strip(), 240)}"
        )

    tone_lines = []
    for profile in relationship_profiles:
        tone = _extract_preferred_tone(profile.notes)
        if tone:
            tone_lines.append(f"user_{profile.user_id}: preferred_tone={tone}")
    if tone_lines:
        lines.extend(["", "=== RELATIONSHIP TONE ===", *tone_lines])

    lines.extend(
        [
            "",
            "=== PERSONA ALIGNMENT ===",
            f"Core identity: {current_persona.identity_summary if current_persona else 'unknown'}",
            "If your drafted reply contradicts your core identity or ignores the feedback above, revise it.",
        ]
    )
    return "\n".join(lines)
