from __future__ import annotations

import re
from collections.abc import Iterable

from conversation_engine.ai_client import ResponseDecision
from conversation_engine.config import EngineConfig


def _normalize_for_dedup(text: str) -> str:
    """Lowercase, strip punctuation/whitespace so trivial variations still count as identical."""
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def is_duplicate_response(text: str, recent_texts: Iterable[str]) -> bool:
    """True if `text` matches any recent bot message after normalization."""
    norm = _normalize_for_dedup(text)
    if not norm:
        return False
    return any(_normalize_for_dedup(prev) == norm for prev in recent_texts)


def validate(
    decision: ResponseDecision,
    config: EngineConfig,
    recent_bot_texts: Iterable[str] | None = None,
) -> tuple[bool, str | None]:
    if not decision.should_respond:
        return False, "decision_should_not_respond"
    if decision.confidence < config.ai.min_confidence_to_send:
        return False, f"low_confidence:{decision.confidence}"
    if decision.reply_to_user_id in set(config.prompt.avoid_users):
        return False, f"avoided_user:{decision.reply_to_user_id}"
    if not decision.response_text or not decision.response_text.strip():
        return False, "empty_response"
    if len(decision.response_text) > 4096:
        return False, "telegram_message_too_long"
    if recent_bot_texts and is_duplicate_response(decision.response_text, recent_bot_texts):
        return False, "duplicate_of_recent_response"
    return True, None
