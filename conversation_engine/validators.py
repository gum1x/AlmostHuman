from __future__ import annotations

from conversation_engine.ai_client import ResponseDecision
from conversation_engine.config import EngineConfig


def validate(decision: ResponseDecision, config: EngineConfig) -> tuple[bool, str | None]:
    if not decision.should_respond:
        return False, "decision_should_not_respond"
    if decision.confidence < config.ai.min_confidence_to_send:
        return False, f"low_confidence:{decision.confidence}"
    if decision.persona_alignment_score < 0.5:
        return False, f"persona_misalignment:{decision.persona_alignment_score}"
    if not decision.response_text or not decision.response_text.strip():
        return False, "empty_response"
    if len(decision.response_text) > 4096:
        return False, "telegram_message_too_long"
    return True, None
