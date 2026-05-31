from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from conversation_engine.config import EngineConfig
from conversation_engine.enrichment import Brief, EnrichedMessage
from conversation_engine.memory_manager import ConversationMemoryManager
from conversation_engine.observability import record_gate


@dataclass(frozen=True)
class GateResult:
    gate_score: float
    gate_factors: dict[str, float | str]
    should_proceed: bool


def score_velocity(velocity: float) -> float:
    if velocity < 0.5:
        return 0.2
    if velocity <= 5.0:
        return 1.0
    if velocity > 10.0:
        return 0.3
    return max(0.3, 1.0 - ((velocity - 5.0) / 5.0) * 0.7)


def get_candidate_user_ids(enriched_messages: list[EnrichedMessage]) -> list[int]:
    seen: list[int] = []
    for message in reversed(enriched_messages[-20:]):
        if message.sender_id is not None and message.sender_id not in seen:
            seen.append(message.sender_id)
    return seen[:5]


def get_active_thread_message_ids(enriched_messages: list[EnrichedMessage], brief: Brief | None) -> list[int]:
    ids = {message.reply_to_message_id for message in enriched_messages[-20:] if message.reply_to_message_id}
    if brief:
        ids.update(thread.root_message_id for thread in brief.active_threads)
    return [int(item) for item in ids if item is not None]


async def compute_gate_score(
    chat_id: int,
    enriched_messages: list[EnrichedMessage],
    brief: Brief | None,
    memory: ConversationMemoryManager,
    config: EngineConfig,
) -> GateResult:
    factors: dict[str, float | str] = {}
    gate_config = config.engagement_gate

    recent_count = await memory.count_messages_in_window(chat_id, minutes=gate_config.velocity_window_minutes)
    velocity = recent_count / gate_config.velocity_window_minutes
    factors["velocity"] = score_velocity(velocity)

    recent_sentiments = [message.sentiment_score for message in enriched_messages[-20:]]
    sentiment_trend = sum(recent_sentiments) / len(recent_sentiments) if recent_sentiments else 0.0
    tension = brief.tension_level if brief else 0.0
    factors["emotional_trend"] = max(0.0, (sentiment_trend + 1.0) / 2.0) * (1.0 - tension)

    if tension > gate_config.anti_flame_tension_threshold:
        active_direct = any(
            thread.status == "active_direct_reply_to_bot" and thread.urgency == "high"
            for thread in (brief.active_threads if brief else [])
        )
        if not active_direct:
            result = GateResult(
                gate_score=0.0,
                gate_factors={"blocked": "anti_flame_protection"},
                should_proceed=False,
            )
            record_gate(result.gate_score, {})
            return result

    responses_last_hour = await memory.count_bot_responses(chat_id, window_minutes=60)
    responses_last_10min = await memory.count_bot_responses(chat_id, window_minutes=10)
    fatigue = min(
        (responses_last_hour / max(1, config.prompt.max_responses_per_hour)) * 0.6
        + (responses_last_10min / 3.0) * 0.4,
        gate_config.max_fatigue_score,
    )
    factors["fatigue"] = 1.0 - fatigue

    candidate_users = get_candidate_user_ids(enriched_messages)
    factors["relationship_strength"] = await memory.avg_relationship_strength(chat_id, candidate_users)

    factors["topic_alignment"] = max(
        (message.topic_overlap_score for message in enriched_messages),
        default=0.0,
    )
    factors["topic_drift_penalty"] = 0.6 if brief and brief.topic_drift else 1.0

    now = datetime.now(timezone.utc)
    pattern = await memory.get_activity_pattern(chat_id, now.hour, now.weekday())
    factors["historical_activity"] = min(pattern.avg_message_velocity / 5.0, 1.0) if pattern else 0.5

    active_thread_ids = get_active_thread_message_ids(enriched_messages, brief)
    recent_thread_responses = await memory.count_bot_responses_in_threads(chat_id, active_thread_ids, 30)
    if recent_thread_responses >= gate_config.thread_repeat_penalty_count:
        factors["thread_repeat"] = max(0.1, 1.0 - (recent_thread_responses - 1) * 0.35)
    else:
        factors["thread_repeat"] = 1.0

    avg_feedback = await memory.get_avg_feedback_score(chat_id, window_hours=24)
    factors["feedback_signal"] = (avg_feedback + 1.0) / 2.0

    weights = {
        "velocity": 0.15,
        "emotional_trend": 0.15,
        "fatigue": 0.20,
        "relationship_strength": 0.10,
        "topic_alignment": 0.15,
        "topic_drift_penalty": 0.05,
        "historical_activity": 0.05,
        "thread_repeat": 0.10,
        "feedback_signal": 0.05,
    }
    gate_score = sum(float(factors[key]) * weight for key, weight in weights.items())
    result = GateResult(
        gate_score=max(0.0, min(1.0, gate_score)),
        gate_factors=factors,
        # The score is advisory context for Grok, not a hard model-call gate.
        should_proceed=True,
    )
    record_gate(result.gate_score, {key: float(value) for key, value in factors.items() if isinstance(value, int | float)})
    return result
