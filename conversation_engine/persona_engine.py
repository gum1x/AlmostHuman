from __future__ import annotations

import random
from datetime import timedelta
from typing import Any

from conversation_engine.ai_client import parse_reflection
from conversation_engine.config import EngineConfig
from conversation_engine.memory_manager import ConversationMemoryManager, RetrievedMemory, utcnow
from conversation_engine.observability import (
    record_reflection_triggered,
    record_vector_memory_retrieved,
)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None


_embedder: Any | None = None


class ZeroEmbedder:
    def encode(self, text: str) -> list[float]:
        return [0.0] * 384


def load_embedder(model_name: str):
    global _embedder
    if _embedder is None:
        if SentenceTransformer is None:
            _embedder = ZeroEmbedder()
        else:
            _embedder = SentenceTransformer(model_name)
    return _embedder


def embed_text(text: str) -> list[float]:
    embedder = _embedder or ZeroEmbedder()
    value = embedder.encode(text)
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [float(item) for item in value]


def initial_identity_text(config: EngineConfig) -> str:
    return (
        f"{config.persona.identity}. Core beliefs: {config.persona.core_beliefs}. "
        f"Speaking style: {config.persona.speaking_style}"
    )


async def seed_persona_core(memory: ConversationMemoryManager, config: EngineConfig):
    text = initial_identity_text(config)
    return await memory.seed_persona_if_empty(
        identity_summary=config.persona.identity,
        core_beliefs=config.persona.core_beliefs,
        speaking_style=config.persona.speaking_style,
        embedding=embed_text(text),
    )


def interaction_importance(positive_feedback: bool = False, high_engagement: bool = False) -> float:
    return min(1.0, 0.5 + (0.3 if positive_feedback else 0.0) + (0.2 if high_engagement else 0.0))


async def write_interaction_memory(
    memory: ConversationMemoryManager,
    chat_id: int,
    user_id: int | None,
    topic: str | None,
    response_text: str,
    positive_feedback: bool = False,
    high_engagement: bool = False,
) -> None:
    content = f"Responded to user_{user_id} about {topic or 'general chat'}: {response_text}"
    await memory.write_vector_memory(
        chat_id=chat_id,
        memory_type="interaction",
        user_id=user_id,
        content=content,
        embedding=embed_text(content),
        importance_score=interaction_importance(positive_feedback, high_engagement),
    )


async def write_stance_memory(memory: ConversationMemoryManager, chat_id: int, user_id: int | None, topic: str, stance: str) -> None:
    content = f"Expressed {stance} about {topic} to user_{user_id}"
    await memory.write_vector_memory(
        chat_id=chat_id,
        memory_type="stance",
        user_id=user_id,
        content=content,
        embedding=embed_text(content),
        importance_score=0.7,
    )


async def write_relationship_memory(memory: ConversationMemoryManager, chat_id: int, user_id: int, notes: str) -> None:
    content = f"Relationship with user_{user_id}: {notes}"
    await memory.write_vector_memory(
        chat_id=chat_id,
        memory_type="relationship",
        user_id=user_id,
        content=content,
        embedding=embed_text(content),
        importance_score=0.6,
    )


async def should_run_self_reflection(
    memory: ConversationMemoryManager,
    chat_id: int,
    config: EngineConfig,
) -> tuple[bool, str, int]:
    latest = await memory.get_latest_self_reflection(chat_id)
    messages_since_last = await memory.count_bot_memory_since_last_reflection(chat_id)
    if messages_since_last >= config.persona_engine.self_reflection_message_threshold:
        return True, "message_threshold", messages_since_last
    if latest is None:
        return messages_since_last > 0, "scheduled", messages_since_last
    interval = config.persona_engine.self_reflection_interval_hours
    randomized = random.uniform(max(4, interval - 2), min(12, interval + 2))
    if utcnow() - latest.created_at >= timedelta(hours=randomized):
        return True, "scheduled", messages_since_last
    return False, "none", messages_since_last


def format_recent_messages(messages) -> str:
    return "\n".join(
        f"{message.sent_at.isoformat()} reply_to={message.reply_to_message_id}: {message.response_text}"
        for message in reversed(messages)
    )


def format_feedback(feedback) -> str:
    return "\n".join(
        f"{item.scored_at.isoformat()} message={item.sent_message_id} outcome={item.outcome} score={item.outcome_score}"
        for item in reversed(feedback)
    )


async def run_self_reflection(
    chat_id: int,
    memory: ConversationMemoryManager,
    ai_client,
    config: EngineConfig,
    trigger: str = "scheduled",
    messages_since_last: int | None = None,
) -> None:
    recent_messages = await memory.get_recent_bot_memory(chat_id, limit=50)
    recent_feedback = await memory.get_recent_feedback(chat_id, limit=50)
    current_persona = await memory.get_persona_core()
    if not current_persona:
        current_persona = await seed_persona_core(memory, config)
    if messages_since_last is None:
        messages_since_last = await memory.count_bot_memory_since_last_reflection(chat_id)

    prompt = f"""
You are reflecting on your recent behavior in a Telegram group chat.

YOUR CORE IDENTITY:
{current_persona.identity_summary}

YOUR CORE BELIEFS:
{current_persona.core_beliefs}

YOUR SPEAKING STYLE:
{current_persona.speaking_style}

YOUR RECENT MESSAGES (last 50):
{format_recent_messages(recent_messages)}

FEEDBACK ON THOSE MESSAGES:
{format_feedback(recent_feedback)}

Reflect on consistency, engagement patterns, what is working, user responses, and tone drift.
Produce only JSON with keys reflection_text, updated_summary, drift_explanation,
relationship_updates, and tone_adjustments. Do not invent numeric scores.
""".strip()
    result = await ai_client.call_perception_model(prompt)
    parsed = parse_reflection(result.text)
    reflection_embedding = embed_text(parsed.reflection_text)

    await memory.insert_self_reflection(
        chat_id=chat_id,
        trigger=trigger,
        messages_since_last=messages_since_last,
        reflection_text=parsed.reflection_text,
        updated_summary=parsed.updated_summary,
        drift_score=parsed.drift_score,
        embedding=reflection_embedding,
    )

    for update in parsed.relationship_updates:
        await memory.upsert_user_relationship(chat_id, update.user_id, update.notes, embed_text(update.notes))
        await write_relationship_memory(memory, chat_id, update.user_id, update.notes)

    await memory.write_vector_memory(
        chat_id=chat_id,
        memory_type="reflection",
        content=parsed.reflection_text,
        embedding=reflection_embedding,
        importance_score=0.9,
    )
    record_reflection_triggered(trigger)


async def get_relevant_persona_vectors(
    chat_id: int,
    current_context_text: str,
    memory: ConversationMemoryManager,
    top_k: int = 5,
) -> tuple[list[RetrievedMemory], Any]:
    memories = await memory.get_relevant_vector_memories(
        chat_id=chat_id,
        query_embedding=embed_text(current_context_text),
        top_k=top_k,
    )
    latest_reflection = await memory.get_latest_self_reflection(chat_id)
    record_vector_memory_retrieved(len(memories))
    return memories, latest_reflection
