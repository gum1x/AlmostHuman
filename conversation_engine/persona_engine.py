from __future__ import annotations

import asyncio
import os
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
from conversation_engine.prompts import build_self_reflection_prompt
from core.logging import get_logger

log = get_logger(__name__)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None


_embedder: Any | None = None


class ZeroEmbedder:
    """All-zero vectors corrupt pgvector cosine ranking — only valid as an
    explicit opt-in fake for tests (ALLOW_FAKE_EMBEDDER=true) when
    sentence-transformers is missing, never a silent default."""

    def encode(self, text: str) -> list[float]:
        return [0.0] * 384


def load_embedder(model_name: str):
    global _embedder
    if _embedder is None:
        if SentenceTransformer is not None:
            _embedder = SentenceTransformer(model_name)
        elif os.getenv("ALLOW_FAKE_EMBEDDER", "").lower() == "true":
            _embedder = ZeroEmbedder()
        else:
            raise RuntimeError(
                "sentence-transformers is not installed: embeddings would be all-zero and "
                "silently corrupt all pgvector cosine ranking. Install sentence-transformers, "
                "or set ALLOW_FAKE_EMBEDDER=true (tests only)."
            )
    return _embedder


def _embed_text_sync(text: str) -> list[float]:
    if _embedder is None:
        raise RuntimeError(
            "Embedder not loaded: call load_embedder() at startup "
            "(or set ALLOW_FAKE_EMBEDDER=true and call load_embedder() in tests)."
        )
    value = _embedder.encode(text)
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [float(item) for item in value]


async def embed_text(text: str) -> list[float]:
    # SentenceTransformer.encode is CPU-bound and blocks the event loop; run it
    # in a worker thread so chat cycles aren't stalled. Output is identical to
    # the synchronous path (_embed_text_sync).
    return await asyncio.to_thread(_embed_text_sync, text)


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
        embedding=await embed_text(text),
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
        embedding=await embed_text(content),
        importance_score=interaction_importance(positive_feedback, high_engagement),
    )


async def write_stance_memory(
    memory: ConversationMemoryManager, chat_id: int, user_id: int | None, topic: str, stance: str
) -> None:
    content = f"Expressed {stance} about {topic} to user_{user_id}"
    await memory.write_vector_memory(
        chat_id=chat_id,
        memory_type="stance",
        user_id=user_id,
        content=content,
        embedding=await embed_text(content),
        importance_score=0.7,
    )


async def write_relationship_memory(
    memory: ConversationMemoryManager, chat_id: int, user_id: int, notes: str
) -> None:
    content = f"Relationship with user_{user_id}: {notes}"
    await memory.write_vector_memory(
        chat_id=chat_id,
        memory_type="relationship",
        user_id=user_id,
        content=content,
        embedding=await embed_text(content),
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

    prompt, system = build_self_reflection_prompt(
        identity_summary=current_persona.identity_summary,
        core_beliefs=current_persona.core_beliefs,
        speaking_style=current_persona.speaking_style,
        recent_messages=format_recent_messages(recent_messages),
        feedback=format_feedback(recent_feedback),
    )

    try:
        result = await ai_client.call_perception_model(prompt, system)
        parsed = parse_reflection(result.text)
    except Exception as exc:
        await log.awarning("self_reflection_skipped_model_error", error=str(exc)[:300])
        return  # Don't fail the whole cycle just because reflection couldn't run

    reflection_embedding = await embed_text(parsed.reflection_text)

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
        await memory.upsert_user_relationship(
            chat_id, update.user_id, update.notes, await embed_text(update.notes)
        )
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
    persona_description: str = "",
) -> tuple[list[RetrievedMemory], Any]:
    # Persona-conditioned query: blend persona description with context so
    # retrieved memories are both topically relevant AND stylistically consistent
    # with the character (PersonaRAG technique).
    query_text = (
        f"{persona_description} {current_context_text}".strip()
        if persona_description
        else current_context_text
    )
    memories = await memory.get_relevant_vector_memories(
        chat_id=chat_id,
        query_embedding=await embed_text(query_text),
        top_k=top_k,
    )
    latest_reflection = await memory.get_latest_self_reflection(chat_id)
    record_vector_memory_retrieved(len(memories))
    return memories, latest_reflection
