from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from conversation_engine.config import PromptConfig
from storage.postgres_models import Message

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:  # pragma: no cover
    SentimentIntensityAnalyzer = None


_ANALYZER = SentimentIntensityAnalyzer() if SentimentIntensityAnalyzer else None


@dataclass(frozen=True)
class EnrichedMessage:
    message_id: int
    chat_id: int
    sender_id: int | None
    text: str
    reply_to_message_id: int | None
    sentiment_score: float
    topic_overlap_score: float
    topics: list[str] = field(default_factory=list)
    timestamp: datetime | None = None
    raw_text: str | None = None
    cleaned_text: str | None = None


@dataclass(frozen=True)
class ActiveThread:
    root_message_id: int
    status: str
    urgency: str


@dataclass(frozen=True)
class Brief:
    tension_level: float
    topic_drift: bool
    active_threads: list[ActiveThread]
    summary: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "tension_level": self.tension_level,
            "topic_drift": self.topic_drift,
            "active_threads": [thread.__dict__ for thread in self.active_threads],
            "summary": self.summary,
        }


def sentiment_score(text: str) -> float:
    if not text:
        return 0.0
    if _ANALYZER:
        return float(_ANALYZER.polarity_scores(text)["compound"])
    lowered = text.lower()
    negative = sum(token in lowered for token in ("bad", "hate", "scam", "wrong", "terrible"))
    positive = sum(token in lowered for token in ("good", "great", "love", "nice", "useful"))
    if negative == positive:
        return 0.0
    return max(-1.0, min(1.0, (positive - negative) / 3.0))


def enrich_messages(messages: list[Message], prompt_config: PromptConfig) -> list[EnrichedMessage]:
    topics = [topic.lower() for topic in prompt_config.topics_of_interest]
    enriched: list[EnrichedMessage] = []
    for message in messages:
        text = message.text_cleaned or message.text_raw or ""
        lowered = text.lower()
        matched = [topic for topic in topics if topic in lowered]
        overlap = len(matched) / len(topics) if topics else 0.0
        enriched.append(
            EnrichedMessage(
                message_id=message.message_id,
                chat_id=message.chat_id,
                sender_id=message.sender_id,
                text=text,
                reply_to_message_id=message.reply_to_message_id,
                sentiment_score=sentiment_score(text),
                topic_overlap_score=max(0.0, min(1.0, overlap)),
                topics=matched,
                timestamp=message.timestamp,
                raw_text=message.text_raw,
                cleaned_text=message.text_cleaned,
            )
        )
    return enriched


def build_brief(enriched_messages: list[EnrichedMessage]) -> Brief:
    recent = enriched_messages[-20:]
    if not recent:
        return Brief(tension_level=0.0, topic_drift=False, active_threads=[], summary="")
    negative_count = sum(1 for msg in recent if msg.sentiment_score < -0.35)
    tension = min(1.0, negative_count / max(1, len(recent)) * 2.0)
    thread_ids = [msg.reply_to_message_id for msg in recent if msg.reply_to_message_id]
    active_threads = [
        ActiveThread(root_message_id=thread_id, status="active", urgency="normal")
        for thread_id in sorted(set(thread_ids))[:5]
    ]
    topic_sets = [set(msg.topics) for msg in recent if msg.topics]
    topic_drift = len({topic for topics in topic_sets for topic in topics}) > 3
    summary = "\n".join(f"user_{msg.sender_id}: {msg.text}" for msg in recent[-10:])
    return Brief(
        tension_level=tension,
        topic_drift=topic_drift,
        active_threads=active_threads,
        summary=summary,
    )


def current_context_text(enriched_messages: list[EnrichedMessage]) -> str:
    return "\n".join(f"user_{msg.sender_id}: {msg.text}" for msg in enriched_messages[-50:])
