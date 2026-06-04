from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from conversation_engine.ai_client import parse_meta_reflection
from conversation_engine.config import EngineConfig
from conversation_engine.enrichment import sentiment_score
from conversation_engine.memory_manager import ConversationMemoryManager, utcnow
from conversation_engine.observability import record_feedback
from conversation_engine.prompts import build_meta_reflection_prompt, build_outcome_scoring_prompt
from storage.database import async_session_factory

if TYPE_CHECKING:
    from conversation_engine.sender import TelegramSender


POSITIVE_EMOJIS = {"👍", "❤️", "🔥", "👏", "💯", "😂", "🤣", "✅"}
NEGATIVE_EMOJIS = {"👎", "😡", "🤡", "💩", "❌"}


@dataclass(frozen=True)
class Reaction:
    emoji: str
    count: int = 1


def count_positive_emojis(reactions: list[Reaction]) -> int:
    return sum(reaction.count for reaction in reactions if reaction.emoji in POSITIVE_EMOJIS)


def count_negative_emojis(reactions: list[Reaction]) -> int:
    return sum(reaction.count for reaction in reactions if reaction.emoji in NEGATIVE_EMOJIS)


async def ai_score_outcome(ai_client, replies: list[Any], reactions: list[Reaction], sentiment: float) -> tuple[str, float]:
    prompt, system = build_outcome_scoring_prompt(
        replies=[getattr(reply, "text_cleaned", None) or getattr(reply, "text_raw", "") for reply in replies[:10]],
        reactions=[reaction.__dict__ for reaction in reactions],
        sentiment=sentiment,
    )
    result = await ai_client.call_perception_model(prompt, system)
    data = json.loads(result.text[result.text.find("{") : result.text.rfind("}") + 1])
    return str(data["outcome"]), float(data["score"])


async def score_outcome(
    replies: list[Any],
    reactions: list[Reaction],
    quote_replies: list[Any],
    sentiment: float,
    ai_client=None,
) -> tuple[str, float]:
    positive_reactions = count_positive_emojis(reactions)
    negative_reactions = count_negative_emojis(reactions)

    if len(replies) == 0 and len(reactions) == 0:
        return "ignored", 0.0
    if negative_reactions > positive_reactions and sentiment < -0.3:
        return "backlash", -0.8
    if len(quote_replies) > 0 or positive_reactions > 2:
        return "positive", min(0.5 + (len(quote_replies) * 0.2), 1.0)
    if sentiment < -0.4:
        return "negative", -0.4
    if len(replies) > 0 and sentiment > -0.2:
        return "neutral", 0.2
    if ai_client:
        return await ai_score_outcome(ai_client, replies, reactions, sentiment)
    return "neutral", 0.0


def avg_vader_sentiment(messages: list[Any]) -> float:
    values = [sentiment_score(getattr(message, "text_cleaned", None) or getattr(message, "text_raw", "") or "") for message in messages]
    return sum(values) / len(values) if values else 0.0


class FeedbackLoop:
    def __init__(self, config: EngineConfig, ai_client, sender: "TelegramSender | None" = None):
        self.config = config
        self.ai_client = ai_client
        self.sender = sender
        self._queue: asyncio.Queue[tuple[int, int, int]] = asyncio.Queue()
        self._shutdown = asyncio.Event()

    async def schedule_observation(self, bot_memory_id: int, sent_message_id: int, chat_id: int) -> None:
        await self._queue.put((bot_memory_id, sent_message_id, chat_id))

    async def run_observation_tasks(self) -> None:
        while not self._shutdown.is_set():
            try:
                bot_memory_id, sent_message_id, chat_id = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            asyncio.create_task(self.observe_response(bot_memory_id, sent_message_id, chat_id))

    def shutdown(self) -> None:
        self._shutdown.set()

    async def _fetch_reactions(self, chat_id: int, message_id: int) -> list[Reaction]:
        """Fetch actual emoji reactions from Telegram for a sent message."""
        if self.sender is None:
            return []
        try:
            msg = await self.sender.client.get_messages(chat_id, ids=message_id)
            if msg is None or getattr(msg, "reactions", None) is None:
                return []
            reactions = []
            for rc in msg.reactions.results:
                emoji = getattr(rc.reaction, "emoticon", None)
                if emoji:
                    reactions.append(Reaction(emoji=emoji, count=rc.count))
            return reactions
        except Exception:
            return []

    async def observe_response(self, bot_memory_id: int, sent_message_id: int, chat_id: int) -> None:
        window_minutes = self.config.feedback_loop.observation_window_minutes
        window_start = utcnow()
        await asyncio.sleep(window_minutes * 60)
        observation_window_end = window_start + timedelta(minutes=window_minutes)
        async with async_session_factory() as session:
            async with session.begin():
                memory = ConversationMemoryManager(session)
                replies = await memory.get_replies_to(chat_id, sent_message_id)
                reactions = await self._fetch_reactions(chat_id, sent_message_id)
                quote_replies = [
                    reply
                    for reply in replies
                    if reply.reply_to_message_id is not None
                    and int(reply.reply_to_message_id) == int(sent_message_id)
                ]
                follow_up = await memory.get_messages_after(
                    chat_id,
                    sent_message_id,
                    window_minutes,
                )
                sentiment = avg_vader_sentiment(follow_up)
                outcome, score = await score_outcome(replies, reactions, quote_replies, sentiment, self.ai_client)
                await memory.insert_response_feedback(
                    chat_id=chat_id,
                    bot_memory_id=bot_memory_id,
                    sent_message_id=sent_message_id,
                    observation_window_end=observation_window_end,
                    reply_count=len(replies),
                    reaction_count=sum(reaction.count for reaction in reactions),
                    reaction_types=[reaction.emoji for reaction in reactions],
                    quote_reply_count=len(quote_replies),
                    follow_up_sentiment=sentiment,
                    outcome=outcome,
                    outcome_score=score,
                )
                record_feedback(outcome, score)


def aggregate_feedback(feedback_rows: list[Any]) -> dict[str, Any]:
    by_time: dict[int, list[float]] = defaultdict(list)
    by_outcome: dict[str, int] = defaultdict(int)
    direct_scores: list[float] = []
    ambient_scores: list[float] = []
    short_scores: list[float] = []   # reply_count == 0 and reaction_count > 0
    scores = []
    for row in feedback_rows:
        scores.append(row.outcome_score)
        by_time[row.scored_at.hour].append(row.outcome_score)
        by_outcome[row.outcome] += 1
        # Direct = bot was explicitly replied to (reply_count > 0)
        if row.reply_count > 0:
            direct_scores.append(row.outcome_score)
        else:
            ambient_scores.append(row.outcome_score)
        # Short = reactions only, no text replies
        if row.reply_count == 0 and row.reaction_count > 0:
            short_scores.append(row.outcome_score)
    return {
        "by_time_of_day": {hour: sum(values) / len(values) for hour, values in by_time.items()},
        "by_outcome_type": dict(by_outcome),
        "direct_avg_score": sum(direct_scores) / len(direct_scores) if direct_scores else None,
        "ambient_avg_score": sum(ambient_scores) / len(ambient_scores) if ambient_scores else None,
        "reaction_only_avg_score": sum(short_scores) / len(short_scores) if short_scores else None,
        "overall_trend": sum(scores) / len(scores) if scores else 0.0,
        "count": len(feedback_rows),
    }


async def run_meta_reflection(chat_id: int, memory: ConversationMemoryManager, ai_client, config: EngineConfig) -> None:
    unprocessed = await memory.get_unprocessed_feedback(chat_id)
    if len(unprocessed) < 10:
        return
    aggregated = aggregate_feedback(unprocessed)
    prompt, system = build_meta_reflection_prompt(len(unprocessed), aggregated)
    result = await ai_client.call_perception_model(prompt, system)
    parsed = parse_meta_reflection(result.text)
    for rec in parsed.updated_stance_recommendations:
        await memory.upsert_stance(chat_id, topic=rec.topic, stance=rec.recommended_approach)
    for pref in parsed.tone_preferences_by_user:
        await memory.upsert_user_relationship(chat_id, pref.user_id, f"Preferred tone: {pref.preferred_tone}")
    await memory.mark_feedback_reflected([row.id for row in unprocessed])
