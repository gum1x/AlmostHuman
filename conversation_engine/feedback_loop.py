from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from conversation_engine.ai_client import parse_meta_reflection
from conversation_engine.config import EngineConfig
from conversation_engine.enrichment import sentiment_score
from conversation_engine.memory_manager import ConversationMemoryManager, utcnow
from conversation_engine.observability import record_feedback
from conversation_engine.prompts import build_meta_reflection_prompt, build_outcome_scoring_prompt
from core.logging import get_logger
from storage.database import async_session_factory

log = get_logger(__name__)

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


async def ai_score_outcome(
    ai_client, replies: list[Any], reactions: list[Reaction], sentiment_shift: float
) -> tuple[str, float]:
    prompt, system = build_outcome_scoring_prompt(
        replies=[
            getattr(reply, "text_cleaned", None) or getattr(reply, "text_raw", "")
            for reply in replies[:10]
        ],
        reactions=[reaction.__dict__ for reaction in reactions],
        sentiment_shift=sentiment_shift,
    )
    result = await ai_client.call_perception_model(prompt, system)
    data = json.loads(result.text[result.text.find("{") : result.text.rfind("}") + 1])
    return str(data["outcome"]), float(data["score"])


async def score_outcome(
    replies: list[Any],
    reactions: list[Reaction],
    sentiment_shift: float,
    ai_client=None,
    chat_killed: bool = False,
) -> tuple[str, float]:
    positive_reactions = count_positive_emojis(reactions)
    negative_reactions = count_negative_emojis(reactions)

    if len(replies) == 0 and len(reactions) == 0:
        if chat_killed:
            return "killed", -0.6
        return "ignored", 0.0
    reply_sentiment = avg_vader_sentiment(replies)
    if negative_reactions > positive_reactions and reply_sentiment < -0.3:
        return "backlash", -0.8
    if len(replies) > 0:
        # Replies to the bot are scored by what the repliers actually said:
        # hostile -> negative, friendly -> positive, mixed -> proportional.
        if reply_sentiment <= -0.25:
            return "negative", max(-1.0, reply_sentiment)
        if reply_sentiment >= 0.25:
            return "positive", min(1.0, reply_sentiment)
        return "neutral", reply_sentiment
    if positive_reactions > 2:
        return "positive", min(0.5 + positive_reactions * 0.1, 1.0)
    if negative_reactions > positive_reactions and negative_reactions >= 3:
        return "backlash", -0.8
    if negative_reactions > positive_reactions:
        return "negative", -0.4
    if ai_client:
        return await ai_score_outcome(ai_client, replies, reactions, sentiment_shift)
    return "neutral", 0.0


def avg_vader_sentiment(messages: list[Any]) -> float:
    values = [
        sentiment_score(
            getattr(message, "text_cleaned", None) or getattr(message, "text_raw", "") or ""
        )
        for message in messages
    ]
    return sum(values) / len(values) if values else 0.0


class FeedbackLoop:
    def __init__(self, config: EngineConfig, ai_client, sender: "TelegramSender | None" = None):
        self.config = config
        self.ai_client = ai_client
        self.sender = sender
        self._queue: asyncio.Queue[tuple[int, int, int]] = asyncio.Queue()
        self._shutdown = asyncio.Event()
        # Strong refs to fire-and-forget observation tasks so they aren't GC'd
        # mid-flight (asyncio only holds weak refs to running tasks).
        self._bg_tasks: set[asyncio.Task] = set()

    async def schedule_observation(
        self, bot_memory_id: int, sent_message_id: int, chat_id: int, session=None
    ) -> None:
        if self.config.feedback_due_at_enabled:
            sent_at = utcnow()
            due_at = sent_at + timedelta(
                minutes=self.config.feedback_loop.observation_window_minutes
            )
            if session is not None:
                # Caller owns the transaction (its surrounding `async with session.begin()`
                # commits this row atomically with the send-recording write).
                await ConversationMemoryManager(session).insert_pending_observation(
                    chat_id=chat_id,
                    bot_memory_id=bot_memory_id,
                    sent_message_id=sent_message_id,
                    due_at=due_at,
                    sent_at=sent_at,
                )
                return
            async with async_session_factory() as session:
                async with session.begin():
                    memory = ConversationMemoryManager(session)
                    await memory.insert_pending_observation(
                        chat_id=chat_id,
                        bot_memory_id=bot_memory_id,
                        sent_message_id=sent_message_id,
                        due_at=due_at,
                        sent_at=sent_at,
                    )
            return
        await self._queue.put((bot_memory_id, sent_message_id, chat_id))

    async def run_observation_tasks(self) -> None:
        while not self._shutdown.is_set():
            try:
                bot_memory_id, sent_message_id, chat_id = await asyncio.wait_for(
                    self._queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue
            task = asyncio.create_task(
                self.observe_response(bot_memory_id, sent_message_id, chat_id)
            )
            self._bg_tasks.add(task)
            task.add_done_callback(self._bg_tasks.discard)

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
        except Exception as exc:
            await log.awarning(
                "feedback_fetch_reactions_failed",
                chat_id=chat_id,
                message_id=message_id,
                error=str(exc),
            )
            return []

    async def observe_response(
        self, bot_memory_id: int, sent_message_id: int, chat_id: int
    ) -> None:
        window_minutes = self.config.feedback_loop.observation_window_minutes
        window_start = utcnow()
        await asyncio.sleep(window_minutes * 60)
        observation_window_end = window_start + timedelta(minutes=window_minutes)
        async with async_session_factory() as session:
            async with session.begin():
                memory = ConversationMemoryManager(session)
                await self._score_and_record(
                    memory,
                    bot_memory_id=bot_memory_id,
                    sent_message_id=sent_message_id,
                    chat_id=chat_id,
                    window_start=window_start,
                    observation_window_end=observation_window_end,
                )

    async def _score_and_record(
        self,
        memory: ConversationMemoryManager,
        bot_memory_id: int,
        sent_message_id: int,
        chat_id: int,
        window_start: datetime,
        observation_window_end: datetime,
    ) -> None:
        window_minutes = self.config.feedback_loop.observation_window_minutes
        replies = await memory.get_replies_to(chat_id, sent_message_id)
        reactions = await self._fetch_reactions(chat_id, sent_message_id)
        follow_up = await memory.get_messages_after(
            chat_id,
            sent_message_id,
            window_start,
            window_minutes,
        )
        baseline = await memory.get_messages_before(chat_id, sent_message_id, limit=20)
        sentiment_shift = avg_vader_sentiment(follow_up) - avg_vader_sentiment(baseline)
        chat_killed = False
        if len(replies) == 0:
            before_count = await memory.count_messages_between(
                chat_id,
                window_start - timedelta(minutes=10),
                window_start,
                exclude_message_id=sent_message_id,
            )
            after_count = await memory.count_messages_between(
                chat_id,
                window_start,
                window_start + timedelta(minutes=10),
                exclude_message_id=sent_message_id,
            )
            # Velocity-drop rule: an absolute after_count cutoff almost never
            # fires in busy chats; killed = activity dropped to <=20% of the
            # pre-send rate.
            chat_killed = before_count >= 5 and after_count <= 0.2 * before_count
        outcome, score = await score_outcome(
            replies, reactions, sentiment_shift, self.ai_client, chat_killed=chat_killed
        )
        await memory.insert_response_feedback(
            chat_id=chat_id,
            bot_memory_id=bot_memory_id,
            sent_message_id=sent_message_id,
            observation_window_end=observation_window_end,
            reply_count=len(replies),
            reaction_count=sum(reaction.count for reaction in reactions),
            reaction_types=[reaction.emoji for reaction in reactions],
            follow_up_sentiment=sentiment_shift,
            outcome=outcome,
            outcome_score=score,
        )
        replies_by_user: dict[int, list[Any]] = defaultdict(list)
        for reply in replies:
            if reply.sender_id is not None:
                replies_by_user[int(reply.sender_id)].append(reply)
        for user_id, user_replies in replies_by_user.items():
            await memory.record_user_exchange(
                chat_id=chat_id,
                user_id=user_id,
                outcome_score=score,
                reply_sentiment=avg_vader_sentiment(user_replies),
            )
        record_feedback(outcome, score)

    async def run_due_observation_loop(self, poll_interval: float = 30.0) -> None:
        """DB-backed poller: claim overdue pending_observations and score them.

        Recovery on restart is automatic — due rows persist, so the poller picks up
        any observation whose window elapsed while the process was down.
        """
        window_minutes = self.config.feedback_loop.observation_window_minutes
        while True:
            # Poll first so an overdue row left by a previous run is picked up
            # immediately on startup (restart recovery).
            now = utcnow()
            # Claim in one short transaction (commit the claimed_at stamps), then
            # score+delete each row in its OWN transaction. Per-row isolation matches
            # the legacy fire-and-forget tasks: one failing observation (e.g. an AI
            # timeout in scoring) can't roll back its batch-mates' deletes and cause
            # reprocessing, and a persistently-failing row can't block the batch.
            async with async_session_factory() as session:
                async with session.begin():
                    claimed = await ConversationMemoryManager(session).claim_due_observations(now)
            for obs in claimed:
                # Anchor the scoring window on the actual SEND time, not claim time.
                # After a restart, overdue rows are claimed long after due_at, so `now`
                # is far past the real send; sampling from `now` would score the wrong
                # interval. Fall back to claim-time only if sent_at is somehow missing.
                window_start = obs.get("sent_at") or (now - timedelta(minutes=window_minutes))
                try:
                    async with async_session_factory() as session:
                        async with session.begin():
                            memory = ConversationMemoryManager(session)
                            await self._score_and_record(
                                memory,
                                bot_memory_id=obs["bot_memory_id"],
                                sent_message_id=obs["sent_message_id"],
                                chat_id=obs["chat_id"],
                                window_start=window_start,
                                observation_window_end=now,
                            )
                            await memory.delete_pending_observation(obs["id"])
                except Exception as exc:
                    # Drop the failed observation (parity with legacy task loss) so it
                    # neither reprocesses nor lingers; one bad row can't wedge the loop.
                    await log.aerror(
                        "feedback_due_observation_failed", obs_id=obs["id"], error=str(exc)
                    )
                    try:
                        async with async_session_factory() as session:
                            async with session.begin():
                                await ConversationMemoryManager(session).delete_pending_observation(
                                    obs["id"]
                                )
                    except Exception as cleanup_exc:
                        await log.awarning(
                            "feedback_due_observation_cleanup_failed",
                            obs_id=obs["id"],
                            error=str(cleanup_exc),
                        )
            if self._shutdown.is_set():
                return
            try:
                await asyncio.wait_for(self._shutdown.wait(), timeout=poll_interval)
            except asyncio.TimeoutError:
                continue
            return


def aggregate_feedback(feedback_rows: list[Any]) -> dict[str, Any]:
    by_time: dict[int, list[float]] = defaultdict(list)
    by_outcome: dict[str, int] = defaultdict(int)
    direct_scores: list[float] = []
    ambient_scores: list[float] = []
    short_scores: list[float] = []  # reply_count == 0 and reaction_count > 0
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


async def run_meta_reflection(
    chat_id: int, memory: ConversationMemoryManager, ai_client, config: EngineConfig
) -> None:
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
        await memory.upsert_user_relationship(
            chat_id, pref.user_id, f"Preferred tone: {pref.preferred_tone}"
        )
    await memory.mark_feedback_reflected([row.id for row in unprocessed])
