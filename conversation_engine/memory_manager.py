from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import Select, func, select, text, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from storage.postgres_models import (
    AiDecision,
    BotMemory,
    BotPersonaCore,
    BotSelfReflection,
    BotVectorMemory,
    BriefCache,
    ChatActivityPattern,
    CircuitBreakerState,
    ConversationSummary,
    FailedCycle,
    Message,
    PendingObservation,
    ResponseFeedback,
    StanceTracker,
    UserRelationshipProfile,
)


@dataclass(frozen=True)
class RetrievedMemory:
    content: str
    memory_type: str
    importance_score: float
    similarity: float


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def normalize_embedding(value: Any) -> list[float] | None:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [float(item) for item in value]


def merge_relationship_notes(existing: str | None, new: str, max_length: int = 1000) -> str:
    """Append new distinct note lines to existing notes, dedupe, cap total length."""
    parts = [part.strip() for part in (existing or "").split("\n") if part.strip()]
    for part in (p.strip() for p in new.split("\n")):
        if part and part not in parts:
            parts.append(part)
    while len(parts) > 1 and len("\n".join(parts)) > max_length:
        parts.pop(0)
    return "\n".join(parts)[:max_length]


class ConversationMemoryManager:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def seed_persona_if_empty(
        self,
        identity_summary: str,
        core_beliefs: list[str],
        speaking_style: str,
        embedding: list[float] | None,
    ) -> BotPersonaCore:
        result = await self.session.execute(
            select(BotPersonaCore).order_by(BotPersonaCore.id.desc()).limit(1)
        )
        existing = result.scalar_one_or_none()
        if existing:
            return existing
        row = BotPersonaCore(
            identity_summary=identity_summary,
            core_beliefs=core_beliefs,
            speaking_style=speaking_style,
            embedding=normalize_embedding(embedding),
            version=1,
        )
        self.session.add(row)
        await self.session.flush()
        return row

    async def get_persona_core(self) -> BotPersonaCore | None:
        result = await self.session.execute(
            select(BotPersonaCore).order_by(BotPersonaCore.version.desc()).limit(1)
        )
        return result.scalar_one_or_none()

    async def update_persona_core(
        self, updated_summary: str, embedding: list[float] | None
    ) -> None:
        persona = await self.get_persona_core()
        if not persona:
            return
        await self.session.execute(
            update(BotPersonaCore)
            .where(BotPersonaCore.id == persona.id)
            .values(
                identity_summary=updated_summary,
                embedding=normalize_embedding(embedding),
                updated_at=func.now(),
                version=BotPersonaCore.version + 1,
            )
        )

    async def write_vector_memory(
        self,
        chat_id: int,
        memory_type: str,
        content: str,
        embedding: list[float] | None,
        importance_score: float,
        user_id: int | None = None,
    ) -> BotVectorMemory:
        row = BotVectorMemory(
            chat_id=chat_id,
            memory_type=memory_type,
            user_id=user_id,
            content=content,
            embedding=normalize_embedding(embedding),
            importance_score=max(0.0, min(1.0, importance_score)),
        )
        self.session.add(row)
        await self.session.flush()
        return row

    async def get_relevant_vector_memories(
        self,
        chat_id: int,
        query_embedding: list[float] | None,
        top_k: int,
    ) -> list[RetrievedMemory]:
        if not query_embedding:
            stmt = (
                select(BotVectorMemory)
                .where(BotVectorMemory.chat_id == chat_id)
                .order_by(
                    BotVectorMemory.importance_score.desc(), BotVectorMemory.created_at.desc()
                )
                .limit(top_k)
            )
            result = await self.session.execute(stmt)
            return [
                RetrievedMemory(row.content, row.memory_type, row.importance_score, 0.0)
                for row in result.scalars().all()
            ]

        try:
            distance = BotVectorMemory.embedding.cosine_distance(
                normalize_embedding(query_embedding)
            )
            similarity = (1 - distance).label("similarity")
            # Ebbinghaus decay: recency weight = exp(-0.05 * days_since_created)
            # Older memories decay in rank so recent context dominates.
            days_old = func.extract("epoch", func.now() - BotVectorMemory.created_at) / 86400.0
            score = (
                (1 - distance) * BotVectorMemory.importance_score * func.exp(-0.05 * days_old)
            ).label("score")
            stmt: Select = (
                select(
                    BotVectorMemory.content,
                    BotVectorMemory.memory_type,
                    BotVectorMemory.importance_score,
                    similarity,
                    score,
                )
                .where(BotVectorMemory.chat_id == chat_id)
                .order_by(score.desc())
                .limit(top_k)
            )
            result = await self.session.execute(stmt)
            return [
                RetrievedMemory(
                    row.content, row.memory_type, row.importance_score, float(row.similarity or 0.0)
                )
                for row in result.all()
            ]
        except AttributeError:
            return await self.get_relevant_vector_memories(chat_id, None, top_k)

    async def get_latest_self_reflection(self, chat_id: int) -> BotSelfReflection | None:
        result = await self.session.execute(
            select(BotSelfReflection)
            .where(BotSelfReflection.chat_id == chat_id)
            .order_by(BotSelfReflection.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def insert_self_reflection(
        self,
        chat_id: int,
        trigger: str,
        messages_since_last: int,
        reflection_text: str,
        updated_summary: str,
        drift_score: float,
        embedding: list[float] | None,
    ) -> BotSelfReflection:
        row = BotSelfReflection(
            chat_id=chat_id,
            trigger=trigger,
            messages_since_last=messages_since_last,
            reflection_text=reflection_text,
            updated_summary=updated_summary,
            drift_score=max(0.0, min(1.0, drift_score)),
            embedding=normalize_embedding(embedding),
        )
        self.session.add(row)
        await self.session.flush()
        return row

    async def get_recent_bot_memory(self, chat_id: int, limit: int = 50) -> list[BotMemory]:
        result = await self.session.execute(
            select(BotMemory)
            .where(BotMemory.chat_id == chat_id)
            .order_by(BotMemory.sent_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def count_bot_memory_since_last_reflection(self, chat_id: int) -> int:
        latest = await self.get_latest_self_reflection(chat_id)
        stmt = select(func.count()).select_from(BotMemory).where(BotMemory.chat_id == chat_id)
        if latest:
            stmt = stmt.where(BotMemory.sent_at > latest.created_at)
        result = await self.session.execute(stmt)
        return int(result.scalar_one())

    async def insert_bot_memory(
        self,
        chat_id: int,
        sent_message_id: int | None,
        response_text: str,
        reply_to_user_id: int | None,
        reply_to_message_id: int | None,
        reasoning: str | None,
        tone_calibration: str | None,
        brief_snapshot: dict[str, Any],
        stances: dict[str, Any],
        prompt_version: str,
        cycle_snapshot_message_id: int | None,
        current_posture: str | None = None,
    ) -> BotMemory:
        row = BotMemory(
            chat_id=chat_id,
            sent_message_id=sent_message_id,
            response_text=response_text,
            reply_to_user_id=reply_to_user_id,
            reply_to_message_id=reply_to_message_id,
            reasoning=reasoning,
            tone_calibration=tone_calibration,
            brief_snapshot=brief_snapshot,
            stances=stances,
            prompt_version=prompt_version,
            cycle_snapshot_message_id=cycle_snapshot_message_id,
            current_posture=current_posture,
        )
        self.session.add(row)
        await self.session.flush()
        return row

    async def get_recent_feedback(self, chat_id: int, limit: int = 50) -> list[ResponseFeedback]:
        result = await self.session.execute(
            select(ResponseFeedback)
            .where(ResponseFeedback.chat_id == chat_id)
            .order_by(ResponseFeedback.scored_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_avg_feedback_score(self, chat_id: int, window_hours: int = 24) -> float:
        since = utcnow() - timedelta(hours=window_hours)
        result = await self.session.execute(
            select(func.avg(ResponseFeedback.outcome_score)).where(
                ResponseFeedback.chat_id == chat_id, ResponseFeedback.scored_at >= since
            )
        )
        value = result.scalar_one_or_none()
        return float(value or 0.0)

    async def insert_response_feedback(
        self,
        chat_id: int,
        bot_memory_id: int,
        sent_message_id: int,
        observation_window_end: datetime,
        reply_count: int,
        reaction_count: int,
        reaction_types: list[str],
        follow_up_sentiment: float,
        outcome: str,
        outcome_score: float,
    ) -> ResponseFeedback:
        row = ResponseFeedback(
            chat_id=chat_id,
            bot_memory_id=bot_memory_id,
            sent_message_id=sent_message_id,
            observation_window_end=observation_window_end,
            reply_count=reply_count,
            reaction_count=reaction_count,
            reaction_types=reaction_types,
            follow_up_sentiment=follow_up_sentiment,
            outcome=outcome,
            outcome_score=outcome_score,
        )
        self.session.add(row)
        await self.session.flush()
        return row

    async def get_unprocessed_feedback(
        self, chat_id: int, limit: int = 100
    ) -> list[ResponseFeedback]:
        result = await self.session.execute(
            select(ResponseFeedback)
            .where(ResponseFeedback.chat_id == chat_id, ResponseFeedback.meta_reflected.is_(False))
            .order_by(ResponseFeedback.scored_at.asc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def mark_feedback_reflected(self, feedback_ids: list[int]) -> None:
        if not feedback_ids:
            return
        await self.session.execute(
            update(ResponseFeedback)
            .where(ResponseFeedback.id.in_(feedback_ids))
            .values(meta_reflected=True)
        )

    async def insert_pending_observation(
        self,
        chat_id: int,
        bot_memory_id: int,
        sent_message_id: int,
        due_at: datetime,
        sent_at: datetime | None = None,
    ) -> int:
        row = PendingObservation(
            chat_id=chat_id,
            bot_memory_id=bot_memory_id,
            sent_message_id=sent_message_id,
            sent_at=sent_at,
            due_at=due_at,
        )
        self.session.add(row)
        await self.session.flush()
        return row.id

    async def claim_due_observations(self, now: datetime, limit: int = 20) -> list[dict[str, Any]]:
        """Atomically claim observations whose due_at has passed.

        Returns lightweight dicts (id/chat_id/bot_memory_id/sent_message_id) and
        stamps claimed_at=now on the same rows in this transaction, so a second
        poller (or a re-run at the same ``now``) will not pick them up again.
        """
        result = await self.session.execute(
            select(PendingObservation)
            .where(PendingObservation.due_at <= now, PendingObservation.claimed_at.is_(None))
            .order_by(PendingObservation.due_at.asc())
            .limit(limit)
            .with_for_update(skip_locked=True)
        )
        rows = list(result.scalars().all())
        claimed = []
        for row in rows:
            row.claimed_at = now
            claimed.append(
                {
                    "id": row.id,
                    "chat_id": row.chat_id,
                    "bot_memory_id": row.bot_memory_id,
                    "sent_message_id": row.sent_message_id,
                    "sent_at": row.sent_at,
                }
            )
        if rows:
            await self.session.flush()
        return claimed

    async def delete_pending_observation(self, obs_id: int) -> None:
        await self.session.execute(
            PendingObservation.__table__.delete().where(PendingObservation.id == obs_id)
        )

    async def count_pending_observations(self) -> int:
        result = await self.session.execute(select(func.count()).select_from(PendingObservation))
        return int(result.scalar_one() or 0)

    async def _get_relationship(self, chat_id: int, user_id: int) -> UserRelationshipProfile | None:
        result = await self.session.execute(
            select(UserRelationshipProfile).where(
                UserRelationshipProfile.chat_id == chat_id,
                UserRelationshipProfile.user_id == user_id,
            )
        )
        return result.scalar_one_or_none()

    async def upsert_user_relationship(
        self,
        chat_id: int,
        user_id: int,
        notes: str | None = None,
        embedding: list[float] | None = None,
        sentiment_trend: float | None = None,
        receptiveness_score: float | None = None,
    ) -> None:
        """Update only the fields the caller explicitly provided; merge notes instead of overwriting."""
        existing = await self._get_relationship(chat_id, user_id)
        if existing is None:
            row = UserRelationshipProfile(
                chat_id=chat_id,
                user_id=user_id,
                notes=notes,
                embedding=normalize_embedding(embedding),
                last_interaction_at=utcnow(),
            )
            if sentiment_trend is not None:
                row.sentiment_trend = sentiment_trend
            if receptiveness_score is not None:
                row.receptiveness_score = receptiveness_score
            try:
                # SAVEPOINT so a concurrent first-insert race only rolls back
                # this insert, not the caller's outer transaction.
                async with self.session.begin_nested():
                    self.session.add(row)
                    await self.session.flush()
                return
            except IntegrityError:
                # Lost the race: the row exists now, fall through to update it.
                existing = await self._get_relationship(chat_id, user_id)
        if notes is not None:
            existing.notes = merge_relationship_notes(existing.notes, notes)
        if embedding is not None:
            existing.embedding = normalize_embedding(embedding)
        if sentiment_trend is not None:
            existing.sentiment_trend = sentiment_trend
        if receptiveness_score is not None:
            existing.receptiveness_score = receptiveness_score
        existing.last_interaction_at = utcnow()

    async def record_user_exchange(
        self,
        chat_id: int,
        user_id: int,
        outcome_score: float,
        reply_sentiment: float,
    ) -> None:
        """Update a relationship profile from a real exchange (user replied to the bot).

        Increments total_exchanges, nudges relationship_strength by outcome sign
        (small bounded step), and tracks sentiment_trend as an EMA of reply sentiment.
        """
        step = 0.05 if outcome_score > 0 else -0.05 if outcome_score < 0 else 0.0
        existing = await self._get_relationship(chat_id, user_id)
        if existing is None:
            try:
                # SAVEPOINT so a concurrent first-insert race only rolls back
                # this insert, not the caller's outer transaction.
                async with self.session.begin_nested():
                    self.session.add(
                        UserRelationshipProfile(
                            chat_id=chat_id,
                            user_id=user_id,
                            total_exchanges=1,
                            relationship_strength=max(0.0, min(1.0, 0.1 + step)),
                            sentiment_trend=reply_sentiment,
                            last_interaction_at=utcnow(),
                        )
                    )
                    await self.session.flush()
                return
            except IntegrityError:
                # Lost the race: the row exists now, fall through to update it.
                existing = await self._get_relationship(chat_id, user_id)
        alpha = 0.3
        existing.total_exchanges = (existing.total_exchanges or 0) + 1
        existing.relationship_strength = max(
            0.0, min(1.0, float(existing.relationship_strength or 0.0) + step)
        )
        existing.sentiment_trend = (1 - alpha) * float(
            existing.sentiment_trend or 0.0
        ) + alpha * reply_sentiment
        existing.last_interaction_at = utcnow()

    async def get_relationship_profiles(
        self, chat_id: int, user_ids: list[int]
    ) -> list[UserRelationshipProfile]:
        if not user_ids:
            return []
        result = await self.session.execute(
            select(UserRelationshipProfile).where(
                UserRelationshipProfile.chat_id == chat_id,
                UserRelationshipProfile.user_id.in_(user_ids),
            )
        )
        return list(result.scalars().all())

    async def get_today_callback_count(self, chat_id: int, since: datetime) -> int:
        """Count BotMemory rows in this chat since `since` that recorded a callback.

        Marker contract: a BotMemory row used a memory callback when its
        reasoning text contains the literal "[callback]" tag.
        """
        result = await self.session.execute(
            select(func.count())
            .select_from(BotMemory)
            .where(
                BotMemory.chat_id == chat_id,
                BotMemory.sent_at >= since,
                BotMemory.reasoning.contains("[callback]"),
            )
        )
        return int(result.scalar_one())

    async def avg_relationship_strength(self, chat_id: int, user_ids: list[int]) -> float:
        if not user_ids:
            return 0.0
        profiles = await self.get_relationship_profiles(chat_id, user_ids)
        by_user = {profile.user_id: profile.relationship_strength for profile in profiles}
        values = [float(by_user.get(user_id, 0.0)) for user_id in user_ids]
        return sum(values) / len(values)

    async def count_messages_in_window(self, chat_id: int, minutes: int) -> int:
        since = utcnow() - timedelta(minutes=minutes)
        result = await self.session.execute(
            select(func.count())
            .select_from(Message)
            .where(
                Message.chat_id == chat_id,
                Message.timestamp >= since,
                Message.is_deleted.is_(False),
            )
        )
        return int(result.scalar_one())

    async def count_bot_responses(self, chat_id: int, window_minutes: int) -> int:
        since = utcnow() - timedelta(minutes=window_minutes)
        result = await self.session.execute(
            select(func.count())
            .select_from(BotMemory)
            .where(BotMemory.chat_id == chat_id, BotMemory.sent_at >= since)
        )
        return int(result.scalar_one())

    async def count_bot_responses_in_threads(
        self,
        chat_id: int,
        thread_message_ids: list[int],
        window_minutes: int,
    ) -> int:
        if not thread_message_ids:
            return 0
        since = utcnow() - timedelta(minutes=window_minutes)
        result = await self.session.execute(
            select(func.count())
            .select_from(BotMemory)
            .where(
                BotMemory.chat_id == chat_id,
                BotMemory.reply_to_message_id.in_(thread_message_ids),
                BotMemory.sent_at >= since,
            )
        )
        return int(result.scalar_one())

    async def get_latest_brief(self, chat_id: int) -> BriefCache | None:
        result = await self.session.execute(
            select(BriefCache)
            .where(BriefCache.chat_id == chat_id)
            .order_by(BriefCache.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_activity_pattern(
        self, chat_id: int, hour: int, day: int | None = None
    ) -> ChatActivityPattern | None:
        stmt = select(ChatActivityPattern).where(
            ChatActivityPattern.chat_id == chat_id, ChatActivityPattern.hour_of_day == hour
        )
        if day is not None:
            stmt = stmt.where(ChatActivityPattern.day_of_week == day)
        result = await self.session.execute(
            stmt.order_by(ChatActivityPattern.sample_count.desc()).limit(1)
        )
        return result.scalar_one_or_none()

    async def upsert_activity_pattern(
        self,
        chat_id: int,
        hour_of_day: int,
        day_of_week: int,
        velocity: float,
        tension: float,
    ) -> None:
        stmt = insert(ChatActivityPattern).values(
            chat_id=chat_id,
            hour_of_day=hour_of_day,
            day_of_week=day_of_week,
            avg_message_velocity=velocity,
            avg_tension=tension,
            sample_count=1,
            last_updated=func.now(),
        )
        stmt = stmt.on_conflict_do_update(
            constraint="uq_activity_chat_hour_day",
            set_={
                "avg_message_velocity": (
                    (ChatActivityPattern.avg_message_velocity * ChatActivityPattern.sample_count)
                    + stmt.excluded.avg_message_velocity
                )
                / (ChatActivityPattern.sample_count + 1),
                "avg_tension": (
                    (ChatActivityPattern.avg_tension * ChatActivityPattern.sample_count)
                    + stmt.excluded.avg_tension
                )
                / (ChatActivityPattern.sample_count + 1),
                "sample_count": ChatActivityPattern.sample_count + 1,
                "last_updated": func.now(),
            },
        )
        await self.session.execute(stmt)

    async def initialize_activity_patterns(self, chat_id: int) -> None:
        for day in range(7):
            for hour in range(24):
                stmt = insert(ChatActivityPattern).values(
                    chat_id=chat_id,
                    hour_of_day=hour,
                    day_of_week=day,
                    avg_message_velocity=0.0,
                    avg_tension=0.0,
                    sample_count=0,
                )
                stmt = stmt.on_conflict_do_nothing(constraint="uq_activity_chat_hour_day")
                await self.session.execute(stmt)

    async def insert_ai_decision(
        self,
        chat_id: int,
        prompt_version: str,
        snapshot_message_id: int | None,
        new_message_count: int,
        should_respond: bool,
        confidence: float,
        response_text: str | None,
        reply_to_message_id: int | None,
        reasoning: str | None,
        gate_score: float | None,
        gate_factors: dict[str, Any],
        request1_latency_ms: int | None = None,
        request2_latency_ms: int | None = None,
        request1_tokens_used: int | None = None,
        request2_tokens_used: int | None = None,
    ) -> AiDecision:
        row = AiDecision(
            chat_id=chat_id,
            prompt_version=prompt_version,
            snapshot_message_id=snapshot_message_id,
            new_message_count=new_message_count,
            should_respond=should_respond,
            confidence=confidence,
            response_text=response_text,
            reply_to_message_id=reply_to_message_id,
            reasoning=reasoning,
            gate_score=gate_score,
            gate_factors=gate_factors,
            request1_latency_ms=request1_latency_ms,
            request2_latency_ms=request2_latency_ms,
            request1_tokens_used=request1_tokens_used,
            request2_tokens_used=request2_tokens_used,
        )
        self.session.add(row)
        await self.session.flush()
        return row

    async def update_ai_decision_sent_message(self, decision_id: int, sent_message_id: int) -> None:
        await self.session.execute(
            update(AiDecision)
            .where(AiDecision.id == decision_id)
            .values(sent_message_id=sent_message_id)
        )

    async def get_latest_ai_decision(self, chat_id: int) -> AiDecision | None:
        result = await self.session.execute(
            select(AiDecision)
            .where(AiDecision.chat_id == chat_id)
            .order_by(AiDecision.evaluated_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def count_messages_after_snapshot(
        self, chat_id: int, snapshot_message_id: int | None
    ) -> int:
        stmt = (
            select(func.count())
            .select_from(Message)
            .where(
                Message.chat_id == chat_id,
                Message.is_deleted.is_(False),
            )
        )
        if snapshot_message_id is not None:
            stmt = stmt.where(Message.message_id > snapshot_message_id)
        result = await self.session.execute(stmt)
        return int(result.scalar_one())

    async def latest_message_id(self, chat_id: int) -> int | None:
        result = await self.session.execute(
            select(Message.message_id)
            .where(Message.chat_id == chat_id, Message.is_deleted.is_(False))
            .order_by(Message.message_id.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def insert_failed_cycle(
        self,
        chat_id: int,
        stage: str,
        error_message: str,
        raw_context_sent: str | None,
        prompt_version: str | None,
    ) -> None:
        self.session.add(
            FailedCycle(
                chat_id=chat_id,
                stage=stage,
                error_message=error_message[:4000],
                raw_context_sent=raw_context_sent,
                prompt_version=prompt_version,
            )
        )

    async def is_circuit_paused(self, chat_id: int) -> bool:
        result = await self.session.execute(
            select(CircuitBreakerState).where(CircuitBreakerState.chat_id == chat_id)
        )
        state = result.scalar_one_or_none()
        return bool(state and state.paused_until and state.paused_until > utcnow())

    async def record_cycle_success(self, chat_id: int) -> None:
        stmt = insert(CircuitBreakerState).values(chat_id=chat_id, failure_count=0)
        stmt = stmt.on_conflict_do_update(
            index_elements=["chat_id"],
            set_={"failure_count": 0, "paused_until": None, "last_failure_at": None},
        )
        await self.session.execute(stmt)

    async def record_cycle_failure(
        self, chat_id: int, failure_threshold: int, pause_minutes: int
    ) -> None:
        stmt = insert(CircuitBreakerState).values(
            chat_id=chat_id,
            failure_count=1,
            last_failure_at=func.now(),
            paused_until=None,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["chat_id"],
            set_={
                "failure_count": CircuitBreakerState.failure_count + 1,
                "last_failure_at": func.now(),
                "paused_until": text(
                    f"CASE WHEN circuit_breaker_state.failure_count + 1 >= {int(failure_threshold)} "
                    f"THEN NOW() + INTERVAL '{int(pause_minutes)} minutes' ELSE circuit_breaker_state.paused_until END"
                ),
            },
        )
        await self.session.execute(stmt)

    async def upsert_stance(
        self, chat_id: int, topic: str, stance: str, user_id: int | None = None
    ) -> None:
        self.session.add(
            StanceTracker(chat_id=chat_id, topic=topic, stance=stance, user_id=user_id)
        )

    async def get_recent_messages(self, chat_id: int, limit: int = 100) -> list[Message]:
        result = await self.session.execute(
            select(Message)
            .where(Message.chat_id == chat_id, Message.is_deleted.is_(False))
            .order_by(Message.timestamp.desc())
            .limit(limit)
        )
        return list(reversed(result.scalars().all()))

    async def get_recent_private_chat_ids(
        self, limit: int = 25, active_within_minutes: int = 24 * 60
    ) -> list[int]:
        since = utcnow() - timedelta(minutes=active_within_minutes)
        result = await self.session.execute(
            select(Message.chat_id, func.max(Message.timestamp).label("last_message_at"))
            .where(
                Message.chat_id > 0,
                Message.timestamp >= since,
                Message.is_deleted.is_(False),
            )
            .group_by(Message.chat_id)
            .order_by(text("last_message_at DESC"))
            .limit(limit)
        )
        return [int(row.chat_id) for row in result.all()]

    async def get_messages_after(
        self, chat_id: int, sent_message_id: int, sent_at: datetime, window_minutes: int
    ) -> list[Message]:
        result = await self.session.execute(
            select(Message)
            .where(
                Message.chat_id == chat_id,
                Message.message_id > sent_message_id,
                Message.timestamp >= sent_at,
                Message.timestamp <= sent_at + timedelta(minutes=window_minutes),
                Message.is_deleted.is_(False),
            )
            .order_by(Message.timestamp.asc())
        )
        return list(result.scalars().all())

    async def get_messages_before(
        self, chat_id: int, sent_message_id: int, limit: int = 20
    ) -> list[Message]:
        result = await self.session.execute(
            select(Message)
            .where(
                Message.chat_id == chat_id,
                Message.message_id < sent_message_id,
                Message.is_deleted.is_(False),
            )
            .order_by(Message.message_id.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def count_messages_between(
        self,
        chat_id: int,
        start: datetime,
        end: datetime,
        exclude_message_id: int | None = None,
    ) -> int:
        stmt = (
            select(func.count())
            .select_from(Message)
            .where(
                Message.chat_id == chat_id,
                Message.timestamp >= start,
                Message.timestamp < end,
                Message.is_deleted.is_(False),
            )
        )
        if exclude_message_id is not None:
            stmt = stmt.where(Message.message_id != exclude_message_id)
        result = await self.session.execute(stmt)
        return int(result.scalar_one())

    async def get_replies_to(self, chat_id: int, sent_message_id: int) -> list[Message]:
        result = await self.session.execute(
            select(Message).where(
                Message.chat_id == chat_id,
                Message.reply_to_message_id == sent_message_id,
                Message.is_deleted.is_(False),
            )
        )
        return list(result.scalars().all())

    async def backfill_bot_memory_from_messages(
        self, chat_id: int, bot_user_id: int, prompt_version: str
    ) -> int:
        result = await self.session.execute(
            select(Message)
            .where(
                Message.chat_id == chat_id,
                Message.sender_id == bot_user_id,
                Message.is_deleted.is_(False),
            )
            .order_by(Message.timestamp.asc())
        )
        count = 0
        for message in result.scalars().all():
            exists = await self.session.execute(
                select(func.count())
                .select_from(BotMemory)
                .where(
                    BotMemory.chat_id == chat_id,
                    BotMemory.sent_message_id == message.message_id,
                )
            )
            if int(exists.scalar_one()) > 0:
                continue
            self.session.add(
                BotMemory(
                    chat_id=chat_id,
                    sent_at=message.timestamp,
                    sent_message_id=message.message_id,
                    response_text=message.text_cleaned or message.text_raw or "",
                    reply_to_user_id=None,
                    reply_to_message_id=message.reply_to_message_id,
                    reasoning="backfilled from existing messages",
                    tone_calibration=None,
                    brief_snapshot={},
                    stances={},
                    prompt_version=prompt_version,
                    cycle_snapshot_message_id=message.message_id,
                )
            )
            count += 1
        await self.session.flush()
        return count

    async def count_summaries(self, chat_id: int) -> int:
        result = await self.session.execute(
            select(func.count())
            .select_from(ConversationSummary)
            .where(ConversationSummary.chat_id == chat_id)
        )
        return int(result.scalar_one())

    async def insert_conversation_summary(
        self,
        chat_id: int,
        chunk_start_message_id: int,
        chunk_end_message_id: int,
        summary: str,
        token_count: int,
    ) -> ConversationSummary:
        row = ConversationSummary(
            chat_id=chat_id,
            chunk_start_message_id=chunk_start_message_id,
            chunk_end_message_id=chunk_end_message_id,
            summary=summary,
            token_count=token_count,
        )
        self.session.add(row)
        await self.session.flush()
        return row

    async def recent_timing_decisions(self, hours: float = 168.0) -> list[dict]:
        """Recent AiDecision rows carrying timing telemetry, projected into the shape
        scripts/timing_rate_monitor.summarize() expects. Returns [] if none."""
        since = utcnow() - timedelta(hours=hours)
        result = await self.session.execute(
            select(AiDecision.gate_factors, AiDecision.should_respond).where(
                AiDecision.evaluated_at >= since,
                AiDecision.gate_factors.has_key("timing_p"),  # JSONB ? operator (Postgres)
            )
        )
        rows = []
        for gate_factors, should_respond in result.all():
            gf = gate_factors or {}
            rows.append(
                {
                    "timing_p": gf.get("timing_p"),
                    "timing_would_pass": gf.get("timing_would_pass"),
                    "timing_is_direct": gf.get("timing_is_direct"),
                    "should_respond": should_respond,
                }
            )
        return rows
