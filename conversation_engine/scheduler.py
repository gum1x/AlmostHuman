from __future__ import annotations

import asyncio
import re
import signal
from datetime import datetime, timezone

from conversation_engine.ai_client import (
    AnthropicAiClient,
    FakeAiClient,
    parse_perception,
    parse_response_decision,
)
from conversation_engine.bootstrap import run_bootstrap
from conversation_engine.config import EngineConfig, load_engine_config
from conversation_engine.context_builder import (
    build_context,
    build_request2_constraints,
    build_target_message_block,
)
from conversation_engine.engagement_gate import GateResult, compute_gate_score
from conversation_engine.enrichment import build_brief, current_context_text, enrich_messages
from conversation_engine.feedback_loop import FeedbackLoop, run_meta_reflection
from conversation_engine.memory_manager import ConversationMemoryManager
from conversation_engine.persona_engine import (
    get_relevant_persona_vectors,
    load_embedder,
    seed_persona_core,
    should_run_self_reflection,
    run_self_reflection,
    write_interaction_memory,
    write_stance_memory,
)
from conversation_engine.prompts import build_perception_prompt, build_response_decision_prompt
from conversation_engine.sender import TelegramSender
from conversation_engine.style_rewriter import LocalStyleRewriter
from conversation_engine.validators import validate
from core.logging import get_logger, setup_logging
from storage.database import async_session_factory, dispose_engine

log = get_logger(__name__)


_UNSAFE_CASUAL_TERMS = {
    "otp",
    "sim",
    "swap",
    "bank",
    "cc",
    "card",
    "cvv",
    "ssn",
    "fraud",
    "scam",
    "steal",
    "stolen",
    "phish",
    "dox",
    "doxx",
    "combo",
    "config",
    "cashapp",
    "paypal",
    "dirty",
    "launder",
    "tumbler",
    "mixer",
    "monero",
    "logs",
    "method",
}

_CASUAL_REPLIES = {
    "hi": "yo",
    "hii": "yo",
    "hello": "yo",
    "hey": "yo",
    "yo": "yo",
    "yoo": "yo",
    "yooo": "yo",
    "yoyoyo": "yo",
    "sup": "sup",
    "wsup": "sup",
    "wsp": "sup",
    "whats up": "not much wbu",
    "what's up": "not much wbu",
    "gm": "gm",
    "gn": "gn",
    "lol": "lmao",
    "lmao": "lmao",
    "lmfao": "lmao",
}

_SLANG_ALIASES = {
    "wsg": "whats good",
    "hyd": "how you doing",
    "hbu": "how about you",
    "wbu": "what about you",
    "rn": "right now",
    "u": "you",
    "ur": "your",
    "r": "are",
    "idk": "i dont know",
    "ngl": "not gonna lie",
    "fr": "for real",
    "wyd": "what you doing",
}

_CLOSERS = {"np", "yep", "yes", "ok", "okay", "k", "thanks", "thank you", "ty", "bet"}


def _casual_key(text: str) -> str:
    key = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]+", " ", text.lower())).strip()
    words = [_SLANG_ALIASES.get(word, word) for word in key.split()]
    return " ".join(words)


def _safe_casual_reply(text: str) -> str | None:
    key = _casual_key(text)
    if not key or any(term in key.split() for term in _UNSAFE_CASUAL_TERMS):
        return None
    if key in _CASUAL_REPLIES:
        return _CASUAL_REPLIES[key]
    if key in {"test", "testing"}:
        return "i'm here"
    if key in {"you there", "u there", "are you there"}:
        return "yeah"
    return None


def _contains_unsafe_terms(text: str) -> bool:
    return bool(set(_casual_key(text).split()) & _UNSAFE_CASUAL_TERMS)


def _looks_like_question(text: str) -> bool:
    key = _casual_key(text)
    if "?" in text:
        return True
    return key.startswith((
        "what ",
        "why ",
        "who ",
        "how ",
        "where ",
        "when ",
        "can ",
        "could ",
        "should ",
        "is ",
        "are ",
        "do ",
        "does ",
        "what you doing",
        "how you doing",
        "whats good",
    ))


def _is_social_hook(text: str) -> bool:
    key = _casual_key(text)
    if key in _CLOSERS:
        return False
    if _safe_casual_reply(text):
        return True
    return any(phrase in key for phrase in ("bored", "dead chat", "what you doing", "how you doing", "whats good"))


def _human_motive_reply(text: str, *, is_direct: bool, is_private_dm: bool) -> tuple[str, str, float] | None:
    key = _casual_key(text)
    if not key:
        return None

    unsafe = _contains_unsafe_terms(text)
    if unsafe and (is_direct or is_private_dm):
        return "can't help with that", "direct unsafe request; brief refusal", 0.85
    if unsafe:
        return None

    casual = _safe_casual_reply(text)
    if casual:
        return casual, "human motive: simple greeting/banter", 0.9

    if is_private_dm:
        if key in _CLOSERS:
            return None
        if _looks_like_question(text):
            if "what you doing" in key or "how you doing" in key or "whats good" in key:
                return "chillin wbu", "human motive: DM small talk", 0.8
            return "idk tbh", "human motive: DM question, honest short answer", 0.75
        if len(key.split()) <= 5:
            return "yeah", "human motive: keep DM conversation alive", 0.7
        return "what happened", "human motive: DM user seems to want to chat", 0.7

    if is_direct:
        if _looks_like_question(text):
            if "what is this group" in key or "what's this group" in key:
                return "idk i just got here", "human motive: directly asked about group", 0.8
            if "what you doing" in key or "how you doing" in key or "whats good" in key:
                return "chillin", "human motive: direct small talk", 0.8
            return "idk tbh", "human motive: direct question to bot", 0.75
        if len(key.split()) <= 8:
            return "yeah?", "human motive: direct mention/reply deserves acknowledgement", 0.75
        return "what happened", "human motive: direct follow-up needs a reply", 0.75

    if len(key.split()) <= 4 and key in {"im bored", "i am bored", "dead chat", "this chat dead"}:
        return "same", "human motive: bored/open social chat", 0.65

    return None


class ConversationScheduler:
    def __init__(
        self,
        config: EngineConfig,
        ai_client,
        sender: TelegramSender,
        feedback_loop: FeedbackLoop,
        bot_user_id: int | None = None,
        bot_username: str | None = None,
        style_rewriter: LocalStyleRewriter | None = None,
    ):
        self.config = config
        self.ai_client = ai_client
        self.sender = sender
        self.feedback_loop = feedback_loop
        self.bot_user_id = bot_user_id
        self.bot_username = bot_username.lower() if bot_username else None
        self.style_rewriter = style_rewriter or LocalStyleRewriter(config)
        self._shutdown = asyncio.Event()

    def shutdown(self) -> None:
        self._shutdown.set()
        self.feedback_loop.shutdown()

    async def run(self) -> None:
        tasks: dict[int, asyncio.Task] = {}
        for chat_id in self.config.active_chat_ids:
            tasks[chat_id] = asyncio.create_task(self._chat_loop(chat_id))
        try:
            while not self._shutdown.is_set():
                if self.config.scheduler.monitor_private_dms:
                    for chat_id in await self._discover_private_chat_ids():
                        if chat_id not in tasks:
                            tasks[chat_id] = asyncio.create_task(self._chat_loop(chat_id))
                            await log.ainfo("dm_chat_monitoring_started", chat_id=chat_id)
                done_ids = [chat_id for chat_id, task in tasks.items() if task.done()]
                for chat_id in done_ids:
                    tasks.pop(chat_id, None)
                await asyncio.sleep(self.config.scheduler.dm_discovery_interval_seconds)
        finally:
            for task in tasks.values():
                task.cancel()
            await asyncio.gather(*tasks.values(), return_exceptions=True)

    async def _discover_private_chat_ids(self) -> list[int]:
        async with async_session_factory() as session:
            async with session.begin():
                memory = ConversationMemoryManager(session)
                return await memory.get_recent_private_chat_ids(
                    limit=self.config.scheduler.dm_max_active_chats,
                )

    async def _chat_loop(self, chat_id: int) -> None:
        interval = self.config.scheduler.initial_interval_seconds
        while not self._shutdown.is_set():
            interval = await self._run_cycle(chat_id, interval)
            await asyncio.sleep(interval)

    async def _run_cycle(self, chat_id: int, previous_interval: int) -> int:
        raw_context: str | None = None
        is_private_dm = chat_id > 0
        active_bot_thread = False
        new_message_threshold = (
            self.config.scheduler.dm_new_message_threshold
            if is_private_dm
            else self.config.scheduler.new_message_threshold
        )
        recent_message_limit = (
            self.config.scheduler.dm_recent_message_limit
            if is_private_dm
            else 50
        )
        async with async_session_factory() as session:
            async with session.begin():
                memory = ConversationMemoryManager(session)
                if await memory.is_circuit_paused(chat_id):
                    return self.config.scheduler.max_interval_seconds

                try:
                    latest_decision = await memory.get_latest_ai_decision(chat_id)
                    snapshot_before = latest_decision.snapshot_message_id if latest_decision else None
                    new_message_count = await memory.count_messages_after_snapshot(chat_id, snapshot_before)
                    active_bot_thread = await self._has_new_user_followup_after_bot(memory, chat_id, snapshot_before)
                    if new_message_count < new_message_threshold:
                        if not active_bot_thread:
                            return min(
                                self.config.scheduler.max_interval_seconds,
                                int(previous_interval * self.config.scheduler.backoff_multiplier),
                            )
                        new_message_count = max(1, new_message_count)

                    await seed_persona_core(memory, self.config)
                    if not is_private_dm:
                        reflection_needed, trigger, messages_since_last = await should_run_self_reflection(
                            memory, chat_id, self.config
                        )
                        if reflection_needed:
                            await run_self_reflection(
                                chat_id=chat_id,
                                memory=memory,
                                ai_client=self.ai_client,
                                config=self.config,
                                trigger=trigger,
                                messages_since_last=messages_since_last,
                            )
                        await run_meta_reflection(chat_id, memory, self.ai_client, self.config)

                    messages = await memory.get_recent_messages(chat_id, limit=recent_message_limit)
                    enriched = enrich_messages(messages, self.config.prompt)
                    brief = build_brief(enriched)
                    if is_private_dm:
                        gate = GateResult(
                            gate_score=1.0,
                            gate_factors={"mode": "private_dm"},
                            should_proceed=True,
                        )
                    else:
                        gate = await compute_gate_score(chat_id, enriched, brief, memory, self.config)
                    outcome_score_24h = await memory.get_avg_feedback_score(chat_id, window_hours=24)
                    visible_numeric_controls = {
                        "tension_level": brief.tension_level,
                        "outcome_score_24h": outcome_score_24h,
                    }
                    snapshot_message_id = await memory.latest_message_id(chat_id)
                    casual_sent = await self._try_safe_casual_reply(
                        chat_id=chat_id,
                        is_private_dm=is_private_dm,
                        enriched=enriched,
                        memory=memory,
                        snapshot_before=snapshot_before,
                        snapshot_message_id=snapshot_message_id,
                        new_message_count=new_message_count,
                        brief=brief,
                        gate_score=gate.gate_score,
                        gate_factors=visible_numeric_controls,
                    )
                    if casual_sent:
                        await memory.record_cycle_success(chat_id)
                        return self.config.scheduler.initial_interval_seconds

                    human_sent = await self._try_human_motive_reply(
                        chat_id=chat_id,
                        is_private_dm=is_private_dm,
                        active_bot_thread=active_bot_thread,
                        enriched=enriched,
                        memory=memory,
                        snapshot_before=snapshot_before,
                        snapshot_message_id=snapshot_message_id,
                        new_message_count=new_message_count,
                        brief=brief,
                        gate_score=gate.gate_score,
                        gate_factors=visible_numeric_controls,
                    )
                    if human_sent:
                        await memory.record_cycle_success(chat_id)
                        return self.config.scheduler.initial_interval_seconds

                    now = datetime.now(timezone.utc)
                    await memory.upsert_activity_pattern(
                        chat_id,
                        hour_of_day=now.hour,
                        day_of_week=now.weekday(),
                        velocity=new_message_count / max(1, self.config.engagement_gate.velocity_window_minutes),
                        tension=brief.tension_level,
                    )

                    if not gate.should_proceed:
                        await memory.insert_ai_decision(
                            chat_id=chat_id,
                            prompt_version=self.config.ai.prompt_version,
                            snapshot_message_id=snapshot_message_id,
                            new_message_count=new_message_count,
                            should_respond=False,
                            confidence=0.0,
                            response_text=None,
                            reply_to_message_id=None,
                            reasoning="engagement gate blocked",
                            gate_score=gate.gate_score,
                            gate_factors=visible_numeric_controls,
                        )
                        await memory.record_cycle_success(chat_id)
                        return self.config.scheduler.initial_interval_seconds

                    await memory.insert_ai_decision(
                        chat_id=chat_id,
                        prompt_version=self.config.ai.prompt_version,
                        snapshot_message_id=snapshot_message_id,
                        new_message_count=new_message_count,
                        should_respond=False,
                        confidence=0.0,
                        response_text=None,
                        reply_to_message_id=None,
                        reasoning="local human-motive decision: no direct address, no open social hook, or unsafe ambient context",
                        gate_score=gate.gate_score,
                        gate_factors=visible_numeric_controls,
                        request1_tokens_used=0,
                        request2_tokens_used=0,
                    )
                    await memory.record_cycle_success(chat_id)
                    return self.config.scheduler.initial_interval_seconds

                    persona_memories, latest_reflection = await get_relevant_persona_vectors(
                        chat_id,
                        current_context_text(enriched),
                        memory,
                        top_k=self.config.ai.persona_top_k,
                    )
                    current_persona = await memory.get_persona_core()
                    context = await build_context(
                        chat_id,
                        enriched,
                        brief,
                        gate,
                        memory,
                        persona_memories,
                        latest_reflection,
                        current_persona,
                    )
                    raw_context = context.context
                    if active_bot_thread:
                        raw_context = (
                            f"{context.context}\n\n"
                            "=== ACTIVE BOT THREAD ===\n"
                            "The bot recently sent a message and at least one user replied or spoke after it, or a user directly mentioned the bot. "
                            "Bias toward responding to the user as part of an ongoing conversation. "
                            "Use a short natural reply. Do not ignore direct follow-ups just because the wider chat is noisy."
                        )
                        context = type(context)(
                            context=raw_context,
                            candidate_user_ids=context.candidate_user_ids,
                            relationship_profiles=context.relationship_profiles,
                            avg_feedback_score=context.avg_feedback_score,
                        )
                    perception_prompt, perception_system = build_perception_prompt(context, self.config)
                    request1 = await self.ai_client.call_perception_model(perception_prompt, perception_system)
                    perception = parse_perception(request1.text)
                    if not perception.should_respond:
                        fallback_sent = await self._try_direct_attention_fallback(
                            chat_id=chat_id,
                            active_bot_thread=active_bot_thread,
                            enriched=enriched,
                            memory=memory,
                            snapshot_before=snapshot_before,
                            snapshot_message_id=snapshot_message_id,
                            new_message_count=new_message_count,
                            brief=brief,
                            gate_score=gate.gate_score,
                            gate_factors=visible_numeric_controls,
                            request1_latency_ms=request1.latency_ms,
                            request1_tokens_used=request1.tokens_used,
                            reasoning=perception.reasoning,
                        )
                        if fallback_sent:
                            await memory.record_cycle_success(chat_id)
                            return self.config.scheduler.initial_interval_seconds
                        await memory.insert_ai_decision(
                            chat_id=chat_id,
                            prompt_version=self.config.ai.prompt_version,
                            snapshot_message_id=snapshot_message_id,
                            new_message_count=new_message_count,
                            should_respond=False,
                            confidence=perception.confidence,
                            response_text=None,
                            reply_to_message_id=None,
                            reasoning=perception.reasoning,
                            gate_score=gate.gate_score,
                            gate_factors=visible_numeric_controls,
                            request1_latency_ms=request1.latency_ms,
                            request1_tokens_used=request1.tokens_used,
                        )
                        await memory.record_cycle_success(chat_id)
                        return self.config.scheduler.initial_interval_seconds

                    constraints = build_request2_constraints(
                        current_persona=current_persona,
                        latest_reflection=latest_reflection,
                        meta_reflection=None,
                        relationship_profiles=context.relationship_profiles,
                        target_message_block=build_target_message_block(
                            enriched,
                            perception.entry_points
                            or ([perception.target_message_id] if perception.target_message_id else []),
                        ),
                    )
                    decision_prompt, decision_system = build_response_decision_prompt(
                        context,
                        constraints,
                        perception,
                        self.config,
                    )
                    request2 = await self.ai_client.call_decision_model(decision_prompt, decision_system)
                    decision = parse_response_decision(request2.text)
                    if decision.should_respond and decision.response_text:
                        decision.response_text = await self.style_rewriter.rewrite(
                            context=context.context,
                            decision=decision.reasoning,
                            draft=decision.response_text,
                        )
                    ok, reason = validate(decision, self.config)
                    stored_decision = await memory.insert_ai_decision(
                        chat_id=chat_id,
                        prompt_version=self.config.ai.prompt_version,
                        snapshot_message_id=snapshot_message_id,
                        new_message_count=new_message_count,
                        should_respond=ok,
                        confidence=decision.confidence,
                        response_text=decision.response_text,
                        reply_to_message_id=decision.reply_to_message_id,
                        reasoning=decision.reasoning if ok else reason,
                        gate_score=gate.gate_score,
                        gate_factors=visible_numeric_controls,
                        request1_latency_ms=request1.latency_ms,
                        request2_latency_ms=request2.latency_ms,
                        request1_tokens_used=request1.tokens_used,
                        request2_tokens_used=request2.tokens_used,
                    )
                    if not ok:
                        await memory.record_cycle_success(chat_id)
                        return self.config.scheduler.initial_interval_seconds

                    sent_message_id = await self.sender.send_message(
                        chat_id,
                        decision.response_text or "",
                        decision.reply_to_message_id,
                    )
                    await memory.update_ai_decision_sent_message(stored_decision.id, sent_message_id)
                    bot_memory = await memory.insert_bot_memory(
                        chat_id=chat_id,
                        sent_message_id=sent_message_id,
                        response_text=decision.response_text or "",
                        reply_to_user_id=decision.reply_to_user_id,
                        reply_to_message_id=decision.reply_to_message_id,
                        reasoning=decision.reasoning,
                        tone_calibration=decision.tone_calibration,
                        brief_snapshot=brief.as_dict(),
                        stances=decision.stances,
                        prompt_version=self.config.ai.prompt_version,
                        cycle_snapshot_message_id=snapshot_message_id,
                    )
                    await write_interaction_memory(
                        memory,
                        chat_id,
                        decision.reply_to_user_id,
                        perception.topic,
                        decision.response_text or "",
                    )
                    for topic, stance in decision.stances.items():
                        await memory.upsert_stance(chat_id, topic=topic, stance=str(stance), user_id=decision.reply_to_user_id)
                        await write_stance_memory(memory, chat_id, decision.reply_to_user_id, topic, str(stance))
                    await self.feedback_loop.schedule_observation(bot_memory.id, sent_message_id, chat_id)
                    await memory.record_cycle_success(chat_id)
                    return self.config.scheduler.initial_interval_seconds
                except Exception as exc:
                    await memory.insert_failed_cycle(
                        chat_id=chat_id,
                        stage="cycle",
                        error_message=str(exc),
                        raw_context_sent=raw_context,
                        prompt_version=self.config.ai.prompt_version,
                    )
                    await memory.record_cycle_failure(
                        chat_id,
                        self.config.circuit_breaker.failure_threshold,
                        self.config.circuit_breaker.pause_duration_minutes,
                    )
                    await log.aexception("conversation_cycle_failed", chat_id=chat_id)
                    return min(
                        self.config.scheduler.max_interval_seconds,
                        int(previous_interval * self.config.scheduler.backoff_multiplier),
                    )

    async def _has_new_user_followup_after_bot(
        self,
        memory: ConversationMemoryManager,
        chat_id: int,
        snapshot_before: int | None,
    ) -> bool:
        recent_bot_memory = await memory.get_recent_bot_memory(chat_id, limit=1)
        if not recent_bot_memory:
            return False
        last_bot = recent_bot_memory[0]
        if not last_bot.sent_message_id:
            return False
        recent_messages = await memory.get_recent_messages(chat_id, limit=25)
        for message in reversed(recent_messages):
            if message.message_id <= int(last_bot.sent_message_id):
                break
            if snapshot_before is not None and message.message_id <= snapshot_before:
                continue
            if self.bot_user_id is not None and message.sender_id == self.bot_user_id:
                continue
            if self._mentions_bot(message.text_cleaned or message.text_raw or ""):
                return True
            if message.reply_to_message_id == int(last_bot.sent_message_id):
                return True
        return False

    def _mentions_bot(self, text: str) -> bool:
        if not text:
            return False
        lowered = text.lower()
        if self.bot_username and f"@{self.bot_username}" in lowered:
            return True
        return "temp3289" in lowered

    def _direct_attention_fallback_text(self, text: str) -> str:
        key = _casual_key(text)
        tokens = set(key.split())
        if tokens & _UNSAFE_CASUAL_TERMS:
            return "can't help with that"
        if "?" in text:
            return "idk tbh"
            return "yeah?"

    def _message_is_direct(self, message, active_bot_thread: bool) -> bool:
        text = message.text or ""
        return bool(active_bot_thread or self._mentions_bot(text) or message.reply_to_message_id)

    async def _send_local_reply(
        self,
        chat_id: int,
        memory: ConversationMemoryManager,
        snapshot_message_id: int | None,
        new_message_count: int,
        message,
        reply: str,
        reasoning: str,
        confidence: float,
        brief,
        gate_score: float | None,
        gate_factors: dict,
        event_name: str,
    ) -> None:
        stored_decision = await memory.insert_ai_decision(
            chat_id=chat_id,
            prompt_version=self.config.ai.prompt_version,
            snapshot_message_id=snapshot_message_id,
            new_message_count=new_message_count,
            should_respond=True,
            confidence=confidence,
            response_text=reply,
            reply_to_message_id=message.message_id,
            reasoning=reasoning,
            gate_score=gate_score,
            gate_factors=gate_factors,
            request1_tokens_used=0,
            request2_tokens_used=0,
        )
        sent_message_id = await self.sender.send_message(chat_id, reply, message.message_id)
        await memory.update_ai_decision_sent_message(stored_decision.id, sent_message_id)
        await memory.insert_bot_memory(
            chat_id=chat_id,
            sent_message_id=sent_message_id,
            response_text=reply,
            reply_to_user_id=message.sender_id,
            reply_to_message_id=message.message_id,
            reasoning=reasoning,
            tone_calibration="local human motive",
            brief_snapshot=brief.as_dict(),
            stances={},
            prompt_version=self.config.ai.prompt_version,
            cycle_snapshot_message_id=snapshot_message_id,
        )
        await log.ainfo(event_name, chat_id=chat_id, message_id=message.message_id, sent_message_id=sent_message_id)

    async def _try_human_motive_reply(
        self,
        chat_id: int,
        is_private_dm: bool,
        active_bot_thread: bool,
        enriched,
        memory: ConversationMemoryManager,
        snapshot_before: int | None,
        snapshot_message_id: int | None,
        new_message_count: int,
        brief,
        gate_score: float | None,
        gate_factors: dict,
    ) -> bool:
        new_messages = [
            message
            for message in enriched
            if snapshot_before is None or message.message_id > snapshot_before
        ]
        for message in reversed(new_messages):
            if self.bot_user_id is not None and message.sender_id == self.bot_user_id:
                continue
            text = message.text or ""
            is_direct = self._message_is_direct(message, active_bot_thread)
            candidate = _human_motive_reply(text, is_direct=is_direct, is_private_dm=is_private_dm)
            if not candidate:
                continue
            if not is_private_dm and not is_direct and not _is_social_hook(text):
                continue

            responses_last_10min = await memory.count_bot_responses(chat_id, window_minutes=10)
            if is_private_dm:
                limit = 8
            elif is_direct:
                limit = 4
            else:
                limit = 2
            if responses_last_10min >= limit:
                await memory.insert_ai_decision(
                    chat_id=chat_id,
                    prompt_version=self.config.ai.prompt_version,
                    snapshot_message_id=snapshot_message_id,
                    new_message_count=new_message_count,
                    should_respond=False,
                    confidence=0.0,
                    response_text=None,
                    reply_to_message_id=None,
                    reasoning="local human-motive reply skipped by rate limit",
                    gate_score=gate_score,
                    gate_factors=gate_factors,
                    request1_tokens_used=0,
                    request2_tokens_used=0,
                )
                return True

            reply, reasoning, confidence = candidate
            await self._send_local_reply(
                chat_id=chat_id,
                memory=memory,
                snapshot_message_id=snapshot_message_id,
                new_message_count=new_message_count,
                message=message,
                reply=reply,
                reasoning=reasoning,
                confidence=confidence,
                brief=brief,
                gate_score=gate_score,
                gate_factors=gate_factors,
                event_name="local_human_motive_reply_sent",
            )
            return True
        return False

    async def _try_direct_attention_fallback(
        self,
        chat_id: int,
        active_bot_thread: bool,
        enriched,
        memory: ConversationMemoryManager,
        snapshot_before: int | None,
        snapshot_message_id: int | None,
        new_message_count: int,
        brief,
        gate_score: float | None,
        gate_factors: dict,
        request1_latency_ms: int,
        request1_tokens_used: int,
        reasoning: str,
    ) -> bool:
        if not active_bot_thread:
            return False
        new_messages = [
            message
            for message in enriched
            if snapshot_before is None or message.message_id > snapshot_before
        ]
        for message in reversed(new_messages):
            if self.bot_user_id is not None and message.sender_id == self.bot_user_id:
                continue
            text = message.text or ""
            if not (self._mentions_bot(text) or message.reply_to_message_id):
                continue
            reply = self._direct_attention_fallback_text(text)
            stored_decision = await memory.insert_ai_decision(
                chat_id=chat_id,
                prompt_version=self.config.ai.prompt_version,
                snapshot_message_id=snapshot_message_id,
                new_message_count=new_message_count,
                should_respond=True,
                confidence=0.65,
                response_text=reply,
                reply_to_message_id=message.message_id,
                reasoning=f"direct mention/follow-up fallback after Claude declined: {reasoning[:300]}",
                gate_score=gate_score,
                gate_factors=gate_factors,
                request1_latency_ms=request1_latency_ms,
                request1_tokens_used=request1_tokens_used,
                request2_tokens_used=0,
            )
            sent_message_id = await self.sender.send_message(chat_id, reply, message.message_id)
            await memory.update_ai_decision_sent_message(stored_decision.id, sent_message_id)
            await memory.insert_bot_memory(
                chat_id=chat_id,
                sent_message_id=sent_message_id,
                response_text=reply,
                reply_to_user_id=message.sender_id,
                reply_to_message_id=message.message_id,
                reasoning="direct mention/follow-up fallback",
                tone_calibration="short direct",
                brief_snapshot=brief.as_dict(),
                stances={},
                prompt_version=self.config.ai.prompt_version,
                cycle_snapshot_message_id=snapshot_message_id,
            )
            await log.ainfo(
                "direct_attention_fallback_sent",
                chat_id=chat_id,
                message_id=message.message_id,
                sent_message_id=sent_message_id,
            )
            return True
        return False

    async def _try_safe_casual_reply(
        self,
        chat_id: int,
        is_private_dm: bool,
        enriched,
        memory: ConversationMemoryManager,
        snapshot_before: int | None,
        snapshot_message_id: int | None,
        new_message_count: int,
        brief,
        gate_score: float | None,
        gate_factors: dict,
    ) -> bool:
        new_messages = [
            message
            for message in enriched
            if snapshot_before is None or message.message_id > snapshot_before
        ]
        for message in reversed(new_messages):
            if self.bot_user_id is not None and message.sender_id == self.bot_user_id:
                continue
            reply = _safe_casual_reply(message.text)
            if not reply:
                continue

            responses_last_10min = await memory.count_bot_responses(chat_id, window_minutes=10)
            if (is_private_dm and responses_last_10min >= 3) or (not is_private_dm and responses_last_10min >= 1):
                await memory.insert_ai_decision(
                    chat_id=chat_id,
                    prompt_version=self.config.ai.prompt_version,
                    snapshot_message_id=snapshot_message_id,
                    new_message_count=new_message_count,
                    should_respond=False,
                    confidence=0.0,
                    response_text=None,
                    reply_to_message_id=None,
                    reasoning="safe casual reply skipped by local rate limit",
                    gate_score=gate_score,
                    gate_factors=gate_factors,
                    request1_tokens_used=0,
                    request2_tokens_used=0,
                )
                return True

            stored_decision = await memory.insert_ai_decision(
                chat_id=chat_id,
                prompt_version=self.config.ai.prompt_version,
                snapshot_message_id=snapshot_message_id,
                new_message_count=new_message_count,
                should_respond=True,
                confidence=0.9,
                response_text=reply,
                reply_to_message_id=message.message_id,
                reasoning="local safe casual reply; skipped Claude for token efficiency",
                gate_score=gate_score,
                gate_factors=gate_factors,
                request1_tokens_used=0,
                request2_tokens_used=0,
            )
            sent_message_id = await self.sender.send_message(chat_id, reply, message.message_id)
            await memory.update_ai_decision_sent_message(stored_decision.id, sent_message_id)
            await memory.insert_bot_memory(
                chat_id=chat_id,
                sent_message_id=sent_message_id,
                response_text=reply,
                reply_to_user_id=message.sender_id,
                reply_to_message_id=message.message_id,
                reasoning="local safe casual reply",
                tone_calibration="tiny casual",
                brief_snapshot=brief.as_dict(),
                stances={},
                prompt_version=self.config.ai.prompt_version,
                cycle_snapshot_message_id=snapshot_message_id,
            )
            await log.ainfo(
                "local_safe_casual_reply_sent",
                chat_id=chat_id,
                message_id=message.message_id,
                sent_message_id=sent_message_id,
            )
            return True
        return False


async def main() -> None:
    config = load_engine_config()
    setup_logging()
    load_embedder(config.persona_engine.embedding_model)
    ai_client = FakeAiClient()
    sender = TelegramSender(config)
    await sender.connect()
    feedback_loop = FeedbackLoop(config, ai_client)
    me = await sender.client.get_me()
    bot_user_id = int(me.id)
    bot_username = getattr(me, "username", None)
    scheduler = ConversationScheduler(
        config,
        ai_client,
        sender,
        feedback_loop,
        bot_user_id,
        bot_username,
        LocalStyleRewriter(config),
    )

    loop = asyncio.get_running_loop()
    for sig_name in ("SIGINT", "SIGTERM"):
        loop.add_signal_handler(getattr(signal, sig_name), scheduler.shutdown)

    try:
        await run_bootstrap(config, ai_client, bot_user_id)
        await asyncio.gather(scheduler.run(), feedback_loop.run_observation_tasks())
    finally:
        await sender.close()
        await dispose_engine()


if __name__ == "__main__":
    asyncio.run(main())
