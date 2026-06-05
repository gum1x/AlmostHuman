from __future__ import annotations

import asyncio
import os
import signal
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from conversation_engine.ai_client import (
    FakeAiClient,
    GrokAiClient,
    ResponseDecision,
    parse_context_summary,
    parse_response_decision,
)
from conversation_engine.bootstrap import run_bootstrap
from conversation_engine.style_rewriter import LocalStyleRewriter
from conversation_engine.config import EngineConfig, load_engine_config
from conversation_engine.context_builder import (
    ContextBundle,
    build_context,
    compute_quantitative_signals,
    format_quantitative_signals,
    select_target_message,
)
from conversation_engine.engagement_gate import GateResult, compute_gate_score
from conversation_engine.enrichment import Brief, build_brief, current_context_text, enrich_messages
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
from conversation_engine.prompts import build_context_summary_prompt, build_response_decision_prompt
from conversation_engine.sender import TelegramSender
from conversation_engine.validators import validate
from core.logging import get_logger, setup_logging
from storage.database import async_session_factory, dispose_engine

log = get_logger(__name__)


@dataclass(frozen=True)
class _CyclePrep:
    chat_id: int
    is_private_dm: bool
    active_bot_thread: bool
    new_message_count: int
    snapshot_message_id: int | None
    gate: GateResult
    visible_numeric_controls: dict[str, Any]
    brief: Brief
    enriched: list
    context: ContextBundle
    raw_context: str
    high_level_enriched: list
    recent_enriched_for_summary: list
    recent_bot_mem: list
    bot_sent_ids: set[int]
    recent_bot_activity: str
    posture: str
    responses_last_hour: int


@dataclass(frozen=True)
class _CycleLlmOutcome:
    decision: ResponseDecision
    request1: Any
    request2: Any
    posture: str
    ok: bool
    reason: str | None


def _append_context_block(context, title: str, body: str):
    if not body.strip():
        return context
    return type(context)(
        context=f"{context.context}\n\n=== {title} ===\n{body.strip()}",
        candidate_user_ids=context.candidate_user_ids,
        relationship_profiles=context.relationship_profiles,
        avg_feedback_score=context.avg_feedback_score,
    )



class ConversationScheduler:
    def __init__(
        self,
        config: EngineConfig,
        ai_client,
        sender: TelegramSender,
        feedback_loop: FeedbackLoop,
        bot_user_id: int | None = None,
        bot_username: str | None = None,
    ):
        self.config = config
        self.ai_client = ai_client
        self.sender = sender
        self.feedback_loop = feedback_loop
        self.bot_user_id = bot_user_id
        self.bot_username = bot_username.lower() if bot_username else None
        self.style_rewriter = LocalStyleRewriter(config)
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

# Patch content - spliced into scheduler.py by maintenance script

    def _backoff_interval(self, previous_interval: int) -> int:
        return min(
            self.config.scheduler.max_interval_seconds,
            int(previous_interval * self.config.scheduler.backoff_multiplier),
        )

    async def _run_reflections_if_needed(self, chat_id: int, is_private_dm: bool) -> None:
        if is_private_dm:
            return
        async with async_session_factory() as session:
            async with session.begin():
                memory = ConversationMemoryManager(session)
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

    async def _run_cycle(self, chat_id: int, previous_interval: int) -> int:
        is_private_dm = chat_id > 0
        raw_context: str | None = None
        try:
            await self._run_reflections_if_needed(chat_id, is_private_dm)

            async with async_session_factory() as session:
                async with session.begin():
                    memory = ConversationMemoryManager(session)
                    if await memory.is_circuit_paused(chat_id):
                        return self.config.scheduler.max_interval_seconds
                    prep_result = await self._prepare_cycle(
                        memory,
                        chat_id,
                        is_private_dm,
                        previous_interval,
                    )
                    if isinstance(prep_result, int):
                        return prep_result
                    prep = prep_result
                    raw_context = prep.raw_context

            llm_out = await self._execute_llm(prep)

            async with async_session_factory() as session:
                async with session.begin():
                    memory = ConversationMemoryManager(session)
                    return await self._finalize_cycle(memory, prep, llm_out)
        except Exception as exc:
            async with async_session_factory() as session:
                async with session.begin():
                    memory = ConversationMemoryManager(session)
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
            return self._backoff_interval(previous_interval)

    async def _prepare_cycle(
        self,
        memory: ConversationMemoryManager,
        chat_id: int,
        is_private_dm: bool,
        previous_interval: int,
    ) -> int | _CyclePrep:
        new_message_threshold = (
            self.config.scheduler.dm_new_message_threshold
            if is_private_dm
            else self.config.scheduler.new_message_threshold
        )
        recent_message_limit = (
            self.config.scheduler.dm_recent_message_limit if is_private_dm else 50
        )

        latest_decision = await memory.get_latest_ai_decision(chat_id)
        snapshot_before = latest_decision.snapshot_message_id if latest_decision else None
        new_message_count = await memory.count_messages_after_snapshot(chat_id, snapshot_before)
        active_bot_thread = await self._has_new_user_followup_after_bot(memory, chat_id, snapshot_before)
        if new_message_count < new_message_threshold:
            if not active_bot_thread:
                return self._backoff_interval(previous_interval)
            new_message_count = max(1, new_message_count)

        await seed_persona_core(memory, self.config)

        messages = await memory.get_recent_messages(chat_id, limit=recent_message_limit)
        enriched = enrich_messages(messages, self.config.prompt)

        high_level_limit = self.config.scheduler.high_level_message_limit
        high_level_messages = await memory.get_recent_messages(chat_id, limit=high_level_limit)
        high_level_enriched = enrich_messages(high_level_messages, self.config.prompt)
        recent_context_limit = self.config.scheduler.recent_context_limit
        recent_for_summary = high_level_messages[-recent_context_limit:] if high_level_messages else messages
        recent_enriched_for_summary = (
            enrich_messages(recent_for_summary, self.config.prompt) if recent_for_summary else enriched
        )

        recent_bot_mem = await memory.get_recent_bot_memory(chat_id, limit=6)
        recent_activity_lines = []
        for bm in recent_bot_mem:
            if bm.response_text:
                recent_activity_lines.append(
                    f"I said (to user_{bm.reply_to_user_id or '?'}): {bm.response_text[:120]}"
                )
                if bm.reasoning:
                    recent_activity_lines.append(f"  (my reasoning at the time: {bm.reasoning[:100]})")
                if getattr(bm, "current_posture", None):
                    recent_activity_lines.append(f"  (my posture after: {bm.current_posture})")
        recent_bot_activity = "\n".join(recent_activity_lines) if recent_activity_lines else ""
        bot_sent_ids = {bm.sent_message_id for bm in recent_bot_mem if bm.sent_message_id is not None}

        brief = build_brief(enriched)
        if is_private_dm:
            gate = GateResult(gate_score=1.0, gate_factors={"mode": "private_dm"}, should_proceed=True)
        else:
            gate = await compute_gate_score(chat_id, enriched, brief, memory, self.config)

        outcome_score_24h = await memory.get_avg_feedback_score(chat_id, window_hours=24)
        visible_numeric_controls = {
            "tension_level": brief.tension_level,
            "outcome_score_24h": outcome_score_24h,
        }
        snapshot_message_id = await memory.latest_message_id(chat_id)
        now = datetime.now(timezone.utc)
        await memory.upsert_activity_pattern(
            chat_id,
            hour_of_day=now.hour,
            day_of_week=now.weekday(),
            velocity=new_message_count / max(1, self.config.engagement_gate.velocity_window_minutes),
            tension=brief.tension_level,
        )

        target_for_direct = select_target_message(enriched)
        is_direct_for_gate = active_bot_thread
        if target_for_direct:
            txt = target_for_direct.cleaned_text or target_for_direct.text or ""
            if self._mentions_bot(txt) or target_for_direct.reply_to_message_id in bot_sent_ids:
                is_direct_for_gate = True

        if not is_private_dm and not gate.should_proceed:
            if is_direct_for_gate:
                gate = GateResult(
                    gate_score=gate.gate_score,
                    gate_factors={**gate.gate_factors, "direct_mention_forced": True},
                    should_proceed=True,
                )
                await log.ainfo(
                    "direct_mention_forced_gate_proceed",
                    chat_id=chat_id,
                    message_id=getattr(target_for_direct, "message_id", None),
                )
            else:
                await memory.insert_ai_decision(
                    chat_id=chat_id,
                    prompt_version=self.config.ai.prompt_version,
                    snapshot_message_id=snapshot_message_id,
                    new_message_count=new_message_count,
                    should_respond=False,
                    confidence=round(max(0.05, gate.gate_score), 3),
                    response_text=None,
                    reply_to_message_id=None,
                    reasoning=(
                        f"gate blocked: score={gate.gate_score:.3f} < min="
                        f"{self.config.engagement_gate.min_gate_score_to_send:.2f}. factors="
                        + ", ".join(
                            f"{k}={round(v, 2) if isinstance(v, (int, float)) else v}"
                            for k, v in gate.gate_factors.items()
                        )
                    ),
                    gate_score=gate.gate_score,
                    gate_factors=gate.gate_factors,
                    request1_latency_ms=0,
                    request1_tokens_used=0,
                    request2_tokens_used=0,
                )
                await memory.record_cycle_success(chat_id)
                return self._backoff_interval(previous_interval)

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
            token_budget=self.config.ai.total_context_token_budget,
            recent_bot_activity=recent_bot_activity,
        )
        raw_context = context.context
        if active_bot_thread:
            raw_context = f"{context.context}\n\nactive_bot_thread: true"
            context = type(context)(
                context=raw_context,
                candidate_user_ids=context.candidate_user_ids,
                relationship_profiles=context.relationship_profiles,
                avg_feedback_score=context.avg_feedback_score,
            )

        posture = await self._infer_social_posture(
            chat_id, is_private_dm, memory, brief, active_bot_thread, recent_bot_mem=recent_bot_mem
        )
        responses_last_hour = await memory.count_bot_responses(chat_id, window_minutes=60)

        return _CyclePrep(
            chat_id=chat_id,
            is_private_dm=is_private_dm,
            active_bot_thread=active_bot_thread,
            new_message_count=new_message_count,
            snapshot_message_id=snapshot_message_id,
            gate=gate,
            visible_numeric_controls=visible_numeric_controls,
            brief=brief,
            enriched=enriched,
            context=context,
            raw_context=raw_context,
            high_level_enriched=high_level_enriched,
            recent_enriched_for_summary=recent_enriched_for_summary,
            recent_bot_mem=recent_bot_mem,
            bot_sent_ids=bot_sent_ids,
            recent_bot_activity=recent_bot_activity,
            posture=posture,
            responses_last_hour=responses_last_hour,
        )

    async def _execute_llm(self, prep: _CyclePrep) -> _CycleLlmOutcome:
        context = prep.context
        summary_prompt, summary_system = build_context_summary_prompt(
            context,
            self.config,
            high_level_enriched=prep.high_level_enriched,
            recent_enriched=prep.recent_enriched_for_summary,
        )
        request1 = await self.ai_client.call_perception_model(summary_prompt, summary_system)
        context_summary = parse_context_summary(request1.text)
        summary_body = context_summary.compressed_relevant_context or context_summary.summary
        if summary_body:
            header = (
                "RELEVANT CONVERSATION CONTEXT"
                if context_summary.compressed_relevant_context
                else "PERCEPTION SUMMARY"
            )
            context = _append_context_block(context, header, summary_body)

        time_since_last_bot: float | None = None
        if prep.recent_bot_mem:
            latest_bot_ts = getattr(prep.recent_bot_mem[0], "created_at", None)
            if latest_bot_ts:
                time_since_last_bot = (datetime.now(timezone.utc) - latest_bot_ts).total_seconds() / 60.0

        quant_signals = compute_quantitative_signals(
            enriched_messages=prep.enriched,
            bot_user_id=self.bot_user_id,
            bot_sent_message_ids=prep.bot_sent_ids,
            time_since_last_bot_msg_min=time_since_last_bot,
            responses_last_hour=prep.responses_last_hour,
            bot_username=self.bot_username,
        )
        if prep.active_bot_thread:
            quant_signals["direct_mention"] = True
        signals_block = format_quantitative_signals(quant_signals)
        enriched_for_decision = (
            f"{context.context}\n\n"
            f"=== PRE-COMPUTED SIGNALS ===\n{signals_block}\n"
            f"current_posture={prep.posture}"
        )
        decision_context = type(context)(
            context=enriched_for_decision,
            candidate_user_ids=context.candidate_user_ids,
            relationship_profiles=context.relationship_profiles,
            avg_feedback_score=context.avg_feedback_score,
        )
        decision_prompt, decision_system = build_response_decision_prompt(
            decision_context,
            "",
            self.config,
        )
        request2 = await self.ai_client.call_decision_model(decision_prompt, decision_system)
        decision = parse_response_decision(request2.text)

        if decision.should_respond and self.style_rewriter.enabled:
            plan_signal = (decision.plan or decision.reasoning or "").strip()
            if plan_signal:
                phrased = await self.style_rewriter.phrase(
                    context=prep.raw_context or "",
                    plan=plan_signal,
                    target_message="",
                    tone=decision.tone_calibration or "",
                )
                if phrased and phrased.strip():
                    decision.response_text = phrased

        recent_bot_texts = [
            bm.response_text for bm in prep.recent_bot_mem if getattr(bm, "response_text", None)
        ]
        ok, reason = validate(decision, self.config, recent_bot_texts=recent_bot_texts)
        if ok:
            ok, reason = self._passes_social_safety(
                is_private_dm=prep.is_private_dm,
                active_bot_thread=prep.active_bot_thread,
                enriched=prep.enriched,
                brief=prep.brief,
                decision=decision,
                bot_sent_ids=prep.bot_sent_ids,
            )

        return _CycleLlmOutcome(
            decision=decision,
            request1=request1,
            request2=request2,
            posture=prep.posture,
            ok=ok,
            reason=reason,
        )

    async def _finalize_cycle(
        self,
        memory: ConversationMemoryManager,
        prep: _CyclePrep,
        llm_out: _CycleLlmOutcome,
    ) -> int:
        decision = llm_out.decision
        stored_decision = await memory.insert_ai_decision(
            chat_id=prep.chat_id,
            prompt_version=self.config.ai.prompt_version,
            snapshot_message_id=prep.snapshot_message_id,
            new_message_count=prep.new_message_count,
            should_respond=llm_out.ok,
            confidence=decision.confidence,
            response_text=decision.response_text,
            reply_to_message_id=decision.reply_to_message_id,
            reasoning=(
                (decision.reasoning or "")
                + (
                    f" | posture update: {decision.updated_engagement_posture}"
                    if decision.updated_engagement_posture
                    else ""
                )
            )
            if llm_out.ok
            else llm_out.reason,
            gate_score=prep.gate.gate_score,
            gate_factors=prep.visible_numeric_controls,
            request1_latency_ms=llm_out.request1.latency_ms,
            request1_tokens_used=llm_out.request1.tokens_used,
            request2_tokens_used=llm_out.request2.tokens_used,
        )
        if not llm_out.ok:
            await memory.record_cycle_success(prep.chat_id)
            return self.config.scheduler.initial_interval_seconds

        sent_message_id = await self.sender.send_message(
            prep.chat_id,
            decision.response_text or "",
            decision.reply_to_message_id,
        )
        await memory.update_ai_decision_sent_message(stored_decision.id, sent_message_id)
        persisted_posture = decision.updated_engagement_posture or llm_out.posture
        bot_memory = await memory.insert_bot_memory(
            chat_id=prep.chat_id,
            sent_message_id=sent_message_id,
            response_text=decision.response_text or "",
            reply_to_user_id=decision.reply_to_user_id,
            reply_to_message_id=decision.reply_to_message_id,
            reasoning=decision.reasoning,
            tone_calibration=decision.tone_calibration,
            brief_snapshot=prep.brief.as_dict(),
            stances=decision.stances,
            prompt_version=self.config.ai.prompt_version,
            cycle_snapshot_message_id=prep.snapshot_message_id,
            current_posture=persisted_posture,
        )
        await write_interaction_memory(
            memory,
            prep.chat_id,
            decision.reply_to_user_id,
            decision.topic,
            decision.response_text or "",
        )
        for topic, stance in decision.stances.items():
            await memory.upsert_stance(
                prep.chat_id, topic=topic, stance=str(stance), user_id=decision.reply_to_user_id
            )
            await write_stance_memory(memory, prep.chat_id, decision.reply_to_user_id, topic, str(stance))
        await self.feedback_loop.schedule_observation(bot_memory.id, sent_message_id, prep.chat_id)
        await memory.record_cycle_success(prep.chat_id)
        return self.config.scheduler.initial_interval_seconds

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
        return bool(self.bot_username and f"@{self.bot_username}" in text.lower())

    async def _infer_social_posture(
        self,
        chat_id: int,
        is_private_dm: bool,
        memory: ConversationMemoryManager,
        brief: Brief,
        active_bot_thread: bool,
        recent_bot_mem: list | None = None,
    ) -> str:
        if is_private_dm:
            return "private_dm: responsive but still brief"
        recent_outcome = await memory.get_avg_feedback_score(chat_id, window_hours=24)
        responses_last_10min = await memory.count_bot_responses(chat_id, window_minutes=10)
        if recent_bot_mem:
            latest_bm = recent_bot_mem[0]
            if latest_bm and getattr(latest_bm, "response_text", None):
                lp = getattr(latest_bm, "current_posture", "") or ""
                if "hyperactive" in lp.lower() or "engaged" in lp.lower():
                    return "old timer: recently spoke, might speak again if another moment or opinion strikes"
        if brief.tension_level >= self.config.engagement_gate.anti_flame_tension_threshold:
            return "burned/quiet: tension is high"
        if recent_outcome <= -0.25:
            return "burned/quiet: recent replies did not land"
        if active_bot_thread:
            return "in_thread: direct follow-up exists"
        if responses_last_10min >= 2:
            return "lurking: already spoke recently"
        if brief.tension_level <= 0.25:
            return "lightly_vibing: available for high-signal or funny moments"
        return "watching: selective and low-ego"

    def _passes_social_safety(
        self,
        *,
        is_private_dm: bool,
        active_bot_thread: bool,
        enriched,
        brief: Brief,
        decision: ResponseDecision,
        bot_sent_ids: set[int],
    ) -> tuple[bool, str | None]:
        if is_private_dm:
            return True, None

        by_id = {message.message_id: message for message in enriched}
        target_id = decision.reply_to_message_id or decision.target_message_id
        target = by_id.get(int(target_id)) if target_id is not None else None
        if not target:
            target = enriched[-1] if enriched else None
        if target and decision.reply_to_message_id is None:
            decision.reply_to_message_id = target.message_id

        if brief.tension_level >= self.config.engagement_gate.anti_flame_tension_threshold:
            if active_bot_thread:
                return True, None
            if target:
                txt = target.cleaned_text or target.text or ""
                if self._mentions_bot(txt) or target.reply_to_message_id in bot_sent_ids:
                    return True, None
            return False, "anti_flame_protection"

        return True, None


async def main() -> None:
    config = load_engine_config()
    setup_logging()
    load_embedder(config.persona_engine.embedding_model)
    key = (config.xai_api_key or "").strip().lower()
    base = (config.xai_base_url or "").strip()

    is_dummy_key = key in ("", "sk-local", "sk-dummy", "local", "ollama", "vllm", "none")
    is_xai_default = "api.x.ai" in base

    # Only use real client if we have a non-dummy key OR an explicit non-xAI base URL (local server)
    use_real_client = bool(config.xai_api_key) and not (is_dummy_key and is_xai_default)
    if not use_real_client and os.getenv("ALLOW_FAKE_AI", "").lower() != "true":
        raise RuntimeError(
            "No AI client configured (missing XAI_API_KEY or local XAI_BASE_URL). "
            "Set ALLOW_FAKE_AI=true for offline development."
        )
    ai_client = GrokAiClient(config) if use_real_client else FakeAiClient()
    sender = TelegramSender(config)
    await sender.connect()
    feedback_loop = FeedbackLoop(config, ai_client, sender)
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
    )

    loop = asyncio.get_running_loop()
    for sig_name in ("SIGINT", "SIGTERM"):
        loop.add_signal_handler(getattr(signal, sig_name), scheduler.shutdown)

    try:
        await run_bootstrap(config, ai_client, bot_user_id)
        await asyncio.gather(scheduler.run(), feedback_loop.run_observation_tasks())
    finally:
        close = getattr(ai_client, "close", None)
        if close:
            await close()
        await sender.close()
        await dispose_engine()


if __name__ == "__main__":
    asyncio.run(main())
