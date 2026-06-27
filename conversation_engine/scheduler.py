from __future__ import annotations

import asyncio
import json
import os
import random
import signal
import time
from datetime import datetime, timedelta, timezone

from conversation_engine import humanizer, suspicion_monitor, volume_governor
from conversation_engine.ai_client import (
    AiCallResult,
    FakeAiClient,
    GrokAiClient,
    ResponseDecision,
    parse_context_summary,
    parse_response_decision,
)
from conversation_engine.bootstrap import run_bootstrap
from conversation_engine.config import EngineConfig, load_engine_config
from conversation_engine.context_builder import (
    build_context,
    build_request2_constraints,
    compute_quantitative_signals,
    format_quantitative_signals,
    select_target_message,
)
from conversation_engine.engagement_gate import GateResult, compute_gate_score
from conversation_engine.enrichment import (
    Brief,
    build_brief,
    current_context_text,
    enrich_messages_async,
)
from conversation_engine.feedback_loop import FeedbackLoop, run_meta_reflection
from conversation_engine.memory_manager import ConversationMemoryManager
from conversation_engine.output_planner import Action, OutputPlan, plan_output
from conversation_engine.persona_engine import (
    get_relevant_persona_vectors,
    load_embedder,
    run_self_reflection,
    seed_persona_core,
    should_run_self_reflection,
    write_interaction_memory,
    write_stance_memory,
)
from conversation_engine.prompts import build_context_summary_prompt, build_response_decision_prompt
from conversation_engine.scheduler_support import (
    _append_context_block,
    _CycleLlmOutcome,
    _CyclePrep,
    _decline_reasoning,
)
from conversation_engine.sender import TelegramSender
from conversation_engine.style_rewriter import LocalStyleRewriter
from conversation_engine.timing_classifier import (
    TimingClassifier,
    compute_regulars,
    history_feature_inputs,
    timing_should_skip,
)
from conversation_engine.validators import (
    apply_donor_casing,
    strip_terminal_period,
    validate,
    violates_ai_tell,
)
from core.logging import get_logger, setup_logging
from storage.database import async_session_factory, dispose_engine

log = get_logger(__name__)

# Train==serve for the timing classifier: training's "regulars" are the top-K most
# active senders over the whole export (scripts/build_timing_dataset.py). The serve-time
# mirror counts senders over this much recent history, cached per chat.
TIMING_REGULARS_HISTORY_LIMIT = 2000
TIMING_REGULARS_CACHE_TTL_SECONDS = 1800

# Liveness: touched after every completed cycle; the compose healthcheck alarms when
# it goes stale (engine hung/dead but container still "running").
HEARTBEAT_FILE = os.getenv("ENGINE_HEARTBEAT_FILE", "/tmp/engine_heartbeat")
DEADMAN_PING_INTERVAL_SECONDS = 60


def _touch_heartbeat() -> None:
    try:
        with open(HEARTBEAT_FILE, "w") as f:
            f.write(str(time.time()))
    except OSError:
        pass


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
        self.timing_classifier = None
        if getattr(config, "timing_classifier_enabled", False) or getattr(
            config, "timing_classifier_shadow", False
        ):
            tc = TimingClassifier(model_path=config.timing_classifier_model_path)
            # Optional threshold override from config (0 = keep the model's own).
            if config.timing_classifier_threshold and config.timing_classifier_threshold > 0:
                tc.threshold = config.timing_classifier_threshold
            self.timing_classifier = tc
        # In local-only mode the timing classifier is the ONLY thing deciding when to
        # speak. Without it, the bot would reply to everything that clears the basic
        # engagement gate — warn loudly so this misconfig is visible.
        if not getattr(config, "cloud_brain_enabled", True) and self.timing_classifier is None:
            log.warning(
                "local_only_without_timing_classifier",
                detail="cloud_brain_enabled=false but timing classifier is off; bot will "
                "respond to everything that passes the engagement gate. Enable "
                "TIMING_CLASSIFIER_ENABLED to control response rate.",
            )
        self._timing_regulars_cache: dict[int, tuple[float, set[int]]] = {}
        # Behavioral-layer rng: a fixed seed (config.behavioral_rng_seed) makes the
        # humanizer/governor/output-planner draws deterministic; None => per-process.
        self._behavioral_rng = random.Random(config.behavioral_rng_seed)
        self._shutdown = asyncio.Event()
        self._deadman_url = (os.getenv("DEADMAN_PING_URL") or "").strip()
        self._last_deadman_ping = 0.0
        # Per-chat "go dark" cooldowns: when the room accuses a bot we suppress
        # sends until this wall-clock time, independent of whether the accusing
        # message is still in the recent window. In-memory only — lost on restart
        # (acceptable: a restart just ends the dark period early).
        self._dark_until: dict[int, datetime] = {}
        # Strong refs to fire-and-forget background tasks so they aren't GC'd
        # mid-flight (asyncio only holds weak refs to running tasks).
        self._bg_tasks: set[asyncio.Task] = set()

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
                await self._sleep_interruptible(self.config.scheduler.dm_discovery_interval_seconds)
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
            _touch_heartbeat()
            self._maybe_ping_deadman()
            await self._sleep_interruptible(interval)

    async def _sleep_interruptible(self, seconds: float) -> None:
        """Sleep that wakes on shutdown so docker stop doesn't hit the SIGKILL grace
        timeout mid-interval (the recurring Exited(137))."""
        try:
            await asyncio.wait_for(self._shutdown.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass

    async def _sleep_keeping_heartbeat(self, seconds: float) -> bool:
        """Wait ``seconds`` total in short, shutdown-interruptible slices, refreshing
        the liveness heartbeat each slice. Used for the humanized send-delay, which can
        be many minutes: a single sleep would let the heartbeat go stale (>900s) and let
        autoheal SIGKILL the container mid-send. Returns True if shutdown was signaled
        (caller should abort), else False once the full delay has elapsed."""
        remaining = seconds
        while remaining > 0:
            slice_s = min(60.0, remaining)
            await self._sleep_interruptible(slice_s)
            if self._shutdown.is_set():
                return True
            _touch_heartbeat()
            remaining -= slice_s
        return False

    def _maybe_ping_deadman(self) -> None:
        if not self._deadman_url:
            return
        now = time.monotonic()
        if now - self._last_deadman_ping < DEADMAN_PING_INTERVAL_SECONDS:
            return
        self._last_deadman_ping = now
        task = asyncio.create_task(self._ping_deadman())
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    async def _ping_deadman(self) -> None:
        try:
            import httpx

            async with httpx.AsyncClient(timeout=10) as client:
                await client.get(self._deadman_url)
        except Exception as exc:  # best-effort liveness ping; never block the loop
            await log.awarning("deadman_ping_failed", error=str(exc))

    def _backoff_interval(self, previous_interval: int) -> int:
        return min(
            self.config.scheduler.max_interval_seconds,
            int(previous_interval * self.config.scheduler.backoff_multiplier),
        )

    async def _get_timing_regulars(
        self, memory: ConversationMemoryManager, chat_id: int
    ) -> set[int]:
        """Top-K most active senders by message count (train==serve with
        scripts/build_timing_dataset.py 'regulars'), cached per chat.

        If the loaded model embeds its frozen training regulars (v2), use those
        directly — recomputing over a recent window would change what the trained
        reply_to_regular/sender_is_regular features mean."""
        frozen = getattr(self.timing_classifier, "regulars", None)
        if frozen is not None:
            return frozen
        now = time.monotonic()
        cached = self._timing_regulars_cache.get(chat_id)
        if cached and now - cached[0] < TIMING_REGULARS_CACHE_TTL_SECONDS:
            return cached[1]
        history = await memory.get_recent_messages(chat_id, limit=TIMING_REGULARS_HISTORY_LIMIT)
        regulars = compute_regulars(
            m.sender_id for m in history if (m.text_cleaned or m.text_raw or "").strip()
        )
        self._timing_regulars_cache[chat_id] = (now, regulars)
        return regulars

    async def _run_reflections_if_needed(self, chat_id: int, is_private_dm: bool) -> None:
        if is_private_dm:
            return
        # Self/meta reflection are OpenRouter calls — skip them entirely in local-only mode.
        if not getattr(self.config, "cloud_brain_enabled", True):
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

            if self.config.behavioral_layer_enabled:
                # Behavioral path manages its own (short) transactions so that
                # humanizing delays are awaited OUTSIDE any open transaction.
                return await self._finalize_cycle_behavioral(chat_id, prep, llm_out)

            # Default path also manages its own (short) transactions so that the
            # physical send happens OUTSIDE any open transaction and a post-send
            # recording failure can never roll back a message that was delivered.
            return await self._finalize_cycle(prep, llm_out)
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
        active_bot_thread = await self._has_new_user_followup_after_bot(
            memory, chat_id, snapshot_before
        )
        if new_message_count < new_message_threshold:
            if not active_bot_thread:
                return self._backoff_interval(previous_interval)
            new_message_count = max(1, new_message_count)

        await seed_persona_core(memory, self.config)

        messages = await memory.get_recent_messages(chat_id, limit=recent_message_limit)
        enriched = await enrich_messages_async(messages, self.config.prompt)

        high_level_limit = self.config.scheduler.high_level_message_limit
        high_level_messages = await memory.get_recent_messages(chat_id, limit=high_level_limit)
        high_level_enriched = await enrich_messages_async(high_level_messages, self.config.prompt)
        recent_context_limit = self.config.scheduler.recent_context_limit
        recent_for_summary = (
            high_level_messages[-recent_context_limit:] if high_level_messages else messages
        )
        recent_enriched_for_summary = (
            await enrich_messages_async(recent_for_summary, self.config.prompt)
            if recent_for_summary
            else enriched
        )

        recent_bot_mem = await memory.get_recent_bot_memory(chat_id, limit=6)
        recent_activity_lines = []
        for bm in recent_bot_mem:
            if bm.response_text:
                recent_activity_lines.append(
                    f"I said (to user_{bm.reply_to_user_id or '?'}): {bm.response_text[:120]}"
                )
                if bm.reasoning:
                    recent_activity_lines.append(
                        f"  (my reasoning at the time: {bm.reasoning[:100]})"
                    )
                if getattr(bm, "current_posture", None):
                    recent_activity_lines.append(f"  (my posture after: {bm.current_posture})")
        recent_bot_activity = "\n".join(recent_activity_lines) if recent_activity_lines else ""
        bot_sent_ids = {
            bm.sent_message_id for bm in recent_bot_mem if bm.sent_message_id is not None
        }

        brief = build_brief(enriched)
        if is_private_dm:
            gate = GateResult(
                gate_score=1.0, gate_factors={"mode": "private_dm"}, should_proceed=True
            )
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
            velocity=new_message_count
            / max(1, self.config.engagement_gate.velocity_window_minutes),
            tension=brief.tension_level,
        )

        target_for_direct = select_target_message(enriched)
        is_direct_for_gate = active_bot_thread
        if target_for_direct:
            txt = target_for_direct.cleaned_text or target_for_direct.text or ""
            if self._mentions_bot(txt) or target_for_direct.reply_to_message_id in bot_sent_ids:
                is_direct_for_gate = True

        # Timing classifier pre-gate (advisor's Part 2): cheaply decide whether this
        # incoming message realistically earns a reply BEFORE spending paid LLM calls.
        # Direct mentions / replies-to-bot / DMs always bypass it (we always engage those).
        if (
            self.timing_classifier is not None
            and not is_private_dm
            and not is_direct_for_gate
            and target_for_direct is not None
        ):
            t_txt = target_for_direct.cleaned_text or target_for_direct.text or ""
            # Train==serve: history-derived features computed exactly as
            # scripts/build_timing_dataset.py does (regulars = top-K active senders,
            # NOT "replied to the bot"; idx gap = message-index gap since sender last spoke).
            regulars = await self._get_timing_regulars(memory, chat_id)
            hist_feats = history_feature_inputs(
                target_message_id=target_for_direct.message_id,
                history=high_level_enriched,
                regulars=regulars,
            )
            ts = self.timing_classifier.score(
                text=t_txt,
                is_reply=hist_feats["is_reply"],
                reply_to_regular=hist_feats["reply_to_regular"],
                sender_is_regular=hist_feats["sender_is_regular"],
                idx_gap_since_sender=hist_feats["idx_gap_since_sender"],
            )
            gate = GateResult(
                gate_score=gate.gate_score,
                gate_factors={
                    **gate.gate_factors,
                    "timing_p": round(ts.score, 3),
                    "timing_would_pass": ts.passes,
                    "timing_is_direct": False,
                },
                should_proceed=gate.should_proceed,
            )
            enforcing = (
                self.config.timing_classifier_enabled and not self.config.timing_classifier_shadow
            )
            if timing_should_skip(passes=ts.passes, enforcing=enforcing):
                await memory.insert_ai_decision(
                    chat_id=chat_id,
                    prompt_version=self.config.ai.prompt_version,
                    snapshot_message_id=snapshot_message_id,
                    new_message_count=new_message_count,
                    should_respond=False,
                    confidence=round(ts.score, 3),
                    response_text=None,
                    reply_to_message_id=None,
                    reasoning=(
                        f"timing_classifier skip: p={ts.score:.3f} < thr="
                        f"{self.timing_classifier.threshold:.2f} "
                        f"(botlike={ts.is_botlike})"
                    ),
                    gate_score=gate.gate_score,
                    gate_factors=gate.gate_factors,
                    request1_latency_ms=0,
                    request1_tokens_used=0,
                    request2_tokens_used=0,
                )
                await log.ainfo(
                    "timing_classifier_skip",
                    chat_id=chat_id,
                    p=round(ts.score, 3),
                    threshold=self.timing_classifier.threshold,
                    botlike=ts.is_botlike,
                    message_id=getattr(target_for_direct, "message_id", None),
                )
                await memory.record_cycle_success(chat_id)
                return self._backoff_interval(previous_interval)
            elif not ts.passes:
                await log.ainfo(
                    "timing_classifier_shadow",
                    chat_id=chat_id,
                    p=round(ts.score, 3),
                    threshold=self.timing_classifier.threshold,
                    would_pass=ts.passes,
                    message_id=getattr(target_for_direct, "message_id", None),
                )

        if not is_private_dm and not gate.should_proceed:
            override_suppressed = (
                self._direct_override_suppression(gate, target_for_direct, recent_bot_mem, enriched)
                if is_direct_for_gate
                else None
            )
            if is_direct_for_gate and override_suppressed is None:
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
                if override_suppressed is not None:
                    gate = GateResult(
                        gate_score=gate.gate_score,
                        gate_factors={
                            **gate.gate_factors,
                            "direct_override_suppressed": override_suppressed,
                        },
                        should_proceed=False,
                    )
                    await log.ainfo(
                        "direct_mention_override_suppressed",
                        chat_id=chat_id,
                        reason=override_suppressed,
                        message_id=getattr(target_for_direct, "message_id", None),
                    )
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

        # Behavioral volume-governor inputs: only queried when the layer is on so the
        # flag-OFF path issues no extra queries.
        group_msgs_last_hour = 0
        bot_sends_last_10min = 0
        if self.config.behavioral_layer_enabled and not is_private_dm:
            group_msgs_last_hour = await memory.count_messages_in_window(chat_id, minutes=60)
            bot_sends_last_10min = await memory.count_bot_responses(chat_id, window_minutes=10)

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
            group_msgs_last_hour=group_msgs_last_hour,
            bot_sends_last_10min=bot_sends_last_10min,
            current_persona=current_persona,
            latest_reflection=latest_reflection,
        )

    async def _execute_llm(self, prep: _CyclePrep) -> _CycleLlmOutcome:
        # Local-only mode: the OpenRouter "brain" (perception + decision) is turned off.
        # The timing classifier already decided this message is worth a reply (it passed
        # the pre-gate), so we skip both paid calls and let the voice model write the words.
        if not getattr(self.config, "cloud_brain_enabled", True):
            return await self._execute_local_only(prep)

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
                time_since_last_bot = (
                    datetime.now(timezone.utc) - latest_bot_ts
                ).total_seconds() / 60.0

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
        constraints = build_request2_constraints(
            current_persona=prep.current_persona,
            latest_reflection=prep.latest_reflection,
            relationship_profiles=decision_context.relationship_profiles,
            avg_feedback_score=decision_context.avg_feedback_score,
        )
        decision_prompt, decision_system = build_response_decision_prompt(
            decision_context,
            constraints,
            self.config,
        )
        request2 = await self.ai_client.call_decision_model(decision_prompt, decision_system)
        try:
            decision = parse_response_decision(request2.text)
        except (ValueError, json.JSONDecodeError) as exc:
            # Malformed/empty decision output => don't respond this cycle rather
            # than crash. (Some providers occasionally return non-JSON.)
            await log.awarning(
                "decision_parse_failed",
                chat_id=prep.chat_id,
                error=str(exc),
                raw_preview=(request2.text or "")[:200],
            )
            decision = ResponseDecision(should_respond=False, reasoning="decision_parse_failed")

        if decision.should_respond and self.style_rewriter.enabled:
            if getattr(self.config, "voice_mode", "standalone") == "standalone":
                # New: single-voice model trained on raw (context -> reply) pairs writes
                # the words itself. The smart model already decided WHETHER to speak.
                # Feed clean "uXXX: text" lines (train==serve), NOT the rich smart-model
                # context which has persona/signal blocks the voice model never saw.
                voice_ctx = self.style_rewriter.build_voice_context(prep.enriched)
                voiced = await self.style_rewriter.generate_voice(context=voice_ctx)
                if voiced and voiced.strip():
                    decision.response_text = voiced
            else:
                # Legacy: smart model emits a plan; local model phrases it.
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
                recent_bot_mem=prep.recent_bot_mem,
            )

        return _CycleLlmOutcome(
            decision=decision,
            request1=request1,
            request2=request2,
            posture=prep.posture,
            ok=ok,
            reason=reason,
        )

    async def _execute_local_only(self, prep: _CyclePrep) -> _CycleLlmOutcome:
        """Local-only path used when the cloud brain (OpenRouter) is disabled.

        No paid perception/decision calls. The timing classifier already approved this
        cycle, so we mark should_respond=True and let the voice model write the reply
        straight from raw context. The same validators + social-safety checks still run.
        """
        zero = AiCallResult(text="", latency_ms=0, tokens_used=0)
        target = select_target_message(prep.enriched)
        reply_to_id = getattr(target, "message_id", None)
        reply_to_user = getattr(target, "sender_id", None)

        # Confidence is set above ai.min_confidence_to_send: the timing classifier
        # already decided this message is worth a reply, so the validator's confidence
        # floor (meant for the LLM decision path) shouldn't reject local-only replies.
        local_conf = max(0.7, float(self.config.ai.min_confidence_to_send) + 0.05)
        decision = ResponseDecision(
            should_respond=True,
            confidence=local_conf,
            reply_to_message_id=reply_to_id,
            reply_to_user_id=reply_to_user,
            target_message_id=reply_to_id,
            reasoning="local_only_mode: timing classifier approved, voice model writing reply",
        )

        if self.style_rewriter.enabled:
            # Clean "uXXX: text" lines (train==serve), not the rich smart-model context.
            voice_ctx = self.style_rewriter.build_voice_context(prep.enriched)
            voiced = await self.style_rewriter.generate_voice(context=voice_ctx)
            if voiced and voiced.strip():
                decision.response_text = voiced
            else:
                # Voice model produced nothing — don't send an empty message.
                decision.should_respond = False
                decision.reasoning = "local_only_mode: voice model returned empty"
        else:
            decision.should_respond = False
            decision.reasoning = (
                "local_only_mode: style_rewriter disabled (no local voice available)"
            )

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
                recent_bot_mem=prep.recent_bot_mem,
            )

        await log.ainfo(
            "local_only_cycle",
            chat_id=prep.chat_id,
            ok=ok,
            reason=reason if not ok else "",
            has_text=bool(decision.response_text),
        )

        return _CycleLlmOutcome(
            decision=decision,
            request1=zero,
            request2=zero,
            posture=prep.posture,
            ok=ok,
            reason=reason,
        )

    async def _finalize_cycle(
        self,
        prep: _CyclePrep,
        llm_out: _CycleLlmOutcome,
    ) -> int:
        """Flag-OFF finalize. Manages its own short transactions so the physical
        send happens OUTSIDE any open transaction (mirroring the behavioral path):
        a post-send recording failure must not roll back a message that was already
        delivered nor trip the circuit breaker for a successful send."""
        decision = llm_out.decision

        # Decline: validate()/social-safety already said no. Record the AiDecision
        # row and close the cycle out as a success — no send.
        if not llm_out.ok:
            async with async_session_factory() as session:
                async with session.begin():
                    memory = ConversationMemoryManager(session)
                    await memory.insert_ai_decision(
                        chat_id=prep.chat_id,
                        prompt_version=self.config.ai.prompt_version,
                        snapshot_message_id=prep.snapshot_message_id,
                        new_message_count=prep.new_message_count,
                        should_respond=False,
                        confidence=decision.confidence,
                        response_text=decision.response_text,
                        reply_to_message_id=decision.reply_to_message_id,
                        reasoning=_decline_reasoning(llm_out.reason, decision),
                        gate_score=prep.gate.gate_score,
                        gate_factors={**prep.gate.gate_factors, **prep.visible_numeric_controls},
                        request1_latency_ms=llm_out.request1.latency_ms,
                        request1_tokens_used=llm_out.request1.tokens_used,
                        request2_tokens_used=llm_out.request2.tokens_used,
                    )
                    await memory.record_cycle_success(prep.chat_id)
            return self.config.scheduler.initial_interval_seconds

        # Send the message FIRST, outside any transaction.
        sent_message_id = await self.sender.send_message(
            prep.chat_id,
            decision.response_text or "",
            decision.reply_to_message_id,
        )
        persisted_posture = decision.updated_engagement_posture or llm_out.posture
        # The message is already out. A failure in the post-send recording below must
        # NOT count as a cycle failure (that would trip the circuit breaker for a send
        # that actually succeeded) — log it and move on instead of re-raising.
        try:
            async with async_session_factory() as session:
                async with session.begin():
                    memory = ConversationMemoryManager(session)
                    stored_decision = await memory.insert_ai_decision(
                        chat_id=prep.chat_id,
                        prompt_version=self.config.ai.prompt_version,
                        snapshot_message_id=prep.snapshot_message_id,
                        new_message_count=prep.new_message_count,
                        should_respond=True,
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
                        ),
                        gate_score=prep.gate.gate_score,
                        gate_factors={**prep.gate.gate_factors, **prep.visible_numeric_controls},
                        request1_latency_ms=llm_out.request1.latency_ms,
                        request1_tokens_used=llm_out.request1.tokens_used,
                        request2_tokens_used=llm_out.request2.tokens_used,
                    )
                    await memory.update_ai_decision_sent_message(
                        stored_decision.id, sent_message_id
                    )
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
                            prep.chat_id,
                            topic=topic,
                            stance=str(stance),
                            user_id=decision.reply_to_user_id,
                        )
                        await write_stance_memory(
                            memory, prep.chat_id, decision.reply_to_user_id, topic, str(stance)
                        )
                    await memory.record_cycle_success(prep.chat_id)
            # Schedule the delayed feedback observation AFTER the recording txn commits,
            # so it is never awaited inside an open transaction.
            await self.feedback_loop.schedule_observation(
                bot_memory.id, sent_message_id, prep.chat_id
            )
        except Exception:
            await log.aexception(
                "post_send_record_failed",
                chat_id=prep.chat_id,
                sent_message_id=sent_message_id,
            )
        return self.config.scheduler.initial_interval_seconds

    # ------------------------------------------------------------------
    # Behavioral-layer finalize (flag-ON path)
    # ------------------------------------------------------------------

    async def _record_behavioral_decline(
        self, prep: _CyclePrep, llm_out: _CycleLlmOutcome, reason: str
    ) -> None:
        """Persist a non-sending behavioral cycle (suppressed/declined) as an
        AiDecision row, mirroring the decline shape of the original finalize."""
        decision = llm_out.decision
        async with async_session_factory() as session:
            async with session.begin():
                memory = ConversationMemoryManager(session)
                await memory.insert_ai_decision(
                    chat_id=prep.chat_id,
                    prompt_version=self.config.ai.prompt_version,
                    snapshot_message_id=prep.snapshot_message_id,
                    new_message_count=prep.new_message_count,
                    should_respond=False,
                    confidence=decision.confidence,
                    response_text=None,
                    reply_to_message_id=None,
                    reasoning=_decline_reasoning(reason, decision),
                    gate_score=prep.gate.gate_score,
                    gate_factors={**prep.gate.gate_factors, **prep.visible_numeric_controls},
                    request1_latency_ms=llm_out.request1.latency_ms,
                    request1_tokens_used=llm_out.request1.tokens_used,
                    request2_tokens_used=llm_out.request2.tokens_used,
                )
                await memory.record_cycle_success(prep.chat_id)

    def _behavioral_suppress_reason(
        self, prep: _CyclePrep, llm_out: _CycleLlmOutcome, now: datetime
    ) -> str | None:
        """Behavioral pre-send suppression checks. Returns a short reason tag when
        the send should be suppressed, else None. Pure (no I/O) given ``now``."""
        rng = self._behavioral_rng
        # Circadian dead window: the bot is "asleep" — never send.
        if humanizer.is_dead_window(now, seed=self.config.behavioral_rng_seed or 0):
            return "dead_window"
        # Still inside a previously-triggered go-dark cooldown: stay silent even though
        # the accusing message has scrolled out of the recent window.
        dark_until = self._dark_until.get(prep.chat_id)
        if dark_until is not None and now < dark_until:
            return "suspicion_dark_cooldown"
        # Suspicion: if the room is accusing a bot, go dark instead of replying.
        recent_texts = [(m.cleaned_text or m.text or "") for m in prep.enriched]
        suspicion = suspicion_monitor.scan_for_accusation(
            recent_texts, bot_username=self.bot_username
        )
        if suspicion.accused:
            dark_s = suspicion_monitor.go_dark_seconds(suspicion, rng)
            # Persist the cooldown so suppression lasts the intended 20min-4h, not just
            # while the accusation stays in the recent window.
            self._dark_until[prep.chat_id] = now + timedelta(seconds=dark_s)
            return f"suspicion_{suspicion.severity}_dark_{int(dark_s)}s"
        # Volume governor: stay an invisible minority of traffic.
        suppress, gov_reason = volume_governor.should_suppress(
            bot_sends_last_hour=prep.responses_last_hour,
            group_msgs_last_hour=prep.group_msgs_last_hour,
            bot_sends_last_10min=prep.bot_sends_last_10min,
            rng=rng,
        )
        if suppress:
            return f"governor_{gov_reason}"
        return None

    async def _finalize_cycle_behavioral(
        self,
        chat_id: int,
        prep: _CyclePrep,
        llm_out: _CycleLlmOutcome,
        now: datetime | None = None,
    ) -> int:
        """Flag-ON finalize: humanized delay + behavioral suppression + output-plan
        send sequence. Opens a fresh short transaction per physical send so the
        delays (awaited here) are never inside an open transaction."""
        now = now or datetime.now(timezone.utc)
        decision = llm_out.decision

        # An LLM-path decline (validate()/social-safety already said no) is recorded
        # exactly as before — no behavioral send.
        if not llm_out.ok:
            await self._record_behavioral_decline(prep, llm_out, llm_out.reason or "declined")
            return self.config.scheduler.initial_interval_seconds

        # Behavioral pre-send suppression (dead window / suspicion / governor).
        if not prep.is_private_dm:
            suppress_reason = self._behavioral_suppress_reason(prep, llm_out, now)
            if suppress_reason:
                await log.ainfo("behavioral_suppress", chat_id=chat_id, reason=suppress_reason)
                await self._record_behavioral_decline(prep, llm_out, suppress_reason)
                return self.config.scheduler.initial_interval_seconds

        # Donor-voice shaping + AI-tell rejection on the outgoing text.
        rng = self._behavioral_rng
        text = decision.response_text or ""
        text = apply_donor_casing(
            text, rng, lowercase_rate=self.config.behavioral_donor_lowercase_rate
        )
        text = strip_terminal_period(text)
        tell = violates_ai_tell(text)
        if tell:
            await log.ainfo("behavioral_ai_tell_rejected", chat_id=chat_id, tell=tell)
            await self._record_behavioral_decline(prep, llm_out, f"ai_tell:{tell}")
            return self.config.scheduler.initial_interval_seconds
        decision.response_text = text

        plan = plan_output(
            text=text,
            reply_to_message_id=decision.reply_to_message_id,
            rng=rng,
            humanizer=humanizer,
            intent_tag=decision.intent_tag,
            burst_rate_target=self.config.behavioral_burst_rate,
            allow_media=self.config.behavioral_allow_media,
        )
        if plan.suppressed or not plan.actions:
            await self._record_behavioral_decline(
                prep, llm_out, f"plan_{plan.suppressed_reason or 'empty'}"
            )
            return self.config.scheduler.initial_interval_seconds

        await self._execute_output_plan(chat_id, prep, llm_out, plan)
        return self.config.scheduler.initial_interval_seconds

    async def _execute_output_plan(
        self,
        chat_id: int,
        prep: _CyclePrep,
        llm_out: _CycleLlmOutcome,
        plan: OutputPlan,
    ) -> None:
        """Execute an OutputPlan's actions in order. Each action's ``delay_before_s``
        is awaited OUTSIDE any open transaction; each physical text send gets its own
        fresh short transaction so a sleep is never held inside one. The first text
        action carries the reply target and records the canonical AiDecision row;
        each text send records its own BotMemory row (bursts => multiple rows)."""
        decision = llm_out.decision
        persisted_posture = decision.updated_engagement_posture or llm_out.posture
        first_text_done = False

        for action in plan.actions:
            if action.delay_before_s > 0:
                # Slice the (possibly multi-minute) humanized delay so the heartbeat
                # stays fresh and a shutdown can abort the send cleanly.
                if await self._sleep_keeping_heartbeat(action.delay_before_s):
                    return

            if action.kind == "react":
                if action.emoji and action.reply_to_message_id is not None:
                    await self.sender.send_reaction(
                        chat_id, action.reply_to_message_id, action.emoji
                    )
                continue

            if action.kind == "media":
                if self.config.behavioral_allow_media:
                    await self.sender.send_sticker(
                        chat_id,
                        source_message_id=action.sticker_source_message_id,
                        reply_to_message_id=action.reply_to_message_id,
                    )
                continue

            if action.kind != "text" or not (action.text or "").strip():
                continue

            sent_message_id = await self.sender.send_message(
                chat_id, action.text or "", action.reply_to_message_id
            )
            # The message is already out. A failure in the post-send recording below
            # must NOT count as a cycle failure (that would trip the circuit breaker for
            # a send that actually succeeded) — log it and move on instead of re-raising.
            try:
                async with async_session_factory() as session:
                    async with session.begin():
                        memory = ConversationMemoryManager(session)
                        if not first_text_done:
                            first_text_done = True
                            await self._record_behavioral_send(
                                memory, prep, llm_out, action, sent_message_id, persisted_posture
                            )
                        else:
                            # Burst follow-up: its own BotMemory row, no extra AiDecision.
                            await memory.insert_bot_memory(
                                chat_id=chat_id,
                                sent_message_id=sent_message_id,
                                response_text=action.text or "",
                                reply_to_user_id=decision.reply_to_user_id,
                                reply_to_message_id=action.reply_to_message_id,
                                reasoning=decision.reasoning,
                                tone_calibration=decision.tone_calibration,
                                brief_snapshot=prep.brief.as_dict(),
                                stances={},
                                prompt_version=self.config.ai.prompt_version,
                                cycle_snapshot_message_id=prep.snapshot_message_id,
                                current_posture=persisted_posture,
                            )
            except Exception:
                first_text_done = True  # send succeeded; treat the cycle as a success
                await log.aexception(
                    "behavioral_post_send_record_failed",
                    chat_id=chat_id,
                    sent_message_id=sent_message_id,
                )

        # If the plan produced no text send at all (e.g. react-only), still close the
        # cycle out so the circuit breaker sees a success.
        if not first_text_done:
            async with async_session_factory() as session:
                async with session.begin():
                    memory = ConversationMemoryManager(session)
                    await memory.record_cycle_success(chat_id)

    async def _record_behavioral_send(
        self,
        memory: ConversationMemoryManager,
        prep: _CyclePrep,
        llm_out: _CycleLlmOutcome,
        action: Action,
        sent_message_id: int,
        persisted_posture: str,
    ) -> None:
        """Record the canonical AiDecision + BotMemory + stances for the lead send,
        matching what the original _finalize_cycle persists on a successful send."""
        decision = llm_out.decision
        stored_decision = await memory.insert_ai_decision(
            chat_id=prep.chat_id,
            prompt_version=self.config.ai.prompt_version,
            snapshot_message_id=prep.snapshot_message_id,
            new_message_count=prep.new_message_count,
            should_respond=True,
            confidence=decision.confidence,
            response_text=action.text,
            reply_to_message_id=action.reply_to_message_id,
            reasoning=(
                (decision.reasoning or "")
                + (
                    f" | posture update: {decision.updated_engagement_posture}"
                    if decision.updated_engagement_posture
                    else ""
                )
            ),
            gate_score=prep.gate.gate_score,
            gate_factors={**prep.gate.gate_factors, **prep.visible_numeric_controls},
            request1_latency_ms=llm_out.request1.latency_ms,
            request1_tokens_used=llm_out.request1.tokens_used,
            request2_tokens_used=llm_out.request2.tokens_used,
        )
        await memory.update_ai_decision_sent_message(stored_decision.id, sent_message_id)
        bot_memory = await memory.insert_bot_memory(
            chat_id=prep.chat_id,
            sent_message_id=sent_message_id,
            response_text=action.text or "",
            reply_to_user_id=decision.reply_to_user_id,
            reply_to_message_id=action.reply_to_message_id,
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
            action.text or "",
        )
        for topic, stance in decision.stances.items():
            await memory.upsert_stance(
                prep.chat_id, topic=topic, stance=str(stance), user_id=decision.reply_to_user_id
            )
            await write_stance_memory(
                memory, prep.chat_id, decision.reply_to_user_id, topic, str(stance)
            )
        # Pass the caller's open session so the pending observation row commits
        # atomically with this BotMemory write (no partial-failure window).
        await self.feedback_loop.schedule_observation(
            bot_memory.id, sent_message_id, prep.chat_id, session=memory.session
        )
        await memory.record_cycle_success(prep.chat_id)

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

    def _direct_override_suppression(
        self,
        gate: GateResult,
        target,
        recent_bot_mem: list,
        enriched,
    ) -> str | None:
        """Why a direct reply/mention may NOT force-proceed past a blocked gate.

        Returns None when the override is allowed. Hard caps the bait loop:
        (a) the 10-min rate cap is absolute, (b) after N consecutive replies to the
        same user with no other human turn in between, that user's direct replies
        stop force-proceeding.
        """
        if "rate_limit_10min" in gate.gate_factors:
            return "rate_limit_10min"
        if self._hit_consecutive_user_cap(target, recent_bot_mem, enriched):
            return "consecutive_user_cap"
        return None

    def _hit_consecutive_user_cap(
        self,
        target,
        recent_bot_mem: list,
        enriched,
    ) -> bool:
        cap = self.config.engagement_gate.max_consecutive_replies_per_user
        if cap <= 0 or target is None or target.sender_id is None:
            return False
        sent = [bm for bm in recent_bot_mem if bm.sent_message_id is not None]  # newest first
        if len(sent) < cap:
            return False
        last_n = sent[:cap]
        if any(bm.reply_to_user_id != target.sender_id for bm in last_n):
            return False
        # "No other human turn in between": since the bot's Nth-last reply, only this
        # user (and the bot itself) have spoken.
        bot_message_ids = {int(bm.sent_message_id) for bm in sent}
        oldest_sent_id = min(int(bm.sent_message_id) for bm in last_n)
        for message in enriched:
            if message.message_id <= oldest_sent_id:
                continue
            if message.sender_id is None or message.sender_id == target.sender_id:
                continue
            if self.bot_user_id is not None and message.sender_id == self.bot_user_id:
                continue
            if message.message_id in bot_message_ids:
                continue
            return False
        return True

    def _anti_flame_bypass_used_recently(self, recent_bot_mem: list | None) -> bool:
        """The anti-flame bypass (responding to a direct reply/mention while tension is
        high) is allowed at most once per same_thread_cooldown_minutes. Derived from
        BotMemory: a response sent within the cooldown whose brief_snapshot recorded
        tension at/above the threshold means the bypass was already spent."""
        if not recent_bot_mem:
            return False
        threshold = self.config.engagement_gate.anti_flame_tension_threshold
        cooldown = timedelta(minutes=self.config.engagement_gate.same_thread_cooldown_minutes)
        now = datetime.now(timezone.utc)
        for bm in recent_bot_mem:
            sent_at = getattr(bm, "sent_at", None)
            if sent_at is None or now - sent_at >= cooldown:
                continue
            snapshot = getattr(bm, "brief_snapshot", None) or {}
            try:
                tension = float(snapshot.get("tension_level", 0.0))
            except (TypeError, ValueError):
                tension = 0.0
            if tension >= threshold:
                return True
        return False

    def _passes_social_safety(
        self,
        *,
        is_private_dm: bool,
        active_bot_thread: bool,
        enriched,
        brief: Brief,
        decision: ResponseDecision,
        bot_sent_ids: set[int],
        recent_bot_mem: list | None = None,
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
            if self._anti_flame_bypass_used_recently(recent_bot_mem):
                return False, "anti_flame_protection"
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

    if config.observe_only:
        log.warning(
            "OBSERVE_ONLY=true: conversation engine is passive — no perception, "
            "decision, or sends. Ingestion + pipeline keep capturing and saving "
            "every message to the DB. Idling."
        )
        try:
            while True:
                # Keep the compose healthcheck green even though no cycle runs,
                # otherwise autoheal would restart-loop the idle container.
                _touch_heartbeat()
                await asyncio.sleep(60)
        finally:
            await dispose_engine()
        return

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
        feedback_tasks = [feedback_loop.run_observation_tasks()]
        if config.feedback_due_at_enabled:
            feedback_tasks.append(feedback_loop.run_due_observation_loop())
        await asyncio.gather(scheduler.run(), *feedback_tasks)
    finally:
        close = getattr(ai_client, "close", None)
        if close:
            await close()
        await sender.close()
        await dispose_engine()


if __name__ == "__main__":
    asyncio.run(main())
