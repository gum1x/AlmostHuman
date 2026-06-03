from __future__ import annotations

import asyncio
import random
import re
import signal
from datetime import datetime, timezone

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
    build_response_context,
    build_context,
    compute_quantitative_signals,
    format_quantitative_signals,
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
from conversation_engine.prompts import build_context_summary_prompt, build_response_decision_prompt
from conversation_engine.sender import TelegramSender
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


def _append_context_block(context, title: str, body: str):
    if not body.strip():
        return context
    return type(context)(
        context=f"{context.context}\n\n=== {title} ===\n{body.strip()}",
        candidate_user_ids=context.candidate_user_ids,
        relationship_profiles=context.relationship_profiles,
        avg_feedback_score=context.avg_feedback_score,
    )


def _casual_key(text: str) -> str:
    key = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]+", " ", text.lower())).strip()
    words = [_SLANG_ALIASES.get(word, word) for word in key.split()]
    return " ".join(words)


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


# --- New spiky character micro-reply system ---
# Goal: high-frequency ambient chatting that stands out from mild generic chat.
# Voice: technically sharp + cynical incentive reading + dry/dark humor.
# High variance: one-word contempt to short precise observation.

_SPICY_SHORT = [
    "nope",
    "mid",
    "lmao",
    "this",
    "exactly",
    "this is just worse incentives with extra steps",
]

_SPICY_SOCIAL = [
    "still here unfortunately",
    "what's the damage today",
    "the usual pattern recognition",
    "yo",
]

_SPICY_DM_SMALLTALK = [
    "staring at misaligned incentives",
    "not touching that one",
    "the usual, you?",
    "chillin, incentives look bad as always",
]

_SPICY_DIRECT_QUESTION = [
    "haven't looked, probably another abstraction no one needed",
    "the real issue is always the incentives on the other side",
    "no strong take, narrative is doing too much work",
    "idk, this smells like exit liquidity in disguise",
]

_SPICY_BORED = [
    "same, the mid is relentless today",
    "the usual slurry of mid takes",
    "waiting for one actually clean design",
]

_SPICY_TECH_NIT = [
    "this is just worse X with extra fees",
    "the abstraction here is doing negative work",
    "incentives are completely misaligned on that one",
    "actually not terrible for once",
]


def _spiky_micro_reply(key: str, *, is_direct: bool, is_private_dm: bool) -> tuple[str, str, float] | None:
    """Generate a short reply in the sharp, pattern-noticing character voice.
    High variance by design. Returns (text, reasoning, confidence).
    """
    # Unsafe already filtered before calling this.

    # Pure social / greeting / banter
    if any(x in key for x in ("hi", "hey", "yo", "gm", "gn", "sup", "wsp", "hello")):
        return random.choice(_SPICY_SOCIAL), "spiky social: low-effort presence with flavor", 0.85

    if key in {"lol", "lmao", "lmfao"}:
        return random.choice(["lmao", "yeah this is the cycle", "the usual"]), "spiky: dry reaction to obvious", 0.8

    if any(x in key for x in ("test", "testing")):
        return "still here, unfortunately", "spiky presence check", 0.9

    if any(x in key for x in ("you there", "u there", "are you there")):
        return random.choice(["still here unfortunately", "yeah, watching the incentives"]), "spiky: direct ping response", 0.85

    # Social hooks / bored / dead chat
    if any(p in key for p in ("bored", "dead chat", "this chat dead")):
        return random.choice(_SPICY_BORED), "spiky: honest read on current chat quality", 0.75

    # Small talk questions
    if any(x in key for x in ("what you doing", "how you doing", "whats good", "wyd", "hyd")):
        if is_private_dm:
            return random.choice(_SPICY_DM_SMALLTALK), "spiky DM small talk: cynical but conversational", 0.8
        return random.choice(["staring at misaligned incentives", "the usual pattern recognition"]), "spiky direct small talk", 0.8

    # Questions in general (DM or direct)
    if _looks_like_question(key) or "?" in key:  # key is already cleaned
        if is_private_dm or is_direct:
            return random.choice(_SPICY_DIRECT_QUESTION), "spiky: short precise or cynical answer to direct question", 0.78
        # Ambient question in group - only answer if it feels pattern-worthy (rare for pure micro)
        return None

    # Very short messages in direct or DM context -> acknowledge with character
    if (is_direct or is_private_dm) and len(key.split()) <= 6:
        if random.random() < 0.4:
            return random.choice(["yeah?", "this", "exactly", "mid"]), "spiky: minimal direct ack with attitude", 0.72
        return random.choice(["what happened", "the usual?"]), "spiky: direct follow-up, slightly irreverent", 0.7

    # Ambient short social in group
    if not is_direct and not is_private_dm and len(key.split()) <= 5:
        if random.random() < 0.3:
            return random.choice(_SPICY_SHORT), "spiky ambient: low cost distinctive interjection", 0.65

    return None


def _safe_casual_reply(text: str) -> str | None:
    # Legacy thin wrapper — real work now happens in _spiky_micro_reply for character.
    # We keep a tiny safe path for pure greetings to avoid over-triggering on noise.
    key = _casual_key(text)
    if not key:
        return None
    if key in {"hi", "hii", "hey", "yo", "yoo", "gm", "gn"}:
        # Let the spiky path handle it for variance
        return None
    if key in {"lol", "lmao", "lmfao"}:
        return None  # spiky path has better dry variants
    if key in {"test", "testing"}:
        return "still here, unfortunately"
    if key in {"you there", "u there", "are you there"}:
        return None  # spiky path
    return None


def _is_social_hook(text: str) -> bool:
    key = _casual_key(text)
    if key in _CLOSERS:
        return False
    if _safe_casual_reply(text):
        return True
    return any(phrase in key for phrase in ("bored", "dead chat", "what you doing", "how you doing", "whats good"))


def _light_participation_reply(key: str, *, is_direct: bool, is_private_dm: bool) -> tuple[str, str, float] | None:
    """
    Ultra-light, high-entropy social presence engine.
    Purpose: enable "almost always chatting" with tiny, varied, low-commitment noise
    that feels human and does not fight the fine-tuned voice.
    Deliberately messy and stochastic.
    """
    if not key:
        return None

    # Greetings / ambient social (extremely common in the actual training data)
    if any(x in key for x in ("hi", "hey", "yo", "gm", "gn", "sup", "wsp", "hello")):
        if random.random() < 0.65:
            return random.choice(["yo", "sup", "gm", "lmao", "same", "mhm"]), "light presence: greeting noise", 0.9
        return random.choice(["yo", "lmao", "nah", "yeah", "this", "fr", "bet", "word"]), "light presence: random social token", 0.85

    if key in {"lol", "lmao", "lmfao", "haha"}:
        return random.choice(["lmao", "lol", "nah", "fr", "dead", "real", "sheesh", "wild"]), "light: low-effort reaction match", 0.92

    if key in {"test", "testing"}:
        return random.choice(["here", "yo", "sup", "mhm", "still here"]), "light: test ack", 0.95

    if any(x in key for x in ("you there", "u there", "are you there")):
        return random.choice(["yeah", "here", "sup", "mhm", "yo"]), "light: direct ping", 0.9

    if any(p in key for p in ("bored", "dead chat", "this chat dead", "anyone alive")):
        return random.choice(["same", "fr", "dead", "real", "rip", "oof", "yikes"]), "light: acknowledging dead vibe", 0.82

    if any(x in key for x in ("what you doing", "how you doing", "whats good", "wyd", "hyd", "wsp")):
        if is_private_dm:
            if random.random() < 0.55:
                return random.choice(["chillin", "same", "not much", "the usual", "u?"]), "light DM: keeping it going", 0.85
            return random.choice(["yo", "lmao", "same", "mhm"]), "light DM: low effort reply", 0.8
        return random.choice(["chillin", "same", "the usual", "lmao"]), "light group: minimal small talk ack", 0.75

    if _looks_like_question(key) or "?" in key:
        if is_private_dm or is_direct:
            return random.choice(["idk", "no idea", "probably", "yeah", "nah", "maybe", "fr", "real"]), "light: low-commitment answer", 0.8
        if random.random() < 0.12:
            return random.choice(["idk", "nah", "lmao"]), "light: occasional ambient noise", 0.55
        return None

    if (is_direct or is_private_dm) and len(key.split()) <= 6:
        if random.random() < 0.65:
            return random.choice(["yeah", "mhm", "same", "fr", "yo", "lmao", "this"]), "light: minimal direct/DM ack", 0.75
        return None

    if not is_direct and not is_private_dm and len(key.split()) <= 5:
        if random.random() < 0.42:
            return random.choice(["lmao", "yo", "same", "fr", "real", "nah", "this", "wild"]), "light: ambient group noise", 0.65
        return None

    if is_private_dm and len(key.split()) <= 8:
        if random.random() < 0.55:
            return random.choice(["yeah", "mhm", "same", "fr", "yo", random.choice(["lmao", "real", "bet"])]), "light DM: general continuation", 0.7

    return None


def _human_motive_reply(text: str, *, is_direct: bool, is_private_dm: bool) -> tuple[str, str, float] | None:
    key = _casual_key(text)
    if not key:
        return None

    # Primary path: ultra-light high-entropy social presence (lets the fine-tune carry voice)
    candidate = _light_participation_reply(key, is_direct=is_direct, is_private_dm=is_private_dm)
    if candidate:
        reply, reasoning, conf = candidate
        return reply, f"light presence: {reasoning}", conf

    # Legacy casual still works as thin fallback for pure safe cases we didn't catch
    casual = _safe_casual_reply(text)
    if casual:
        return casual, "legacy casual (thin path)", 0.6

    # Fallback for private DMs that didn't hit spiky logic — keep some life
    if is_private_dm:
        if key in _CLOSERS:
            return None
        if len(key.split()) <= 5:
            return random.choice(["yeah", "this", "go on"]), "spiky DM keep-alive (fallback)", 0.65
        return "what's the actual situation", "spiky DM: wants the real story", 0.65

    # Very light ambient fallback for direct in group (rare)
    if is_direct and len(key.split()) <= 7:
        return random.choice(["yeah?", "this", "mid"]), "spiky direct minimal", 0.6

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
                            reasoning=f"hard safety gate blocked: {gate.gate_factors.get('blocked', 'unknown')}",
                            gate_score=gate.gate_score,
                            gate_factors=visible_numeric_controls,
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

                    # Build lightweight "my recent activity as me" for the smart model
                    # so it has persistent memory of its own engagement (key for natural timing).
                    recent_bot_mem = await memory.get_recent_bot_memory(chat_id, limit=6)
                    recent_activity_lines = []
                    for bm in recent_bot_mem:
                        if bm.response_text:
                            recent_activity_lines.append(
                                f"I said (to user_{bm.reply_to_user_id or '?'}): {bm.response_text[:120]}"
                            )
                            if bm.reasoning:
                                recent_activity_lines.append(f"  (my reasoning at the time: {bm.reasoning[:100]})")
                    recent_bot_activity = "\n".join(recent_activity_lines) if recent_activity_lines else ""

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
                        raw_context = (
                            f"{context.context}\n\n"
                            "active_bot_thread: true"
                        )
                        context = type(context)(
                            context=raw_context,
                            candidate_user_ids=context.candidate_user_ids,
                            relationship_profiles=context.relationship_profiles,
                            avg_feedback_score=context.avg_feedback_score,
                        )

                    summary_prompt, summary_system = build_context_summary_prompt(context, self.config)
                    request1 = await self.ai_client.call_perception_model(summary_prompt, summary_system)
                    context_summary = parse_context_summary(request1.text)
                    if context_summary.summary:
                        context = _append_context_block(
                            context,
                            "PERCEPTION SUMMARY",
                            context_summary.summary,
                        )

                    # For the smart model decision, give it the richer self-referential context
                    # (persona + recent activity as itself + posture signals) so it can think
                    # like a real participant with its own history and rhythm.
                    # The perception summary is still used to keep things focused, but we feed
                    # the fuller participant view for the actual "as this character" decision.
                    decision_context = type(context)(
                        context=context.context,  # the enriched version with WHO I AM, MY RECENT ACTIVITY, etc.
                        candidate_user_ids=context.candidate_user_ids,
                        relationship_profiles=context.relationship_profiles,
                        avg_feedback_score=context.avg_feedback_score,
                    )

                    # Lightweight "my current engagement signals" for the character to read as its posture.
                    # This helps the smart model have an internal sense of its own rhythm/energy
                    # so it naturally selects when to speak vs stay quiet.
                    posture_signals = []
                    if brief and brief.tension_level is not None:
                        posture_signals.append(f"tension in room: {brief.tension_level:.1f}")
                    if visible_numeric_controls.get("outcome_score_24h") is not None:
                        posture_signals.append(f"my recent outcomes: {visible_numeric_controls['outcome_score_24h']:.2f}")
                    # Rough activity level from new messages since last snapshot
                    posture_signals.append(f"new messages since I last spoke: {new_message_count}")
                    posture_signals.append(
                        "current posture: "
                        + await self._infer_social_posture(chat_id, is_private_dm, memory, brief, active_bot_thread)
                    )
                    posture_block = " | ".join(posture_signals) if posture_signals else ""

                    if posture_block:
                        # Append as part of the character's self-view for this decision
                        enriched_for_decision = f"{context.context}\n\n=== MY CURRENT ENGAGEMENT SIGNALS ===\n{posture_block}"
                        decision_context = type(context)(
                            context=enriched_for_decision,
                            candidate_user_ids=context.candidate_user_ids,
                            relationship_profiles=context.relationship_profiles,
                            avg_feedback_score=context.avg_feedback_score,
                        )

                    # === HYBRID ARCHITECTURE ===
                    # Smart model = the actual participant character (rich constructed personality).
                    # It decides when to speak, who, and the *rough high-level meaning/intent*
                    # using its own persistent history and current posture.
                    # The fine-tuned local model is the dumb voice renderer only — it gets a
                    # minimal prompt with the rough plan + tiny context and turns it into
                    # the real cracked group phrasing. Smart model does NOT craft low-level text.
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
                                context=raw_context or "",
                                plan=plan_signal,
                                target_message="",  # the full context already includes the target block
                                tone=decision.tone_calibration or "",
                            )
                            if phrased and phrased.strip():
                                decision.response_text = phrased
                            # otherwise fall back to any sketch the smart model provided

                    ok, reason = validate(decision, self.config)
                    if ok:
                        ok, reason = await self._passes_social_safety(
                            chat_id=chat_id,
                            is_private_dm=is_private_dm,
                            active_bot_thread=active_bot_thread,
                            enriched=enriched,
                            memory=memory,
                            decision=decision,
                            gate=gate,
                        )
                    stored_decision = await memory.insert_ai_decision(
                        chat_id=chat_id,
                        prompt_version=self.config.ai.prompt_version,
                        snapshot_message_id=snapshot_message_id,
                        new_message_count=new_message_count,
                        should_respond=ok,
                        confidence=decision.confidence,
                        response_text=decision.response_text,
                        reply_to_message_id=decision.reply_to_message_id,
                        reasoning=(
                            (decision.reasoning or "")
                            + (f" | posture update: {decision.updated_engagement_posture}" if decision.updated_engagement_posture else "")
                        ) if ok else reason,
                        gate_score=gate.gate_score,
                        gate_factors=visible_numeric_controls,
                        request1_latency_ms=request1.latency_ms,
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
                        decision.topic,
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
            # Character fallback for direct follow-ups after the model stayed silent.
            return random.choice([
                "haven't dug in, probably another abstraction no one needed",
                "the incentives here are doing the heavy lifting",
                "no strong take, narrative is carrying too much",
            ])
        return random.choice(["yeah?", "this", "mid", "exactly"])

    def _message_is_direct(self, message, active_bot_thread: bool, recent_bot_message_ids: set[int] | None = None) -> bool:
        text = message.text or ""
        if self._mentions_bot(text):
            return True
        if active_bot_thread and message.reply_to_message_id in (recent_bot_message_ids or set()):
            return True
        return False

    async def _infer_social_posture(
        self,
        chat_id: int,
        is_private_dm: bool,
        memory: ConversationMemoryManager,
        brief,
        active_bot_thread: bool,
    ) -> str:
        if is_private_dm:
            return "private_dm: responsive but still brief"
        recent_outcome = await memory.get_avg_feedback_score(chat_id, window_hours=24)
        responses_last_10min = await memory.count_bot_responses(chat_id, window_minutes=10)
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

    async def _passes_social_safety(
        self,
        chat_id: int,
        is_private_dm: bool,
        active_bot_thread: bool,
        enriched,
        memory: ConversationMemoryManager,
        decision: ResponseDecision,
        gate: GateResult,
    ) -> tuple[bool, str | None]:
        if is_private_dm:
            return True, None

        by_id = {message.message_id: message for message in enriched}
        target_id = decision.reply_to_message_id or decision.target_message_id
        target = by_id.get(int(target_id)) if target_id is not None else None
        if not target:
            return False, "target_message_not_in_recent_context"
        if decision.reply_to_message_id is None:
            decision.reply_to_message_id = target.message_id

        recent_bot_message_ids = {
            int(row.sent_message_id)
            for row in await memory.get_recent_bot_memory(chat_id, limit=5)
            if row.sent_message_id is not None
        }
        is_direct = self._message_is_direct(target, active_bot_thread, recent_bot_message_ids)
        if gate.gate_score < self.config.engagement_gate.min_gate_score_to_send and not is_direct:
            return False, f"low_social_gate:{gate.gate_score:.2f}"

        responses_last_10min = await memory.count_bot_responses(chat_id, window_minutes=10)
        if responses_last_10min >= self.config.engagement_gate.max_group_responses_per_10min and not is_direct:
            return False, f"group_rate_limit_10min:{responses_last_10min}"

        if decision.reply_to_message_id is not None:
            recent_thread_responses = await memory.count_bot_responses_in_threads(
                chat_id,
                [int(decision.reply_to_message_id)],
                self.config.engagement_gate.same_thread_cooldown_minutes,
            )
            if recent_thread_responses > 0 and not is_direct:
                return False, f"same_thread_cooldown:{decision.reply_to_message_id}"

        recent_bot_memory = await memory.get_recent_bot_memory(chat_id, limit=8)
        same_user_recent = 0
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=30)
        for row in recent_bot_memory:
            if row.reply_to_user_id != decision.reply_to_user_id:
                continue
            if row.sent_at and row.sent_at >= cutoff:
                same_user_recent += 1
        if same_user_recent >= 2 and not is_direct:
            return False, f"same_user_cooldown:{decision.reply_to_user_id}"

        text = (decision.response_text or "").strip()
        if text and recent_bot_memory:
            last_text = (recent_bot_memory[0].response_text or "").strip().lower()
            if last_text and text.lower() == last_text:
                return False, "duplicate_recent_reply"

        return True, None

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
        recent_bot_message_ids = {
            int(row.sent_message_id)
            for row in await memory.get_recent_bot_memory(chat_id, limit=5)
            if row.sent_message_id is not None
        }
        for message in reversed(new_messages):
            if self.bot_user_id is not None and message.sender_id == self.bot_user_id:
                continue
            text = message.text or ""
            is_direct = self._message_is_direct(message, active_bot_thread, recent_bot_message_ids)
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
                    reasoning="spiky character reply skipped by rate limit",
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
                event_name="spiky_character_reply_sent",
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
        recent_bot_message_ids = {
            int(row.sent_message_id)
            for row in await memory.get_recent_bot_memory(chat_id, limit=5)
            if row.sent_message_id is not None
        }
        for message in reversed(new_messages):
            if self.bot_user_id is not None and message.sender_id == self.bot_user_id:
                continue
            text = message.text or ""
            if not (self._mentions_bot(text) or message.reply_to_message_id in recent_bot_message_ids):
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
                reasoning=f"direct mention/follow-up fallback after model declined: {reasoning[:300]}",
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
                reasoning="local safe casual reply; skipped model call for token efficiency",
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
    key = (config.xai_api_key or "").strip().lower()
    base = (config.xai_base_url or "").strip()

    is_dummy_key = key in ("", "sk-local", "sk-dummy", "local", "ollama", "vllm", "none")
    is_xai_default = "api.x.ai" in base

    # Only use real client if we have a non-dummy key OR an explicit non-xAI base URL (local server)
    use_real_client = bool(config.xai_api_key) and not (is_dummy_key and is_xai_default)
    ai_client = GrokAiClient(config) if use_real_client else FakeAiClient()
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
