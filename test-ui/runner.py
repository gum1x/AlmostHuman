"""
Workflow runner that executes the full conversation engine pipeline
on injected JSON chat data — no database or Telegram connection needed.

Mocks: memory manager, sender (we capture the response instead of sending).
Real: enrichment, gate scoring, context building (with persona + recent activity
      memory injection), slim activity numbers (the kept high-value counts),
      AI calls (now using the qualitative three-question decision prompt),
      style rewriter (local model phrasing), validation.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from conversation_engine.ai_client import (
    GrokAiClient,
    FakeAiClient,
    parse_context_summary,
    parse_response_decision,
    ResponseDecision,
    ContextSummary,
)
from conversation_engine.config import EngineConfig, load_engine_config
from conversation_engine.context_builder import (
    ContextBundle,
    build_target_message_block,
    compute_quantitative_signals,
    format_quantitative_signals,
)
from conversation_engine.engagement_gate import compute_gate_score, GateResult, get_candidate_user_ids
from conversation_engine.enrichment import (
    EnrichedMessage,
    Brief,
    enrich_messages,
    build_brief,
    sentiment_score,
)
from conversation_engine.prompts import (
    build_context_summary_prompt,
    build_response_decision_prompt,
    SMART_PARTICIPANT_SYSTEM,
)
from conversation_engine.style_rewriter import LocalStyleRewriter
from conversation_engine.validators import validate
from storage.postgres_models import BotPersonaCore, BotSelfReflection, UserRelationshipProfile


def _make_context_preview(full_context: str, max_chars: int = 600) -> str:
    """Return a preview that shows the start + the end (where recent activity / posture / precomp live).
    This helps in the Test UI see the memory blocks the qualitative 3-question model is using.
    """
    if len(full_context) <= max_chars:
        return full_context
    head = full_context[: max_chars // 2]
    tail = full_context[- (max_chars // 2): ]
    return head + "\n... [middle truncated for preview] ...\n" + tail


# =============================================================================
# TUNABLE PARAMETERS FOR THE TEST UI
# Gate weights + threshold (real pre-filter before any LLM) + the willingness bias dial.
# The decision model is now purely qualitative (three questions: situation / what kind of
# person am I / what does a person like me do, grounded in injected WHO I AM + recent activity
# + posture memory). The PRE-COMPUTED SIGNALS block now only contains the slim high-value
# raw activity numbers (responses_last_hour, time_since, is_reply_to_bot, direct_mention) plus the willingness
# value if the dial is used. The gate threshold is a simple pre-filter (hard skip before LLM).
# Detailed gate weights are no longer tunable in the UI (qualitative 3Q model does not listen to them).
# Willingness and slim counts + direct_mention are what the model sees in PRE-COMPUTED.
# =============================================================================

TUNABLE_META = {
    # Simplified dials for the current architecture.
    # Gate is now a simple cheap structural pre-filter (threshold only) before the qualitative 3Q model.
    # Detailed per-factor weights are no longer exposed (the 3Q reasoning model does not consume
    # the gate score or its internal factors; it only sees the slim activity numbers + direct_mention
    # + posture + optional willingness bias in PRE-COMPUTED SIGNALS).
    # Direct mentions/continuations always force proceed past gate to the reasoning AI.
    "min_gate_score_to_send": {
        "label": "Gate Threshold (min score to even consider responding)",
        "min": 0.0, "max": 1.0, "step": 0.01, "default": 0.25,
        "desc": "Cheap pre-filter score must be >= this or we hard-skip the model (no LLM call). Lower = more willing to think on marginal social moments. Directs always bypass."
    },
    "willingness_to_respond": {
        "label": "Willingness to Respond (social / presence bias)",
        "min": 0.0, "max": 1.0, "step": 0.05, "default": 0.75,
        "desc": "Pure test dial (appended raw to the slim PRE-COMPUTED SIGNALS block). The qualitative decision model (three questions grounded in persona memory) can attend to it as an explicit bias. The character is a bold old timer who is actively in the mix, speaks when the energy or opinion pulls them, enjoys the chaos, and is more willing to chat and participate."
    },
}

READONLY_SIGNALS = {"is_reply_to_bot"}
# is_reply_to_bot is derived from the input message list + previous_bot_memories (the bot_sent check that scans history). You cannot "edit" it without editing the source chat JSON or memories. (chat_velocity was removed from the slim activity numbers block.)


# ---------------------------------------------------------------------------
# Lightweight fake memory manager so we can call the *real* gate (simple threshold pre-filter) + slim
# compute_quantitative_signals (the high-value activity counts the 3Q model actually sees) without a DB.
# Seeded from previous_bot_memories for posture + fatigue/time_since. Detailed gate weights are no longer
# primary UI dials (qualitative model does not listen to them).
# ---------------------------------------------------------------------------

@dataclass
class _BotMem:
    """Duck-type stand-in for storage.postgres_models.BotMemory rows."""
    current_posture: str | None = None
    response_text: str | None = None
    reasoning: str | None = None
    created_at: datetime | None = None
    reply_to_user_id: int | None = None
    sent_message_id: int | None = None


class FakeConversationMemoryManager:
    """Test-only in-memory implementation of the bits of ConversationMemoryManager
    that compute_gate_score, _infer_social_posture, and context prep use.
    """

    def __init__(self, previous_bot_memories: list[dict[str, Any]] | None = None, bot_user_id: int | None = None):
        self._mems: list[_BotMem] = []
        prev = previous_bot_memories or []
        for m in prev:
            self._mems.append(_BotMem(
                current_posture=m.get("current_posture") or m.get("posture") or m.get("updated_engagement_posture"),
                response_text=m.get("response_text") or m.get("text"),
                reasoning=m.get("reasoning"),
                created_at=self._parse_ts(m.get("created_at") or m.get("timestamp")),
                reply_to_user_id=m.get("reply_to_user_id"),
                sent_message_id=m.get("sent_message_id"),
            ))
        self.bot_user_id = bot_user_id or 0
        self._bot_response_count = sum(1 for m in self._mems if m.response_text)

    def _parse_ts(self, ts: Any) -> datetime | None:
        if ts is None:
            return None
        if isinstance(ts, datetime):
            return ts
        try:
            return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except Exception:
            return datetime.now(timezone.utc)

    async def count_messages_in_window(self, chat_id: int, minutes: int = 10) -> int:
        # Plausible activity for a test chat so velocity factor isn't zero.
        return 12

    async def count_bot_responses(self, chat_id: int, window_minutes: int = 60) -> int:
        return max(0, min(12, self._bot_response_count))

    async def avg_relationship_strength(self, chat_id: int, candidate_users: list[int]) -> float:
        return 0.6

    async def get_activity_pattern(self, chat_id: int, hour: int, weekday: int):
        return None

    async def count_bot_responses_in_threads(self, chat_id: int, thread_ids: list[int], window_minutes: int) -> int:
        return 0

    async def get_avg_feedback_score(self, chat_id: int, window_hours: int = 24) -> float:
        return 0.2

    async def get_relationship_profiles(self, chat_id: int, user_ids: list[int]):
        return []

    async def get_recent_bot_memory(self, chat_id: int, limit: int = 6):
        # newest first like real
        return list(reversed(self._mems[-limit:])) if self._mems else []

    async def get_persona_core(self):
        return None

    async def insert_bot_memory(self, chat_id: int = 0, **kwargs: Any) -> None:
        mem = _BotMem(
            current_posture=kwargs.get("current_posture"),
            response_text=kwargs.get("response_text"),
            reasoning=kwargs.get("reasoning"),
            created_at=datetime.now(timezone.utc),
            reply_to_user_id=kwargs.get("reply_to_user_id"),
            sent_message_id=kwargs.get("sent_message_id"),
        )
        self._mems.append(mem)
        if kwargs.get("response_text"):
            self._bot_response_count += 1

    async def latest_message_id(self, chat_id: int) -> int | None:
        return 100000

    async def upsert_activity_pattern(self, *args: Any, **kwargs: Any) -> None:
        pass


def _infer_posture_for_test(
    brief: Brief,
    avg_feedback: float,
    responses_last_10min: int,
    tension_threshold: float = 0.75,
    active_thread: bool = False,
    previous_bot_memories: list[dict[str, Any]] | None = None,
) -> str:
    """Minimal version of scheduler._infer_social_posture for test runs (no full config/mem)."""
    # Hyperactive streak from previous_bot_memories (seeded with current_posture from prior responses)
    prevs = previous_bot_memories or []
    if prevs:
        last = prevs[-1] if prevs else {}
        lp = last.get("current_posture") or last.get("posture") or last.get("updated_engagement_posture") or ""
        if "hyperactive" in lp.lower() or "engaged" in lp.lower():
            return "old timer: recently spoke, might speak again if another moment or opinion strikes"
    if brief.tension_level >= tension_threshold:
        return "burned/quiet: tension is high"
    if avg_feedback <= -0.25:
        return "burned/quiet: recent replies did not land"
    if active_thread:
        return "in_thread: direct follow-up exists"
    if responses_last_10min >= 2:
        return "lurking: already spoke recently"
    if brief.tension_level <= 0.25:
        return "lightly_vibing: available for high-signal or funny moments or just to be present after long silence"
    return "watching: selective and low-ego"


# ---------------------------------------------------------------------------
# Lightweight message stub that mimics storage.postgres_models.Message
# just enough for enrich_messages() to work.
# ---------------------------------------------------------------------------

class MessageStub:
    """Minimal duck-type of storage.postgres_models.Message for enrichment."""

    def __init__(self, data: dict[str, Any]):
        self.message_id: int = int(data["message_id"])
        self.chat_id: int = int(data.get("chat_id", -1001234567890))
        self.sender_id: int | None = data.get("sender_id")
        self.text_raw: str | None = data.get("text_raw") or data.get("text") or ""
        self.text_cleaned: str | None = data.get("text_cleaned") or data.get("text") or self.text_raw
        self.reply_to_message_id: int | None = data.get("reply_to_message_id")
        ts = data.get("timestamp")
        if isinstance(ts, str):
            try:
                self.timestamp = datetime.fromisoformat(ts)
            except ValueError:
                self.timestamp = datetime.now(timezone.utc)
        elif isinstance(ts, (int, float)):
            self.timestamp = datetime.fromtimestamp(ts, tz=timezone.utc)
        else:
            self.timestamp = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Step results collected during a pipeline run
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    name: str
    duration_ms: int = 0
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class PipelineResult:
    chat_id: int
    steps: list[StepResult] = field(default_factory=list)
    response_text: str | None = None
    decision: dict[str, Any] | None = None
    total_duration_ms: int = 0


# ---------------------------------------------------------------------------
# Fake context builder that works without DB
# ---------------------------------------------------------------------------

# (Real compute_gate_score + slim compute_quantitative_signals + Fake mem are used so that
# the threshold dial affects the hard pre-filter skip, and willingness + slim counts affect
# the block the qualitative 3Q decision model receives. Detailed gate weights are not exposed
# in TUNABLE_META because the 3Q model does not listen to gate internals.)

# ---------------------------------------------------------------------------
# Main pipeline runner
# ---------------------------------------------------------------------------

async def run_pipeline(
    messages_json: list[dict[str, Any]],
    config: EngineConfig | None = None,
    use_real_ai: bool = True,
    target_message_id: int | None = None,
    previous_bot_memories: list[dict[str, Any]] | None = None,
    bot_user_id: int | None = 0,
    overrides: dict[str, Any] | None = None,
) -> PipelineResult:
    """
    Run the full conversation engine pipeline on a list of JSON messages.
    This is the "live tuning" entrypoint for the test UI: the Gate Threshold dial
    controls the cheap pre-filter hard-skip (before any LLM). The Willingness dial
    is appended to the slim PRE-COMPUTED SIGNALS block (responses_last_hour,
    time_since_last_bot_msg_min, is_reply_to_bot, direct_mention + willingness)
    that the qualitative decision model (three questions) receives and can attend to.

    Only overrides for keys in TUNABLE_META (and not in READONLY_SIGNALS) are
    honored. Detailed gate factor weights are no longer primary UI dials (the 3Q
    model does not listen to gate internals). The decision model answers the three
    qualitative questions grounded in the injected persona memory + recent "I said"
    activity + relevant compressed context + direct mention rule.

    previous_bot_memories: list of prior {"current_posture":, "response_text":, "reasoning":, ...}
      used to seed Fake mem for realistic fatigue, posture roundtrip, time_since,
      bot_sent_ids for is_reply checks, *and* to reconstruct "=== MY RECENT ACTIVITY AS ME ==="
      in the context passed to the decision model. This ensures the qualitative
      three-question prompt ("What kind of situation...? What kind of person am I?
      What does a person like me do...?") has the character's own history to reason from,
      exactly as production does. Server accumulates these across /api/send turns.

    The provided messages_json (the "chat") is split into high-level (~200 most recent)
    and recent (~10) windows and passed to build_context_summary_prompt. This runs the
    two-level relevance compressor (the "different prompt") before the 3Q decision so the
    reasoning AI receives a compressed_relevant_context (with selective exact quotes from
    high-level only when relevant to the current recent/target). "direct_mention" is also
    computed (and overridable) and appears in the PRE-COMPUTED block; the decision prompt
    contains a hard rule that the bot must engage on direct mentions/continuations.
    """
    t0 = time.perf_counter()
    config = config or load_engine_config()
    chat_id = messages_json[0].get("chat_id", -1001234567890) if messages_json else -1001234567890
    result = PipelineResult(chat_id=chat_id)
    overrides = dict(overrides or {})

    # Extract gate-specific overrides (threshold for pre-filter; weights kept for power-user / API use
    # even though they are no longer primary dials in the UI — the 3Q model does not listen to gate internals).
    gate_weight_overrides: dict[str, float] = {}
    min_gate_override: float | None = None
    for k, v in list(overrides.items()):
        if k == "min_gate_score_to_send":
            try:
                min_gate_override = float(v)
            except Exception:
                pass
        elif k.startswith("weight_"):
            real_key = k[7:]  # weight_fatigue -> fatigue
            try:
                gate_weight_overrides[real_key] = float(v)
            except Exception:
                pass

    # --- Step 1: Parse messages ---
    t1 = time.perf_counter()
    stubs = [MessageStub(m) for m in messages_json]
    result.steps.append(StepResult(
        name="parse_messages",
        duration_ms=int((time.perf_counter() - t1) * 1000),
        data={"count": len(stubs), "target_message_id": target_message_id},
    ))

    # --- Step 2: Enrichment ---
    t1 = time.perf_counter()
    enriched = enrich_messages(stubs, config.prompt)
    result.steps.append(StepResult(
        name="enrichment",
        duration_ms=int((time.perf_counter() - t1) * 1000),
        data={
            "count": len(enriched),
            "avg_sentiment": round(sum(m.sentiment_score for m in enriched) / max(1, len(enriched)), 3),
            "topics_found": list({t for m in enriched for t in m.topics}),
        },
    ))

    # High-level (~200) / recent (~10) split for the context summarizer (the "different prompt"
    # that produces the compressed relevant context the 3Q reasoning AI actually sees).
    # Mirrors the prod scheduler fetch + slice. The input "msgs" to the test is treated as the
    # full available chat history for the scenario.
    n = len(enriched)
    high_level_enriched = enriched[-200:] if n > 200 else list(enriched)
    recent_enriched = enriched[-10:] if n > 10 else list(enriched)

    # --- Step 3: Brief ---
    t1 = time.perf_counter()
    brief = build_brief(enriched)
    result.steps.append(StepResult(
        name="brief",
        duration_ms=int((time.perf_counter() - t1) * 1000),
        data=brief.as_dict(),
    ))

    # --- Step 4: Gate (REAL compute_gate_score + overrides) ---
    # Gate is a simple pre-filter (threshold controls hard skip before LLM). The 3Q qualitative
    # model does not receive or "listen to" the gate score/factors (only the slim PRE-COMPUTED counts
    # + direct_mention + willingness if set + posture + persona). Detailed weights dials removed from UI.
    t1 = time.perf_counter()
    fake_mem = FakeConversationMemoryManager(previous_bot_memories=previous_bot_memories, bot_user_id=bot_user_id)
    gate = await compute_gate_score(
        chat_id=chat_id,
        enriched_messages=enriched,
        brief=brief,
        memory=fake_mem,
        config=config,
        weights_override=gate_weight_overrides or None,
        min_score_override=min_gate_override,
    )
    gate_data = {
        "gate_score": round(gate.gate_score, 3),
        "should_proceed": gate.should_proceed,
        "factors": {k: round(v, 3) if isinstance(v, float) else v for k, v in gate.gate_factors.items()},
    }
    if gate_weight_overrides or min_gate_override is not None:
        # Note: weight overrides are for advanced use; UI no longer surfaces the per-factor weight dials.
        gate_data["overrides_applied"] = {**gate_weight_overrides, **({"min_gate_score_to_send": min_gate_override} if min_gate_override is not None else {})}
    result.steps.append(StepResult(
        name="engagement_gate",
        duration_ms=int((time.perf_counter() - t1) * 1000),
        data=gate_data,
    ))

    if not gate.should_proceed:
        result.steps.append(StepResult(name="blocked", data={"reason": "gate_blocked", "gate_score": gate.gate_score}))
        result.total_duration_ms = int((time.perf_counter() - t0) * 1000)
        return result

    # --- Step 4b: Timing classifier pre-gate (mirrors production scheduler) ---
    # When enabled, this is the data-trained "would a regular bother to reply?" filter
    # that runs before any LLM call. Matches conversation_engine/scheduler.py.
    if getattr(config, "timing_classifier_enabled", False):
        t1 = time.perf_counter()
        target_tc = None
        for m in reversed(enriched):
            if (m.cleaned_text or m.text or "").strip():
                target_tc = m
                break
        if target_tc is not None:
            from conversation_engine.timing_classifier import TimingClassifier
            tc = TimingClassifier(model_path=config.timing_classifier_model_path)
            if config.timing_classifier_threshold and config.timing_classifier_threshold > 0:
                tc.threshold = config.timing_classifier_threshold
            t_txt = target_tc.cleaned_text or target_tc.text or ""
            ts = tc.score(
                text=t_txt,
                is_reply=target_tc.reply_to_message_id is not None,
                reply_to_regular=False,
                sender_is_regular=True,
                idx_gap_since_sender=-1,
            )
            result.steps.append(StepResult(
                name="timing_classifier",
                duration_ms=int((time.perf_counter() - t1) * 1000),
                data={
                    "enabled": True,
                    "p": round(ts.score, 3),
                    "threshold": tc.threshold,
                    "passes": ts.passes,
                    "is_botlike": ts.is_botlike,
                    "target_text": t_txt[:120],
                },
            ))
            if not ts.passes:
                result.steps.append(StepResult(
                    name="blocked",
                    data={"reason": "timing_classifier_skip", "p": round(ts.score, 3), "threshold": tc.threshold},
                ))
                result.total_duration_ms = int((time.perf_counter() - t0) * 1000)
                return result

    # --- Step 5: Slim activity numbers (high-value counts only) + overrides ---
    # The decision model uses the qualitative three-question prompt. We compute/pass the kept
    # raw counts (responses_last_hour tracker, time_since, is_reply_to_bot + direct_mention checks)
    # + willingness (if dialed) so the 3Q model has awareness of volume, directness, and test bias.
    # Gate threshold (if tuned) already affected the pre-filter above; detailed weights not exposed.
    t1 = time.perf_counter()

    # time_since: prefer explicit override (for "what if long silence"), else derive from previous_bot_memories, else -1 (long unknown)
    time_since = overrides.get("time_since_last_bot_msg_min")
    if time_since is None:
        if previous_bot_memories:
            last = previous_bot_memories[-1]
            ts = last.get("created_at") or last.get("timestamp")
            if ts:
                try:
                    last_dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                    time_since = round((datetime.now(timezone.utc) - last_dt).total_seconds() / 60.0, 1)
                except Exception:
                    time_since = 60.0  # assume decent silence for test carry-over
            else:
                time_since = 60.0
        else:
            time_since = -1  # classic "hasn't spoken in a while" starting state

    # responses and feedback can be overridden for fatigue/what-if experiments
    responses_lh = overrides.get("responses_last_hour")
    if responses_lh is None:
        responses_lh = await fake_mem.count_bot_responses(chat_id, 60)
    avg_fb = overrides.get("avg_feedback_24h")
    if avg_fb is None:
        avg_fb = await fake_mem.get_avg_feedback_score(chat_id, 24)

    # bot_sent for is_reply detection in the (now slim) activity numbers
    bot_sent = {m.get("sent_message_id") for m in (previous_bot_memories or []) if m.get("sent_message_id") is not None}

    quant = compute_quantitative_signals(
        enriched_messages=enriched,
        bot_user_id=bot_user_id,
        bot_sent_message_ids=bot_sent,
        time_since_last_bot_msg_min=time_since,
        responses_last_hour=responses_lh,
        bot_username=overrides.get("bot_username"),
    )

    # Apply user overrides, but never to readonly derived fields.
    # Only the slim set from TUNABLE_META (weights + threshold + willingness) are exposed in the UI.
    applied_overrides: dict[str, Any] = {}
    for k, v in overrides.items():
        if k in READONLY_SIGNALS:
            continue
        if k in quant:
            try:
                quant[k] = float(v) if isinstance(quant[k], (int, float)) else v
                applied_overrides[k] = v
            except Exception:
                pass
        elif k == "willingness_to_respond":
            try:
                quant[k] = float(v)
                applied_overrides[k] = v
            except Exception:
                pass

    # Re-format the block that the decision model will actually see (slim activity numbers + optional willingness).
    # The qualitative model (three questions) receives this as raw facts about recent output volume.
    signals_block = format_quantitative_signals(quant)
    if "willingness_to_respond" in applied_overrides:
        signals_block += f" | willingness_to_respond={applied_overrides['willingness_to_respond']}"

    # Posture: prefer persisted from previous mems (roundtrip), else infer for this snapshot.
    # This is critical for the "after long silence" presence logic.
    recent_mems = await fake_mem.get_recent_bot_memory(chat_id, 3)
    latest_persisted = recent_mems[0].current_posture if recent_mems and getattr(recent_mems[0], "current_posture", None) else None

    avg_fb_post = overrides.get("avg_feedback_24h", avg_fb)
    resp_10min = await fake_mem.count_bot_responses(chat_id, 10)
    tension_thr = overrides.get("anti_flame_tension_threshold", config.engagement_gate.anti_flame_tension_threshold)
    posture = latest_persisted or _infer_posture_for_test(brief, avg_fb_post, resp_10min, tension_thr, previous_bot_memories=previous_bot_memories)

    # Long silence bias (matches the prompt rule we added): after quiet, default posture leans available for social/chaos presence
    ts_val = quant.get("time_since_last_bot_msg_min", -1)
    if (ts_val is None or ts_val > 30 or ts_val == -1) and "burned" not in (posture or ""):
        if "vibing" not in (posture or ""):
            posture = "lightly_vibing: available for high-signal or funny moments or just to be present after long silence"

    # Reconstruct "MY RECENT ACTIVITY AS ME" from the seeded previous_bot_memories.
    # This is critical for the qualitative decision model: the three questions
    # ("What kind of situation is this? What kind of person am I? What does a person
    # like me do in a situation like this?") explicitly tell the model to ground
    # answers in the injected memory, especially the concrete "I said..." history
    # so it knows its own recent behavior and rhythm (matching the real scheduler path).
    recent_bot_mem_for_activity = await fake_mem.get_recent_bot_memory(chat_id, limit=6)
    recent_activity_lines = []
    for bm in recent_bot_mem_for_activity:
        if getattr(bm, "response_text", None):
            recent_activity_lines.append(
                f"I said (to user_{getattr(bm, 'reply_to_user_id', None) or '?'}): {bm.response_text[:120]}"
            )
            if getattr(bm, "reasoning", None):
                recent_activity_lines.append(f"  (my reasoning at the time: {bm.reasoning[:100]})")
            if getattr(bm, "current_posture", None):
                recent_activity_lines.append(f"  (my posture after: {bm.current_posture})")
    recent_bot_activity = "\n".join(recent_activity_lines) if recent_activity_lines else ""

    # Assemble the exact context the decision model receives (mirrors scheduler,
    # including the recent activity reconstruction so the 3-question qualitative
    # prompt has the memory blocks it references).
    target_block = build_target_message_block(enriched, [target_message_id] if target_message_id else None)
    whoami_lines = [
        "=== WHO I AM (my character) ===",
        getattr(config.persona, "identity", "") or "",
    ]
    if getattr(config.persona, "core_beliefs", None):
        whoami_lines.append("Core beliefs: " + " | ".join(config.persona.core_beliefs))
    whoami_lines.append("How I talk: " + (getattr(config.persona, "speaking_style", "") or ""))
    precomp_lines = f"=== PRE-COMPUTED SIGNALS ===\n{signals_block}\ncurrent_posture={posture}"
    full_ctx_for_decision = "\n".join([target_block, *whoami_lines, precomp_lines]).strip()
    if recent_bot_activity:
        full_ctx_for_decision += f"\n\n=== MY RECENT ACTIVITY AS ME ===\n{recent_bot_activity}"

    context = ContextBundle(
        context=full_ctx_for_decision,
        candidate_user_ids=get_candidate_user_ids(enriched),
        relationship_profiles=[],
        avg_feedback_score=avg_fb,
    )

    result.steps.append(StepResult(
        name="context_building",
        duration_ms=int((time.perf_counter() - t1) * 1000),
        data={
            "context_length": len(context.context),
            "candidate_users": context.candidate_user_ids,
            "target_message_id": target_message_id,
            "posture": posture,
            "precomputed_block": signals_block,
            "overrides_applied": applied_overrides,
            "context_preview": _make_context_preview(context.context),
            "includes_my_recent_activity_as_me": bool(recent_bot_activity),
            "recent_activity_length": len(recent_bot_activity) if recent_bot_activity else 0,
            "recent_activity_excerpt": recent_bot_activity[:250] if recent_bot_activity else "",
        },
    ))

    # Also surface a dedicated precomp step so UI can highlight the slim activity numbers block
    # the (qualitative) decision model actually receives.
    result.steps.append(StepResult(
        name="precomputed_signals",
        duration_ms=0,
        data={
            "block": signals_block,
            "posture": posture,
            "time_since": ts_val,
            "overrides": applied_overrides,
        },
    ))

    # --- Local-only mode: cloud brain (OpenRouter) disabled ---
    # Mirrors conversation_engine/scheduler.py::_execute_local_only. No perception/
    # decision LLM calls: the timing classifier already approved this cycle, so we
    # mark should_respond=True and let the voice model write the reply.
    if not getattr(config, "cloud_brain_enabled", True):
        raw_context = context.context
        target_lo = None
        for m in reversed(enriched):
            if (m.cleaned_text or m.text or "").strip():
                target_lo = m
                break
        # Confidence above ai.min_confidence_to_send (timing classifier already approved).
        local_conf = max(0.7, float(config.ai.min_confidence_to_send) + 0.05)
        decision = ResponseDecision(
            should_respond=True,
            confidence=local_conf,
            reply_to_message_id=getattr(target_lo, "message_id", None),
            reply_to_user_id=getattr(target_lo, "sender_id", None),
            target_message_id=getattr(target_lo, "message_id", None),
            reasoning="local_only_mode: timing classifier approved, voice model writing reply",
        )
        result.steps.append(StepResult(
            name="decision",
            duration_ms=0,
            data={**decision.model_dump(), "mode": "local_only (cloud_brain disabled)", "tokens_used": 0},
        ))
        result.decision = decision.model_dump()

        style_rewriter = LocalStyleRewriter(config)
        if style_rewriter.enabled:
            t1 = time.perf_counter()
            try:
                voiced = await style_rewriter.generate_voice(context=raw_context or "")
                voiced_text = (voiced or "").strip()
                result.steps.append(StepResult(
                    name="voice_generate",
                    duration_ms=int((time.perf_counter() - t1) * 1000),
                    data={
                        "enabled": True,
                        "mode": "standalone",
                        "voiced_text": voiced_text[:200],
                        "used": bool(voiced_text),
                    },
                ))
                if voiced_text:
                    decision.response_text = voiced_text
                    result.decision["response_text"] = voiced_text
                else:
                    decision.should_respond = False
            except Exception as exc:
                result.steps.append(StepResult(
                    name="voice_generate",
                    duration_ms=int((time.perf_counter() - t1) * 1000),
                    error=str(exc),
                ))
                decision.should_respond = False
        else:
            result.steps.append(StepResult(
                name="voice_generate",
                duration_ms=0,
                data={"enabled": False, "skipped": "style_rewriter disabled"},
            ))
            decision.should_respond = False

        # Same validators production runs.
        recent_bot_texts = [
            (m or {}).get("response_text") for m in (previous_bot_memories or [])
            if (m or {}).get("response_text")
        ]
        ok, reason = validate(decision, config, recent_bot_texts=recent_bot_texts)
        result.steps.append(StepResult(
            name="validation",
            duration_ms=0,
            data={"ok": ok, "reason": reason if not ok else "passed",
                  "final_text": decision.response_text if ok else None},
        ))
        result.decision["validated"] = ok
        result.decision["validation_reason"] = reason
        if ok and decision.response_text:
            result.response_text = decision.response_text
        result.total_duration_ms = int((time.perf_counter() - t0) * 1000)
        return result

    # --- Step 6: AI Perception ---
    if use_real_ai and config.xai_api_key:
        ai_client = GrokAiClient(config)
    else:
        ai_client = FakeAiClient()

    t1 = time.perf_counter()
    try:
        summary_prompt, summary_system = build_context_summary_prompt(
            context, config,
            high_level_enriched=high_level_enriched,
            recent_enriched=recent_enriched,
        )
        request1 = await ai_client.call_perception_model(summary_prompt, summary_system)
        context_summary = parse_context_summary(request1.text)
        result.steps.append(StepResult(
            name="perception",
            duration_ms=int((time.perf_counter() - t1) * 1000),
            data={
                "relevant_context": context_summary.relevant_context,
                "summary": context_summary.summary,
                "compressed_relevant_context": context_summary.compressed_relevant_context,
                "high_level_included": context_summary.high_level_included,
                "direct_mention_or_continuation": context_summary.direct_mention_or_continuation,
                "reasoning": context_summary.reasoning,
                "tokens_used": request1.tokens_used,
                "raw_response": request1.text[:500],
            },
        ))
        summary_body = context_summary.compressed_relevant_context or context_summary.summary
        if summary_body:
            header = "RELEVANT CONVERSATION CONTEXT" if context_summary.compressed_relevant_context else "PERCEPTION SUMMARY"
            enriched_ctx = f"{context.context}\n\n=== {header} ===\n{summary_body}"
            context = ContextBundle(
                context=enriched_ctx,
                candidate_user_ids=context.candidate_user_ids,
                relationship_profiles=context.relationship_profiles,
                avg_feedback_score=context.avg_feedback_score,
            )
    except Exception as exc:
        result.steps.append(StepResult(
            name="perception",
            duration_ms=int((time.perf_counter() - t1) * 1000),
            error=str(exc),
        ))

    # --- Step 7: AI Decision (the model sees the overridden precomp + posture) ---
    t1 = time.perf_counter()
    decision = None
    raw_context = context.context
    try:
        decision_prompt, decision_system = build_response_decision_prompt(context, "", config)
        request2 = await ai_client.call_decision_model(decision_prompt, decision_system)
        decision = parse_response_decision(request2.text)
        decision_data = decision.model_dump()
        result.steps.append(StepResult(
            name="decision",
            duration_ms=int((time.perf_counter() - t1) * 1000),
            data={
                **decision_data,
                "tokens_used": request2.tokens_used,
                "raw_response": request2.text[:500],
            },
        ))
        result.decision = decision_data
    except Exception as exc:
        result.steps.append(StepResult(
            name="decision",
            duration_ms=int((time.perf_counter() - t1) * 1000),
            error=str(exc),
        ))
        if hasattr(ai_client, "close"):
            await ai_client.close()
        result.total_duration_ms = int((time.perf_counter() - t0) * 1000)
        return result

    # --- Step 8: Local voice / phrasing (matches scheduler.py::_execute_llm) ---
    # voice_mode=standalone (production default): the voice model writes the reply from
    # raw context. voice_mode=phrase (legacy): smart model emits a plan, local renders it.
    style_rewriter = LocalStyleRewriter(config)
    voice_mode = getattr(config, "voice_mode", "standalone")
    if decision.should_respond and style_rewriter.enabled:
        t1 = time.perf_counter()
        if voice_mode == "standalone":
            try:
                voiced = await style_rewriter.generate_voice(context=raw_context or "")
                voiced_text = (voiced or "").strip()
                result.steps.append(StepResult(
                    name="style_rewriter",
                    duration_ms=int((time.perf_counter() - t1) * 1000),
                    data={
                        "enabled": True,
                        "mode": "standalone",
                        "original_text": (decision.response_text or "")[:200],
                        "voiced_text": voiced_text[:200],
                        "used": bool(voiced_text),
                    },
                ))
                if voiced_text:
                    decision.response_text = voiced_text
                    result.decision["response_text"] = voiced_text
                    result.decision["style_rewriter_applied"] = True
            except Exception as exc:
                result.steps.append(StepResult(
                    name="style_rewriter",
                    duration_ms=int((time.perf_counter() - t1) * 1000),
                    error=str(exc),
                ))
        else:
            plan_signal = (decision.plan or decision.reasoning or "").strip()
            if plan_signal:
                try:
                    phrased = await style_rewriter.phrase(
                        context=raw_context or "",
                        plan=plan_signal,
                        target_message="",
                        tone=decision.tone_calibration or "",
                    )
                    phrased_text = (phrased or "").strip()
                    result.steps.append(StepResult(
                        name="style_rewriter",
                        duration_ms=int((time.perf_counter() - t1) * 1000),
                        data={
                            "enabled": True,
                            "mode": "phrase",
                            "plan": plan_signal[:200],
                            "tone": decision.tone_calibration or "",
                            "original_text": (decision.response_text or "")[:200],
                            "phrased_text": phrased_text[:200],
                            "used_phrased": bool(phrased_text),
                        },
                    ))
                    if phrased_text:
                        decision.response_text = phrased_text
                        result.decision["response_text"] = phrased_text
                        result.decision["style_rewriter_applied"] = True
                except Exception as exc:
                    result.steps.append(StepResult(
                        name="style_rewriter",
                        duration_ms=int((time.perf_counter() - t1) * 1000),
                        error=str(exc),
                    ))
            else:
                result.steps.append(StepResult(
                    name="style_rewriter",
                    duration_ms=0,
                    data={"enabled": True, "mode": "phrase", "skipped": "no plan signal from decision"},
                ))
    else:
        result.steps.append(StepResult(
            name="style_rewriter",
            duration_ms=0,
            data={
                "enabled": style_rewriter.enabled,
                "skipped": "not enabled" if not style_rewriter.enabled else "decision is silent",
            },
        ))

    # --- Step 9: Validation ---
    t1 = time.perf_counter()
    recent_bot_texts = [
        bm.response_text
        for bm in recent_bot_mem_for_activity
        if getattr(bm, "response_text", None)
    ]
    ok, reason = validate(decision, config, recent_bot_texts=recent_bot_texts)
    result.steps.append(StepResult(
        name="validation",
        duration_ms=int((time.perf_counter() - t1) * 1000),
        data={"passed": ok, "reason": reason},
    ))

    if ok and decision.response_text:
        result.response_text = decision.response_text

    if hasattr(ai_client, "close"):
        await ai_client.close()

    result.total_duration_ms = int((time.perf_counter() - t0) * 1000)
    return result
