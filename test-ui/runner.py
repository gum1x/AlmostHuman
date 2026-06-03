"""
Workflow runner that executes the full conversation engine pipeline
on injected JSON chat data — no database or Telegram connection needed.

Mocks: memory manager, sender (we capture the response instead of sending).
Real: enrichment, gate scoring (simplified), context building, AI calls,
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
    select_target_message,
    format_vector_memories,
    build_response_context,
    compute_quantitative_signals,
    format_quantitative_signals,
)
from conversation_engine.engagement_gate import GateResult, get_candidate_user_ids
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

def _build_context_offline(
    chat_id: int,
    enriched: list[EnrichedMessage],
    brief: Brief,
    gate: GateResult,
    config: EngineConfig,
    target_message_id: int | None = None,
) -> ContextBundle:
    """Build context without async DB calls — uses only the enriched messages."""
    candidate_users = get_candidate_user_ids(enriched)
    entry_points = [target_message_id] if target_message_id else None
    target = select_target_message(enriched, entry_points)
    lines = [build_target_message_block(enriched, entry_points)]

    # Nearby window (simplified inline)
    if target:
        idx = next(
            (i for i, m in enumerate(enriched) if m.message_id == target.message_id),
            len(enriched) - 1,
        )
        start = max(0, idx - 2)
        end = min(len(enriched), idx + 3)
        thread_ids = {target.message_id}
        if target.reply_to_message_id is not None:
            thread_ids.add(target.reply_to_message_id)
        nearby = [
            f"{m.message_id} user_{m.sender_id} reply_to={m.reply_to_message_id or 'none'}: {(m.cleaned_text or m.text)[:180]}"
            for m in enriched[start:end]
            if m.message_id not in thread_ids
        ]
        if nearby:
            lines.extend(["nearby:", *nearby])

    # Persona identity from config
    lines.append("=== WHO I AM (my character) ===")
    lines.append(config.persona.identity)
    if config.persona.core_beliefs:
        lines.append("Core beliefs: " + " | ".join(config.persona.core_beliefs))
    lines.append("How I talk: " + config.persona.speaking_style)

    # Brief signals
    if brief.tension_level >= 0.5:
        lines.append(f"signals: tension={brief.tension_level:.2f}")

    # Pre-computed quantitative signals for the decision model
    quant_signals = compute_quantitative_signals(
        enriched_messages=enriched,
        brief=brief,
        bot_user_id=None,
        bot_sent_message_ids=set(),
        time_since_last_bot_msg_min=None,
        chat_velocity=None,
        responses_last_hour=0,
        avg_feedback_score=0.0,
    )
    lines.append("=== PRE-COMPUTED SIGNALS ===")
    lines.append(format_quantitative_signals(quant_signals))

    context_str = "\n".join(lines).strip()
    return ContextBundle(
        context=context_str,
        candidate_user_ids=candidate_users,
        relationship_profiles=[],
        avg_feedback_score=0.0,
    )


def _compute_gate_offline(enriched: list[EnrichedMessage], brief: Brief) -> GateResult:
    """Simplified gate that always proceeds but computes a score from the data."""
    sentiments = [m.sentiment_score for m in enriched[-20:]]
    avg_sent = sum(sentiments) / len(sentiments) if sentiments else 0.0
    tension = brief.tension_level
    score = max(0.0, min(1.0, 0.5 + avg_sent * 0.3 - tension * 0.2))
    return GateResult(
        gate_score=score,
        gate_factors={
            "velocity": 1.0,
            "emotional_trend": max(0.0, (avg_sent + 1.0) / 2.0),
            "fatigue": 1.0,
            "topic_alignment": max((m.topic_overlap_score for m in enriched), default=0.0),
            "tension": tension,
        },
        should_proceed=True,
    )


# ---------------------------------------------------------------------------
# Main pipeline runner
# ---------------------------------------------------------------------------

async def run_pipeline(
    messages_json: list[dict[str, Any]],
    config: EngineConfig | None = None,
    use_real_ai: bool = True,
    target_message_id: int | None = None,
) -> PipelineResult:
    """
    Run the full conversation engine pipeline on a list of JSON messages.

    target_message_id: if set, forces the pipeline to treat that specific
    message as the one to respond to (instead of defaulting to the latest).

    Returns a PipelineResult with step-by-step details and the final response.
    """
    t0 = time.perf_counter()
    config = config or load_engine_config()
    chat_id = messages_json[0].get("chat_id", -1001234567890) if messages_json else -1001234567890
    result = PipelineResult(chat_id=chat_id)

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

    # --- Step 3: Brief ---
    t1 = time.perf_counter()
    brief = build_brief(enriched)
    result.steps.append(StepResult(
        name="brief",
        duration_ms=int((time.perf_counter() - t1) * 1000),
        data=brief.as_dict(),
    ))

    # --- Step 4: Gate ---
    t1 = time.perf_counter()
    gate = _compute_gate_offline(enriched, brief)
    result.steps.append(StepResult(
        name="engagement_gate",
        duration_ms=int((time.perf_counter() - t1) * 1000),
        data={
            "gate_score": round(gate.gate_score, 3),
            "should_proceed": gate.should_proceed,
            "factors": {k: round(v, 3) if isinstance(v, float) else v for k, v in gate.gate_factors.items()},
        },
    ))

    if not gate.should_proceed:
        result.steps.append(StepResult(name="blocked", data={"reason": "gate_blocked"}))
        result.total_duration_ms = int((time.perf_counter() - t0) * 1000)
        return result

    # --- Step 5: Context building ---
    t1 = time.perf_counter()
    context = _build_context_offline(chat_id, enriched, brief, gate, config, target_message_id)
    result.steps.append(StepResult(
        name="context_building",
        duration_ms=int((time.perf_counter() - t1) * 1000),
        data={
            "context_length": len(context.context),
            "candidate_users": context.candidate_user_ids,
            "target_message_id": target_message_id,
            "context_preview": context.context[:500],
        },
    ))

    # --- Step 6: AI Perception ---
    if use_real_ai and config.xai_api_key:
        ai_client = GrokAiClient(config)
    else:
        ai_client = FakeAiClient()

    t1 = time.perf_counter()
    try:
        summary_prompt, summary_system = build_context_summary_prompt(context, config)
        request1 = await ai_client.call_perception_model(summary_prompt, summary_system)
        context_summary = parse_context_summary(request1.text)
        result.steps.append(StepResult(
            name="perception",
            duration_ms=int((time.perf_counter() - t1) * 1000),
            data={
                "relevant_context": context_summary.relevant_context,
                "summary": context_summary.summary,
                "reasoning": context_summary.reasoning,
                "tokens_used": request1.tokens_used,
                "raw_response": request1.text[:500],
            },
        ))
        # Append perception summary to context if relevant
        if context_summary.summary:
            enriched_ctx = f"{context.context}\n\n=== PERCEPTION SUMMARY ===\n{context_summary.summary}"
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

    # --- Step 7: AI Decision ---
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

    # --- Step 8: Local Style Rewriter (LoRA phrasing) ---
    style_rewriter = LocalStyleRewriter(config)
    if decision.should_respond and style_rewriter.enabled:
        t1 = time.perf_counter()
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
                        "plan": plan_signal[:200],
                        "tone": decision.tone_calibration or "",
                        "original_text": (decision.response_text or "")[:200],
                        "phrased_text": phrased_text[:200],
                        "used_phrased": bool(phrased_text),
                    },
                ))
                if phrased_text:
                    decision.response_text = phrased_text
                    # Update decision data with the new text
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
                data={"enabled": True, "skipped": "no plan signal from decision"},
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
    ok, reason = validate(decision, config)
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
