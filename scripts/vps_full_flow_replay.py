#!/usr/bin/env python3
"""
VPS Full Flow Replay Test

Runs 100 real messages from DWCusers_Chat through the COMPLETE conversation engine flow
on the VPS (Grok for perception+decision + local fine-tuned style rewriter for phrasing).

- Loads real messages
- Uses eval chat_id to not pollute real data
- Two modes:
  1. Real-history pass: for each of 100, run full flow against the real prefix history (no bot insertions)
  2. Branching sim pass: progressively insert real msgs + when AI decides to speak, insert simulated bot reply into history
     so subsequent decisions see the AI's own words (true "how the convo would have gone")

Records detailed jsonl + summary md with per-step:
  input text, should_respond, confidence, plan, styled_response (if any), reasoning, gate_score, latencies, etc.

Then produces a convo map of the simulated branching turns.

Run inside the conversation-engine container on VPS after placing messages json.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# Ensure we run from source in container
sys.path.insert(0, "/app")

from conversation_engine.ai_client import (
    GrokAiClient,
    parse_context_summary,
    parse_response_decision,
)
from conversation_engine.config import EngineConfig, load_engine_config
from conversation_engine.context_builder import build_context
from conversation_engine.engagement_gate import compute_gate_score
from conversation_engine.enrichment import build_brief, enrich_messages
from conversation_engine.memory_manager import ConversationMemoryManager
from conversation_engine.persona_engine import (
    get_relevant_persona_vectors,
    seed_persona_core,
)
from conversation_engine.prompts import (
    build_context_summary_prompt,
    build_response_decision_prompt,
)
from conversation_engine.style_rewriter import LocalStyleRewriter
from conversation_engine.validators import validate
from core.logging import setup_logging
from storage.database import async_session_factory, dispose_engine
from sqlalchemy import text

EVAL_CHAT_ID = -999999999999  # synthetic, isolated
BOT_USER_ID = 8856082053      # from real bot_memory in production chat
MAX_HISTORY_FOR_CONTEXT = 40  # similar to scheduler recent limit


def load_messages(path: str | Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def parse_message_timestamp(m: dict, fallback: datetime | None = None) -> datetime:
    ts_raw = m.get("timestamp")
    if isinstance(ts_raw, str):
        try:
            return datetime.fromisoformat(ts_raw.replace(" ", "T").replace("+00", "+00:00"))
        except Exception:
            return fallback or datetime.now(timezone.utc)
    if isinstance(ts_raw, datetime):
        return ts_raw
    return fallback or datetime.now(timezone.utc)


def env_path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default)).expanduser()


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


async def cleanup_eval_data(session, chat_id: int) -> None:
    """Remove any prior eval run data for this synthetic chat."""
    tables = [
        "ai_decisions",
        "bot_memory",
        "bot_vector_memories",
        "chat_activity_patterns",
        "user_relationship_profiles",
        "conversation_summaries",
        "messages",
    ]
    for tbl in tables:
        try:
            await session.execute(
                text(f"DELETE FROM {tbl} WHERE chat_id = :cid"), {"cid": chat_id}
            )
            await session.commit()
        except Exception:
            await session.rollback()
            pass  # table may not have chat_id or partial


async def insert_messages(session, chat_id: int, msgs: list[dict]) -> None:
    """Insert a batch of messages for the eval chat. Use original message_id (safe because chat_id differs)."""
    if not msgs:
        return
    now = datetime.now(timezone.utc)
    for m in msgs:
        ts = parse_message_timestamp(m, now)
        v = {
            "chat_id": chat_id,
            "message_id": m["message_id"],
            "sender_id": m["sender_id"],
            "timestamp": ts,
            "message_type": "text",
            "text_raw": m.get("text_raw", ""),
            "text_cleaned": (m.get("text_raw") or "")[:2000],
            "reply_to_message_id": m.get("reply_to_message_id"),
            "is_deleted": False,
            "created_at": now,
            "updated_at": now,
        }
        await session.execute(
            text(
                "INSERT INTO messages (chat_id, message_id, sender_id, timestamp, message_type, "
                "text_raw, text_cleaned, reply_to_message_id, is_deleted, created_at, updated_at) "
                "VALUES (:chat_id, :message_id, :sender_id, :timestamp, :message_type, "
                ":text_raw, :text_cleaned, :reply_to_message_id, :is_deleted, :created_at, :updated_at) "
                "ON CONFLICT (chat_id, message_id) DO NOTHING"
            ),
            v,
        )
    await session.commit()


async def insert_sim_bot_message(
    session,
    chat_id: int,
    sim_message_id: int,
    text: str,
    reply_to_message_id: int | None,
    timestamp: datetime | None = None,
) -> int:
    """Insert a simulated bot utterance into messages so future context sees it."""
    ts = timestamp or datetime.now(timezone.utc)
    await session.execute(
        text(
            """
            INSERT INTO messages (chat_id, message_id, sender_id, timestamp, message_type,
                                  text_raw, text_cleaned, reply_to_message_id, is_deleted, created_at, updated_at)
            VALUES (:chat, :mid, :sender, :ts, 'text', :txt, :clean, :reply, false, :now, :now)
            ON CONFLICT (chat_id, message_id) DO NOTHING
            """
        ),
        {
            "chat": chat_id,
            "mid": sim_message_id,
            "sender": BOT_USER_ID,
            "ts": ts,
            "txt": text,
            "clean": text[:2000],
            "reply": reply_to_message_id,
            "now": ts,
        },
    )
    await session.commit()
    return sim_message_id


async def insert_bot_memory_for_sim(
    memory: ConversationMemoryManager,
    chat_id: int,
    sent_message_id: int,
    response_text: str,
    reply_to_message_id: int | None,
    reasoning: str | None,
) -> None:
    await memory.insert_bot_memory(
        chat_id=chat_id,
        sent_message_id=sent_message_id,
        response_text=response_text,
        reply_to_user_id=None,
        reply_to_message_id=reply_to_message_id,
        reasoning=reasoning or "",
        tone_calibration=None,
        brief_snapshot={},
        stances={},
        prompt_version="replay-eval-v1",
        cycle_snapshot_message_id=sent_message_id,
    )


async def run_decision_pass(
    chat_id: int,
    memory: ConversationMemoryManager,
    ai_client: GrokAiClient,
    style_rewriter: LocalStyleRewriter,
    config: EngineConfig,
    new_msg_count_hint: int = 1,
) -> dict[str, Any]:
    """
    Core of the complete flow, adapted from scheduler._run_cycle for replay.
    Returns rich dict with all interesting signals + final decision + styled text.
    Does NOT send, does NOT do full reflections (to keep test focused and cheap).
    """
    started = time.perf_counter()
    result: dict[str, Any] = {"started_at": datetime.now(timezone.utc).isoformat()}

    messages = await memory.get_recent_messages(chat_id, limit=MAX_HISTORY_FOR_CONTEXT)
    if not messages:
        return {"error": "no messages", "should_respond": False}

    enriched = enrich_messages(messages, config.prompt)
    brief = build_brief(enriched)

    # Gate (group mode)
    gate = await compute_gate_score(chat_id, enriched, brief, memory, config)

    outcome_score_24h = await memory.get_avg_feedback_score(chat_id, window_hours=24)
    visible_numeric = {
        "tension_level": brief.tension_level,
        "outcome_score_24h": outcome_score_24h,
    }
    result["gate"] = {
        "should_proceed": gate.should_proceed,
        "gate_score": gate.gate_score,
        "gate_factors": visible_numeric,
    }

    if not gate.should_proceed:
        result["should_respond"] = False
        result["reason"] = f"gate_blocked:{gate.gate_factors.get('blocked', 'unknown')}"
        return result

    # Persona + context (light bootstrap if needed)
    await seed_persona_core(memory, config)
    persona_memories, latest_reflection = await get_relevant_persona_vectors(
        chat_id, " ".join(m.text_cleaned or m.text_raw or "" for m in messages[-10:])[:2000],
        memory, top_k=config.ai.persona_top_k,
    )
    current_persona = await memory.get_persona_core()

    recent_bot_mem = await memory.get_recent_bot_memory(chat_id, limit=6)
    recent_activity_lines = []
    for bm in recent_bot_mem:
        if bm.response_text:
            recent_activity_lines.append(f"I said: {bm.response_text[:100]}")
    recent_bot_activity = "\n".join(recent_activity_lines)

    context = await build_context(
        chat_id,
        enriched,
        brief,
        gate,
        memory,
        persona_memories,
        latest_reflection,
        current_persona,
        token_budget=config.ai.total_context_token_budget,
        recent_bot_activity=recent_bot_activity,
    )
    raw_context = context.context

    # Perception
    summary_prompt, summary_system = build_context_summary_prompt(context, config)
    req1 = await ai_client.call_perception_model(summary_prompt, summary_system)
    ctx_summary = parse_context_summary(req1.text)
    result["perception"] = {
        "latency_ms": req1.latency_ms,
        "tokens": req1.tokens_used,
        "relevant": ctx_summary.relevant_context,
        "summary": ctx_summary.summary[:300] if ctx_summary.summary else "",
    }

    # Decision context + posture
    posture_signals = []
    if brief and brief.tension_level is not None:
        posture_signals.append(f"tension in room: {brief.tension_level:.1f}")
    if outcome_score_24h is not None:
        posture_signals.append(f"my recent outcomes: {outcome_score_24h:.2f}")
    posture_signals.append(f"new messages since last: {new_msg_count_hint}")
    posture_block = " | ".join(posture_signals)

    decision_context = context
    if posture_block:
        enriched_dec = f"{context.context}\n\n=== MY CURRENT ENGAGEMENT SIGNALS ===\n{posture_block}"
        decision_context = type(context)(
            context=enriched_dec,
            candidate_user_ids=context.candidate_user_ids,
            relationship_profiles=context.relationship_profiles,
            avg_feedback_score=context.avg_feedback_score,
        )

    decision_prompt, decision_system = build_response_decision_prompt(decision_context, "", config)
    req2 = await ai_client.call_decision_model(decision_prompt, decision_system)
    decision = parse_response_decision(req2.text)

    result["decision_raw"] = {
        "latency_ms": req2.latency_ms,
        "tokens": req2.tokens_used,
        "should_respond": decision.should_respond,
        "confidence": decision.confidence,
        "plan": decision.plan,
        "reasoning": decision.reasoning,
        "tone": decision.tone_calibration,
        "reply_to": decision.reply_to_message_id,
    }

    # Style rewrite (the local model part of "complete flow")
    styled = None
    style_latency = None
    if decision.should_respond and style_rewriter.enabled:
        plan_signal = (decision.plan or decision.reasoning or "").strip()
        if plan_signal:
            t0 = time.perf_counter()
            phrased = await style_rewriter.phrase(
                context=raw_context or "",
                plan=plan_signal,
                target_message="",
                tone=decision.tone_calibration or "",
            )
            style_latency = (time.perf_counter() - t0) * 1000
            if phrased and phrased.strip():
                styled = phrased.strip()
                decision.response_text = styled

    result["style"] = {
        "applied": bool(styled),
        "latency_ms": int(style_latency) if style_latency else None,
        "response_text": styled or decision.response_text,
    }

    ok, reason = validate(decision, config)

    result["final"] = {
        "ok": ok,
        "should_respond": ok and decision.should_respond,
        "confidence": decision.confidence,
        "response_text": decision.response_text,
        "reply_to_message_id": decision.reply_to_message_id,
        "reasoning": (decision.reasoning or "") + (f" | {reason}" if not ok else ""),
        "plan": decision.plan,
    }
    result["total_time_s"] = round(time.perf_counter() - started, 2)
    result["gate_score"] = gate.gate_score
    return result


def summarize_results(res_list: list[dict], name: str) -> dict:
    responded = [r for r in res_list if r.get("final", {}).get("should_respond")]
    styles = [r for r in res_list if r.get("style", {}).get("applied")]
    avg_conf = sum(r.get("final", {}).get("confidence", 0) for r in res_list) / max(1, len(res_list))
    return {
        "name": name,
        "total": len(res_list),
        "responded": len(responded),
        "respond_rate": round(len(responded) / max(1, len(res_list)), 3),
        "style_rewrites": len(styles),
        "avg_confidence": round(avg_conf, 3),
        "example_responses": [
            {
                "input": r["input_text"][:80],
                "out": r.get("style", {}).get("response_text") or r.get("final", {}).get("response_text"),
                "reason": (r.get("final", {}).get("reasoning") or "")[:120],
            }
            for r in responded[:5]
        ],
    }


async def main():
    setup_logging()
    config = load_engine_config()
    # Force real client (we have the key in env)
    ai_client = GrokAiClient(config)
    style_rewriter = LocalStyleRewriter(config)

    try:
        msgs_path = env_path("REPLAY_MESSAGES_PATH", "/tmp/eval_messages_100.json")
        if not msgs_path.exists():
            # fallback for host/VPS runs
            msgs_path = env_path("REPLAY_MESSAGES_FALLBACK_PATH", "/home/USER/Research/eval_messages_100.json")
        msgs = load_messages(msgs_path)
        print(f"Loaded {len(msgs)} messages for replay from {msgs_path}. chat={EVAL_CHAT_ID}")
        limit = env_int("REPLAY_LIMIT", 100)
        msgs = msgs[: min(limit, len(msgs))]
        print(f"Using first {len(msgs)} messages for this replay run")

        results: list[dict] = []
        sim_results: list[dict] = []  # for branching

        # Use separate sessions / commits for setup vs long loops to avoid long tx / closed tx errors
        async with async_session_factory() as session:
            memory = ConversationMemoryManager(session)
            await cleanup_eval_data(session, EVAL_CHAT_ID)
            await memory.initialize_activity_patterns(EVAL_CHAT_ID)
            await seed_persona_core(memory, config)
            await session.commit()

        out_dir = env_path("REPLAY_OUT_DIR", "/tmp")
        out_dir.mkdir(parents=True, exist_ok=True)

        # --- PASS 1: Real history, decide for each incoming in sequence (no self-insertion)
        print(f"=== PASS 1: real-history decisions ({len(msgs)} steps) ===")
        async with async_session_factory() as session:
            memory = ConversationMemoryManager(session)
            for i, m in enumerate(msgs):
                print(f"  [pass1] step {i+1}/{len(msgs)} input={m['message_id']} {m['text_raw'][:30]!r} ...", flush=True)
                await insert_messages(session, EVAL_CHAT_ID, [m])
                try:
                    pass_result = await run_decision_pass(
                        EVAL_CHAT_ID, memory, ai_client, style_rewriter, config, new_msg_count_hint=1
                    )
                except Exception as e:
                    print(f"    ERROR in step {i}: {e}")
                    pass_result = {"idx": i, "error": str(e), "should_respond": False}
                pass_result["idx"] = i
                pass_result["input_message_id"] = m["message_id"]
                pass_result["input_text"] = m["text_raw"][:200]
                pass_result["input_sender"] = m["sender_id"]
                results.append(pass_result)
                # flush partial
                (out_dir / "replay_pass1_partial.jsonl").write_text("\n".join(json.dumps(r, default=str) for r in results))
                if (i + 1) % 5 == 0:
                    print(f"  processed {i+1}/{len(msgs)} ... last_should={pass_result.get('final', {}).get('should_respond')}")

        # --- PASS 2: Branching simulation (insert bot replies when it speaks)
        print("\n=== PASS 2: branching sim (insert AI replies into history) ===")
        async with async_session_factory() as session:
            memory = ConversationMemoryManager(session)
            await cleanup_eval_data(session, EVAL_CHAT_ID)
            await memory.initialize_activity_patterns(EVAL_CHAT_ID)
            await seed_persona_core(memory, config)
            await session.commit()

        sim_bot_msg_id = 900000000  # high synthetic ids for inserted bot msgs
        inserted_bot_count = 0

        for i, m in enumerate(msgs):
            print(f"  [sim] step {i+1}/{len(msgs)} input={m['message_id']} {m['text_raw'][:30]!r} ...", flush=True)
            async with async_session_factory() as session:
                memory = ConversationMemoryManager(session)
                # Insert this real message progressively (new session each time for isolation)
                await insert_messages(session, EVAL_CHAT_ID, [m])

                try:
                    pass_result = await run_decision_pass(
                        EVAL_CHAT_ID, memory, ai_client, style_rewriter, config, new_msg_count_hint=1
                    )
                except Exception as e:
                    print(f"    SIM ERROR in step {i}: {e}")
                    pass_result = {"idx": i, "error": str(e), "should_respond": False}
                pass_result["idx"] = i
                pass_result["input_message_id"] = m["message_id"]
                pass_result["input_text"] = m["text_raw"][:200]
                pass_result["input_sender"] = m["sender_id"]

                final = pass_result.get("final", {})
                if final.get("should_respond") and final.get("response_text"):
                    sim_mid = sim_bot_msg_id
                    sim_bot_msg_id += 1
                    reply_to = final.get("reply_to_message_id") or m["message_id"]
                    sim_ts = parse_message_timestamp(m) + timedelta(milliseconds=500 + inserted_bot_count)
                    await insert_sim_bot_message(
                        session,
                        EVAL_CHAT_ID,
                        sim_mid,
                        final["response_text"],
                        reply_to,
                        timestamp=sim_ts,
                    )
                    await insert_bot_memory_for_sim(
                        memory,
                        EVAL_CHAT_ID,
                        sim_mid,
                        final["response_text"],
                        reply_to,
                        final.get("reasoning"),
                    )
                    await session.commit()
                    inserted_bot_count += 1
                    pass_result["sim_inserted_as"] = sim_mid

                sim_results.append(pass_result)
                if (i + 1) % 10 == 0:
                    print(f"  sim processed {i+1}/{len(msgs)} ... bots_inserted_so_far={inserted_bot_count}")

        # Write artifacts
        (out_dir / "replay_results_real_history.jsonl").write_text(
            "\n".join(json.dumps(r, default=str) for r in results)
        )
        (out_dir / "replay_results_branching.jsonl").write_text(
            "\n".join(json.dumps(r, default=str) for r in sim_results)
        )

        # Also to exports for persistence
        exports_dir = env_path("REPLAY_EXPORTS_DIR", "/home/USER/Research/exports")
        exports_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        label = len(msgs)
        (exports_dir / f"replay_{label}_real_history_{ts}.jsonl").write_text(
            "\n".join(json.dumps(r, default=str) for r in results)
        )
        (exports_dir / f"replay_{label}_branching_{ts}.jsonl").write_text(
            "\n".join(json.dumps(r, default=str) for r in sim_results)
        )

        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "eval_chat": EVAL_CHAT_ID,
            "msg_range": [msgs[0]["message_id"], msgs[-1]["message_id"]] if msgs else [],
            "real_history": summarize_results(results, "real_history"),
            "branching": summarize_results(sim_results, "branching_sim"),
            "config": {
                "perception_model": config.ai.perception_model,
                "decision_model": config.ai.decision_model,
                "local_style_enabled": style_rewriter.enabled,
            },
        }
        (out_dir / "replay_summary.json").write_text(json.dumps(summary, indent=2, default=str))
        print("\n=== SUMMARY ===")
        print(json.dumps(summary, indent=2, default=str))

        # Minimal MD map for the branching convo
        md_lines = ["# Branching Convo Map (what the AI would have done)\n"]
        md_lines.append(f"Tested {len(sim_results)} real messages from DWCusers_Chat.\n")
        md_lines.append("When AI decided to speak in the sim, its reply was inserted so later steps saw it.\n\n")
        for r in sim_results:
            if r.get("final", {}).get("should_respond"):
                inp = r["input_text"][:70].replace("\n", " ")
                out = (r.get("style", {}).get("response_text") or r.get("final", {}).get("response_text") or "")[:80]
                md_lines.append(f"- On real: `{inp}` ... AI would reply: **{out}** (conf={r['final']['confidence']:.2f})")
                if r.get("sim_inserted_as"):
                    md_lines.append(f"  (inserted as sim msg {r['sim_inserted_as']})")
        (exports_dir / f"replay_branching_convo_map_{ts}.md").write_text("\n".join(md_lines))
        print(f"\nWrote artifacts to {exports_dir} and {out_dir}. Done.")
    finally:
        await ai_client.close()
        await dispose_engine()


if __name__ == "__main__":
    asyncio.run(main())
