from __future__ import annotations

import asyncio
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
from conversation_engine.context_builder import build_context, build_request2_constraints
from conversation_engine.engagement_gate import compute_gate_score
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
from conversation_engine.sender import TelegramSender
from conversation_engine.validators import validate
from core.logging import get_logger, setup_logging
from storage.database import async_session_factory, dispose_engine

log = get_logger(__name__)


class ConversationScheduler:
    def __init__(self, config: EngineConfig, ai_client, sender: TelegramSender, feedback_loop: FeedbackLoop):
        self.config = config
        self.ai_client = ai_client
        self.sender = sender
        self.feedback_loop = feedback_loop
        self._shutdown = asyncio.Event()

    def shutdown(self) -> None:
        self._shutdown.set()
        self.feedback_loop.shutdown()

    async def run(self) -> None:
        queue: asyncio.Queue[int] = asyncio.Queue()
        for chat_id in self.config.active_chat_ids:
            await queue.put(chat_id)
        workers = [
            asyncio.create_task(self._worker(queue))
            for _ in range(max(1, self.config.scheduler.worker_pool_size))
        ]
        try:
            await self._shutdown.wait()
        finally:
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

    async def _worker(self, queue: asyncio.Queue[int]) -> None:
        while not self._shutdown.is_set():
            chat_id = await queue.get()
            interval = self.config.scheduler.initial_interval_seconds
            try:
                while not self._shutdown.is_set():
                    interval = await self._run_cycle(chat_id, interval)
                    await asyncio.sleep(interval)
            finally:
                queue.task_done()

    async def _run_cycle(self, chat_id: int, previous_interval: int) -> int:
        raw_context: str | None = None
        async with async_session_factory() as session:
            async with session.begin():
                memory = ConversationMemoryManager(session)
                if await memory.is_circuit_paused(chat_id):
                    return self.config.scheduler.max_interval_seconds

                try:
                    latest_decision = await memory.get_latest_ai_decision(chat_id)
                    snapshot_before = latest_decision.snapshot_message_id if latest_decision else None
                    new_message_count = await memory.count_messages_after_snapshot(chat_id, snapshot_before)
                    if new_message_count < self.config.scheduler.new_message_threshold:
                        return min(
                            self.config.scheduler.max_interval_seconds,
                            int(previous_interval * self.config.scheduler.backoff_multiplier),
                        )

                    await seed_persona_core(memory, self.config)
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

                    messages = await memory.get_recent_messages(chat_id, limit=150)
                    enriched = enrich_messages(messages, self.config.prompt)
                    brief = build_brief(enriched)
                    gate = await compute_gate_score(chat_id, enriched, brief, memory, self.config)
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
                            reasoning="engagement gate blocked",
                            gate_score=gate.gate_score,
                            gate_factors=gate.gate_factors,
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
                    perception_prompt = (
                        f"{context.context}\n\n"
                        "Decide whether there is a worthwhile opening to respond. "
                        "Return JSON with should_respond, confidence, reasoning, entry_points, topic."
                    )
                    request1 = await self.ai_client.call_perception_model(perception_prompt)
                    perception = parse_perception(request1.text)
                    if not perception.should_respond:
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
                            gate_factors=gate.gate_factors,
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
                    )
                    decision_prompt = (
                        f"{context.context}\n\n{constraints}\n\n"
                        "Draft the response decision. Return JSON matching: "
                        "should_respond, confidence, response_text, reply_to_message_id, reply_to_user_id, "
                        "reasoning, tone_calibration, stances, persona_alignment_score, feedback_informed."
                    )
                    request2 = await self.ai_client.call_decision_model(decision_prompt)
                    decision = parse_response_decision(request2.text)
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
                        gate_factors=gate.gate_factors,
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


async def main() -> None:
    config = load_engine_config()
    setup_logging()
    load_embedder(config.persona_engine.embedding_model)
    ai_client = AnthropicAiClient(config) if config.anthropic_api_key else FakeAiClient()
    sender = TelegramSender(config)
    await sender.connect()
    feedback_loop = FeedbackLoop(config, ai_client)
    scheduler = ConversationScheduler(config, ai_client, sender, feedback_loop)

    loop = asyncio.get_running_loop()
    for sig_name in ("SIGINT", "SIGTERM"):
        loop.add_signal_handler(getattr(signal, sig_name), scheduler.shutdown)

    try:
        bot_user_id = await sender.get_bot_user_id()
        await run_bootstrap(config, ai_client, bot_user_id)
        await asyncio.gather(scheduler.run(), feedback_loop.run_observation_tasks())
    finally:
        await sender.close()
        await dispose_engine()


if __name__ == "__main__":
    asyncio.run(main())
