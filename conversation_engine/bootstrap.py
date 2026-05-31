from __future__ import annotations

from conversation_engine.ai_client import FakeAiClient, GrokAiClient
from conversation_engine.config import EngineConfig, load_engine_config
from conversation_engine.memory_manager import ConversationMemoryManager
from conversation_engine.persona_engine import load_embedder, run_self_reflection, seed_persona_core
from conversation_engine.sender import TelegramSender
from core.logging import get_logger, setup_logging
from storage.database import async_session_factory

log = get_logger(__name__)


def _summarize_messages(messages) -> str:
    if not messages:
        return ""
    lines = [f"user_{msg.sender_id}: {msg.text_cleaned or msg.text_raw or ''}" for msg in messages[-30:]]
    return "\n".join(lines)[:4000]


async def bootstrap_chat(
    chat_id: int,
    config: EngineConfig,
    ai_client,
    bot_user_id: int | None = None,
) -> int:
    async with async_session_factory() as session:
        async with session.begin():
            memory = ConversationMemoryManager(session)
            await seed_persona_core(memory, config)
            await memory.initialize_activity_patterns(chat_id)

            if await memory.count_summaries(chat_id) == 0:
                messages = await memory.get_recent_messages(chat_id, limit=500)
                if messages:
                    await memory.insert_conversation_summary(
                        chat_id=chat_id,
                        chunk_start_message_id=messages[0].message_id,
                        chunk_end_message_id=messages[-1].message_id,
                        summary=_summarize_messages(messages),
                        token_count=sum(len((msg.text_cleaned or msg.text_raw or "").split()) for msg in messages),
                    )

            backfilled = 0
            if bot_user_id is not None:
                backfilled = await memory.backfill_bot_memory_from_messages(
                    chat_id=chat_id,
                    bot_user_id=bot_user_id,
                    prompt_version=config.ai.prompt_version,
                )
            if backfilled >= 50:
                await run_self_reflection(
                    chat_id=chat_id,
                    memory=memory,
                    ai_client=ai_client,
                    config=config,
                    trigger="message_threshold",
                    messages_since_last=backfilled,
                )
            return backfilled


async def run_bootstrap(config: EngineConfig, ai_client, bot_user_id: int | None = None) -> None:
    load_embedder(config.persona_engine.embedding_model)
    for chat_id in config.active_chat_ids:
        backfilled = await bootstrap_chat(chat_id, config, ai_client, bot_user_id)
        await log.ainfo("conversation_bootstrap_chat_complete", chat_id=chat_id, backfilled=backfilled)


async def main() -> None:
    config = load_engine_config()
    setup_logging()
    load_embedder(config.persona_engine.embedding_model)
    ai_client = GrokAiClient(config) if config.xai_api_key else FakeAiClient()
    sender = TelegramSender(config)
    bot_user_id = None
    try:
        await sender.connect()
        bot_user_id = await sender.get_bot_user_id()
    finally:
        await sender.close()
    try:
        await run_bootstrap(config, ai_client, bot_user_id)
    finally:
        close = getattr(ai_client, "close", None)
        if close:
            await close()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
