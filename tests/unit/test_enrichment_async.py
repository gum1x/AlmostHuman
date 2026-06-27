"""enrich_messages_async must be behavior-identical to the sync enrich_messages.

The async variant only offloads the CPU-bound VADER work to a worker thread; it
must never change the enriched output (sentiment, topics, overlap, etc.).
"""

from __future__ import annotations

from types import SimpleNamespace

from conversation_engine.config import PromptConfig
from conversation_engine.enrichment import enrich_messages, enrich_messages_async


def _msg(message_id: int, text: str):
    return SimpleNamespace(
        message_id=message_id,
        chat_id=-100,
        sender_id=111,
        reply_to_message_id=None,
        text_cleaned=text,
        text_raw=text,
        timestamp=None,
    )


async def test_enrich_messages_async_matches_sync():
    prompt = PromptConfig(topics_of_interest=["crypto", "trading"])
    messages = [
        _msg(1, "this is a scam, total cope"),
        _msg(2, "based crypto trading legit vouch"),
        _msg(3, ""),
        _msg(4, "love this, great stuff"),
    ]

    sync_result = enrich_messages(messages, prompt)
    async_result = await enrich_messages_async(messages, prompt)

    assert async_result == sync_result
    # Sanity: at least one message carried a non-zero VADER/override sentiment,
    # so the equivalence check is actually exercising the offloaded work.
    assert any(m.sentiment_score != 0.0 for m in sync_result)
