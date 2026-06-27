"""The two AI backends both satisfy the AiClient Protocol, and the offline fake
is a genuine drop-in (implements call_raw + close, not just the two model calls).

This locks in the contract behind the engine's `ai_client` handoff: a regression
that drops a method from either client, or adds a method the fake doesn't mirror,
fails here instead of at runtime in a live cycle.
"""

from __future__ import annotations

from conversation_engine.ai_client import AiCallResult, AiClient, FakeAiClient, GrokAiClient
from conversation_engine.config import load_engine_config


def test_fake_client_satisfies_protocol() -> None:
    assert isinstance(FakeAiClient(), AiClient)


async def test_real_client_satisfies_protocol() -> None:
    client = GrokAiClient(load_engine_config())
    try:
        assert isinstance(client, AiClient)
    finally:
        await client.close()


async def test_fake_client_call_raw_is_a_drop_in() -> None:
    """call_raw exists on the fake (the word-generator path) and returns an empty
    AiCallResult so callers degrade gracefully instead of hitting AttributeError."""
    fake = FakeAiClient()
    result = await fake.call_raw([{"role": "user", "content": "hi"}], temperature=0.5)
    assert isinstance(result, AiCallResult)
    assert result.text == ""
    assert result.tokens_used == 0


async def test_fake_client_close_is_noop() -> None:
    # Present so FakeAiClient mirrors GrokAiClient's lifecycle; must not raise.
    await FakeAiClient().close()
