"""enrich_messages is per-message: enriching a list then slicing equals enriching
the slice. The conversation cycle relies on this to take one VADER pass over the
high-level window and slice the recent/summary windows from it (instead of
re-querying + re-enriching subsets). If enrichment ever becomes context-dependent
this test fails, flagging that the _prepare_cycle dedup is no longer valid.
"""

from __future__ import annotations

from types import SimpleNamespace

from conversation_engine.config import PromptConfig
from conversation_engine.enrichment import enrich_messages


def _msg(message_id: int, text: str, sender_id: int = 1) -> SimpleNamespace:
    # enrich_messages duck-types on these attributes (it never touches the ORM).
    return SimpleNamespace(
        message_id=message_id,
        chat_id=-100,
        sender_id=sender_id,
        text_cleaned=text,
        text_raw=text,
        reply_to_message_id=None,
        timestamp=None,
    )


def test_slicing_enriched_equals_enriching_the_slice() -> None:
    texts = ["gm", "this is a scam ngl", "based deal on sol", "lol ok", "mid tbh", "vouch for him"]
    msgs = [_msg(i, t, sender_id=i % 3) for i, t in enumerate(texts)]
    cfg = PromptConfig(topics_of_interest=["sol", "deal"])

    full = enrich_messages(msgs, cfg)
    # Every suffix length must match: full[-k:] == enrich(msgs[-k:]).
    for k in range(1, len(msgs) + 1):
        assert full[-k:] == enrich_messages(msgs[-k:], cfg)


def test_enrichment_is_positionally_stable() -> None:
    """A message's enrichment does not depend on its neighbours -- the same Message
    enriched alone or inside a batch yields an equal EnrichedMessage."""
    cfg = PromptConfig(topics_of_interest=["sol"])
    target = _msg(99, "huge sol deal, totally legit", sender_id=7)
    alone = enrich_messages([target], cfg)[0]
    in_batch = enrich_messages([_msg(1, "noise"), target, _msg(2, "more noise")], cfg)[1]
    assert alone == in_batch
