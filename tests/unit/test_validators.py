"""Offline, deterministic tests for the pre-send safety net.

conversation_engine.validators is the last gate before a generated reply hits
Telegram. It rejects empty/over-length/avoided/low-confidence/redundant replies,
enforces the cross-message emoji budget, flags assistant-register "AI tells", and
applies donor-voice surface shaping. These are pure functions; the donor-shaping
helpers take an injected random.Random so behavior is fully reproducible.

No network, no DB. Where randomness is involved, a seeded random.Random is used
and determinism is asserted.
"""

from __future__ import annotations

import dataclasses
import random

import pytest

from conversation_engine import validators as V
from conversation_engine.ai_client import ResponseDecision
from conversation_engine.config import PromptConfig, load_engine_config


@pytest.fixture
def config():
    # Defaults-only EngineConfig (no config.toml on the path). emoji_window=5,
    # min_confidence_to_send=0.6, avoid_users=[] under defaults.
    return load_engine_config("/nonexistent-config-for-tests.toml")


def _decision(**overrides) -> ResponseDecision:
    base = {
        "should_respond": True,
        "confidence": 0.9,
        "response_text": "honestly that take is wild lol",
    }
    base.update(overrides)
    return ResponseDecision(**base)


# ---------------------------------------------------------------------------
# validate(): gating reasons
# ---------------------------------------------------------------------------


def test_validate_happy_path(config):
    ok, reason = V.validate(_decision(), config, [])
    assert ok is True
    assert reason is None


def test_validate_rejects_should_not_respond(config):
    ok, reason = V.validate(_decision(should_respond=False), config, [])
    assert ok is False
    assert reason == "decision_should_not_respond"


def test_validate_rejects_low_confidence(config):
    ok, reason = V.validate(_decision(confidence=0.1), config, [])
    assert ok is False
    assert reason.startswith("low_confidence:")


def test_validate_rejects_empty_response(config):
    ok, reason = V.validate(_decision(response_text="   "), config, [])
    assert ok is False
    assert reason == "empty_response"


def test_validate_rejects_over_length(config):
    ok, reason = V.validate(_decision(response_text="x" * 4097), config, [])
    assert ok is False
    assert reason == "telegram_message_too_long"


def test_validate_rejects_avoided_user(config):
    cfg = dataclasses.replace(config, prompt=PromptConfig(avoid_users=[777]))
    ok, reason = V.validate(_decision(reply_to_user_id=777), cfg, [])
    assert ok is False
    assert reason == "avoided_user:777"


# ---------------------------------------------------------------------------
# validate(): redundancy suppression (exact / near-dupe / verbal tic)
# ---------------------------------------------------------------------------


def test_validate_rejects_exact_duplicate(config):
    # Normalization strips punctuation/case, so "Hello!" duplicates "hello".
    ok, reason = V.validate(_decision(response_text="Hello!"), config, ["hello"])
    assert ok is False
    assert reason == "duplicate_of_recent_response"


def test_validate_rejects_near_duplicate_shared_opener(config):
    # Same 4-word opening run (the bot's habit of starting replies identically)
    # with otherwise low token overlap and no tracked verbal tic, so this is
    # specifically the near-duplicate/formulaic-opener path.
    recent = ["honestly that whole situation went sideways for unrelated reasons entirely"]
    ok, reason = V.validate(
        _decision(response_text="honestly that whole situation is completely overblown and dumb"),
        config,
        recent,
    )
    assert ok is False
    assert reason == "too_similar_to_recent_response"


def test_validate_rejects_reused_verbal_tic(config):
    # "pick a lane" is a tracked tic; reusing it when a recent msg also used it
    # is blocked. Texts are otherwise dissimilar so this is the tic path, not
    # the near-dupe path.
    recent = ["maybe just pick a lane already and commit"]
    ok, reason = V.validate(
        _decision(response_text="nah you gotta pick a lane here"),
        config,
        recent,
    )
    assert ok is False
    assert reason == "reused_verbal_tic"


def test_validate_allows_distinct_response_against_recent(config):
    recent = ["totally unrelated chatter about lunch plans"]
    ok, reason = V.validate(
        _decision(response_text="the gas fees today are actually insane"),
        config,
        recent,
    )
    assert ok is True
    assert reason is None


# ---------------------------------------------------------------------------
# validate(): emoji-window budget (mutation of outgoing text)
# ---------------------------------------------------------------------------


def test_validate_strips_emoji_when_recent_message_had_emoji(config):
    decision = _decision(response_text="lmao yeah 😭")
    ok, reason = V.validate(decision, config, ["earlier reply 💀"])
    assert ok is True
    assert reason is None
    # The outgoing text is mutated in place: emoji removed.
    assert decision.response_text == "lmao yeah"


def test_validate_keeps_emoji_when_recent_messages_clean(config):
    decision = _decision(response_text="lmao yeah 😭")
    ok, _ = V.validate(decision, config, ["plain reply", "another plain one"])
    assert ok is True
    # Untouched when no recent emoji in the window.
    assert decision.response_text == "lmao yeah 😭"


def test_validate_emoji_only_message_dropped_after_strip(config):
    # An emoji-only reply becomes empty after stripping (recent had emoji) and
    # is rejected rather than sent blank.
    ok, reason = V.validate(_decision(response_text="😭😭"), config, ["prev 💀"])
    assert ok is False
    assert reason == "empty_after_emoji_strip"


def test_validate_emoji_window_zero_disables_stripping(config):
    cfg = dataclasses.replace(config, emoji_window=0)
    decision = _decision(response_text="hype 😭")
    ok, _ = V.validate(decision, cfg, ["prev 💀"])
    assert ok is True
    # With the window disabled the emoji survives even though recent had one.
    assert decision.response_text == "hype 😭"


# ---------------------------------------------------------------------------
# Emoji helpers (pure)
# ---------------------------------------------------------------------------


def test_count_and_strip_emojis():
    assert V.count_emojis("a 😭 b 💀 c") == 2
    assert V.strip_emojis("hey 😭 there") == "hey there"
    assert V.count_emojis("no emojis here") == 0


def test_enforce_emoji_budget_window_boundary():
    # window=2 means only the single most-recent message is inspected.
    # An emoji two messages back (outside the window) does NOT trigger stripping.
    assert V.enforce_emoji_budget("keep 😭", ["clean recent", "old 💀"], window=2) == "keep 😭"
    # The same emoji one message back IS inside the window -> stripped.
    assert V.enforce_emoji_budget("keep 😭", ["old 💀"], window=2) == "keep"


# ---------------------------------------------------------------------------
# AI-tell rejection (standalone helper; not invoked inside validate())
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Certainly, here is what you asked for", "certainly"),
        ("I'd be happy to help with that", "i'd be happy to"),
        ("As an AI, I cannot do that", "as an ai"),
        ("Let me delve into the details", "delve"),
        ("In conclusion, the answer is yes", "in conclusion"),
        ("Feel free to reach out anytime", "feel free to"),
        ("this — with two — em dashes reads as AI", "em-dash-heavy"),
    ],
)
def test_violates_ai_tell_flags_known_tells(text, expected):
    assert V.violates_ai_tell(text) == expected


def test_violates_ai_tell_returns_none_for_clean_text():
    assert V.violates_ai_tell("nah that rug was obvious from the chart") is None
    assert V.violates_ai_tell("") is None


def test_violates_ai_tell_returns_first_match_in_pattern_order():
    # "certainly" precedes "delve" in the banlist, so it is reported first even
    # though both appear.
    assert V.violates_ai_tell("Certainly, let me delve in") == "certainly"


def test_ai_tell_banlist_is_in_pattern_order():
    # The exported convenience banlist mirrors the internal pattern order.
    assert V.AI_TELL_BANLIST[0] == "certainly"
    assert "delve" in V.AI_TELL_BANLIST


# ---------------------------------------------------------------------------
# apply_donor_casing: determinism with a seeded random.Random
# ---------------------------------------------------------------------------


def test_apply_donor_casing_is_deterministic_with_same_seed():
    texts = ["HELLO World", "MiXeD CaSe Here", "Another LINE"]
    out_a = [V.apply_donor_casing(t, random.Random(123)) for t in texts]
    out_b = [V.apply_donor_casing(t, random.Random(123)) for t in texts]
    assert out_a == out_b


def test_apply_donor_casing_sequence_deterministic_with_shared_rng():
    # Driving multiple calls off one shared seeded rng is reproducible too.
    rng1 = random.Random(7)
    rng2 = random.Random(7)
    texts = ["One TWO three", "FOUR five SIX", "Seven EIGHT"]
    assert [V.apply_donor_casing(t, rng1) for t in texts] == [
        V.apply_donor_casing(t, rng2) for t in texts
    ]


def test_apply_donor_casing_always_lowercases_at_rate_one():
    # rate=1.0 forces the lowercase branch deterministically (no seed needed).
    assert (
        V.apply_donor_casing("LOUD Message Here", random.Random(0), lowercase_rate=1.0)
        == "loud message here"
    )


def test_apply_donor_casing_never_lowercases_at_rate_zero():
    # rate=0.0 forces the passthrough branch: text returned verbatim.
    assert (
        V.apply_donor_casing("Keep ME Exact", random.Random(0), lowercase_rate=0.0)
        == "Keep ME Exact"
    )


def test_apply_donor_casing_preserves_handles_and_urls_when_lowercasing():
    # Even on the lowercase branch, @handles and URLs keep their literal casing
    # (they are spliced back in verbatim); the rest is lowercased.
    out = V.apply_donor_casing(
        "CHECK https://Example.com/AbC and @CoolHandle NOW",
        random.Random(0),
        lowercase_rate=1.0,
    )
    assert "https://Example.com/AbC" in out
    assert "@CoolHandle" in out
    assert out.startswith("check ")
    assert out.endswith(" now")


def test_apply_donor_casing_empty_text_passthrough():
    assert V.apply_donor_casing("", random.Random(0)) == ""


# ---------------------------------------------------------------------------
# Redundancy helpers (pure) -- direct coverage of the dedup primitives
# ---------------------------------------------------------------------------


def test_is_duplicate_response_normalizes_punctuation_and_case():
    assert V.is_duplicate_response("Same Thing!!!", ["same thing"]) is True
    assert V.is_duplicate_response("different entirely", ["same thing"]) is False
    # An empty/normalized-to-nothing text is never a duplicate.
    assert V.is_duplicate_response("   ", ["whatever"]) is False


def test_is_similar_response_high_token_overlap():
    # Reworded but heavy overlap -> caught by the jaccard threshold.
    assert (
        V.is_similar_response(
            "the market is absolutely cooked today",
            ["market is absolutely cooked today honestly"],
        )
        is True
    )


def test_is_similar_response_distinct_texts_not_similar():
    assert (
        V.is_similar_response(
            "gm everyone hope the day treats you well",
            ["the contract just got verified on etherscan"],
        )
        is False
    )


def test_reuses_recent_tic_only_when_shared():
    assert V.reuses_recent_tic("famous last words honestly", ["famous last words"]) is True
    # A tic in the new text but absent from recent is allowed (no reuse).
    assert V.reuses_recent_tic("famous last words", ["totally unrelated"]) is False
    # No tic at all -> never a reuse.
    assert V.reuses_recent_tic("just a normal line", ["another normal line"]) is False


# ---------------------------------------------------------------------------
# enforce_ack_dedup: per-day trivial-ack cap
# ---------------------------------------------------------------------------


def test_enforce_ack_dedup_blocks_at_threshold():
    # "lol" already sent twice today -> blocked at the default max_per_day=2.
    assert V.enforce_ack_dedup("lol", ["lol", "LOL"]) is True


def test_enforce_ack_dedup_allows_below_threshold():
    assert V.enforce_ack_dedup("lol", ["lol"]) is False


def test_enforce_ack_dedup_empty_text_allowed():
    assert V.enforce_ack_dedup("   ", ["lol", "lol"]) is False
