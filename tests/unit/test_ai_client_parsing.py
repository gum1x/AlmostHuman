"""Offline, deterministic tests for the untrusted-LLM-output parsing boundary.

These cover the pure extractor/parser functions in conversation_engine.ai_client
that turn raw model text (which may be fenced, prose-wrapped, truncated, or
outright garbage) into typed decisions. This is a safety boundary: the rest of
the engine assumes these never crash unexpectedly and always yield either a
valid object, the documented safe default, or a well-typed raised error.

No network, no DB -- just text in, object/exception out.
"""

from __future__ import annotations

import json

import pytest

from conversation_engine.ai_client import (
    ContextSummary,
    ResponseDecision,
    _sanitize_response_text,
    extract_json_object,
    parse_context_summary,
    parse_response_decision,
)

# ---------------------------------------------------------------------------
# extract_json_object: the shared first stage every parser depends on
# ---------------------------------------------------------------------------


def test_extract_clean_json_object():
    assert extract_json_object('{"a": 1, "b": "two"}') == '{"a": 1, "b": "two"}'


def test_extract_json_inside_markdown_json_fence():
    # ```json ... ``` is the most common DeepSeek/Grok fencing; the language
    # label after the opening fence must be stripped along with the fence.
    text = '```json\n{"should_respond": true}\n```'
    assert extract_json_object(text) == '{"should_respond": true}'


def test_extract_json_inside_plain_markdown_fence():
    text = '```\n{"should_respond": true}\n```'
    assert extract_json_object(text) == '{"should_respond": true}'


def test_extract_json_with_leading_prose():
    # Reasoning leaked before the object: the first balanced { ... } is returned.
    text = 'Here is my reasoning and then the answer: {"a": 1} -- hope that helps'
    assert extract_json_object(text) == '{"a": 1}'


def test_extract_json_ignores_trailing_tail_after_object():
    text = '{"a": 1}\nSome trailing chain-of-thought that should be dropped.'
    assert extract_json_object(text) == '{"a": 1}'


def test_extract_json_does_not_split_on_brace_inside_string_value():
    # A '}' inside a string value must not be treated as the closing brace.
    text = '{"msg": "this } is fine", "n": 2}'
    extracted = extract_json_object(text)
    assert extracted == text
    # And it must still be valid, round-trippable JSON.
    assert json.loads(extracted) == {"msg": "this } is fine", "n": 2}


def test_extract_json_skips_unbalanced_prefix_then_finds_real_object():
    # A stray '{' that never closes is skipped; the next balanced object wins.
    text = 'noise {unclosed and then {"real": true}'
    assert extract_json_object(text) == '{"real": true}'


def test_extract_garbage_raises_value_error():
    with pytest.raises(ValueError, match="did not contain a JSON object"):
        extract_json_object("absolutely no json here at all")


def test_extract_truncated_object_raises_value_error():
    # An opening brace that never balances is treated as "no JSON object".
    with pytest.raises(ValueError, match="did not contain a JSON object"):
        extract_json_object('{"a": 1')


# ---------------------------------------------------------------------------
# parse_response_decision: the decision-model output contract
# ---------------------------------------------------------------------------


def test_parse_decision_should_respond_true_basic():
    text = '{"should_respond": true, "confidence": 0.7, "response_text": "yo"}'
    decision = parse_response_decision(text)
    assert isinstance(decision, ResponseDecision)
    assert decision.should_respond is True
    assert decision.confidence == 0.7
    assert decision.response_text == "yo"


def test_parse_decision_should_respond_false_shape_defaults():
    # The minimal "stay silent" shape: only should_respond present. Every
    # optional field must fall back to its typed default, not raise.
    decision = parse_response_decision('{"should_respond": false}')
    assert decision.should_respond is False
    assert decision.confidence == 0.0
    assert decision.response_text is None
    assert decision.plan == ""
    assert decision.reasoning == ""
    assert decision.stances == {}
    assert decision.intent_tag is None
    assert decision.updated_engagement_posture is None


def test_parse_decision_confidence_above_one_is_rescaled_to_unit_interval():
    # Models sometimes emit confidence on a 0-100 scale; >1 is divided by 100
    # and clamped to 1.0.
    decision = parse_response_decision(
        '{"should_respond": true, "confidence": 85, "response_text": "x"}'
    )
    assert decision.confidence == pytest.approx(0.85)


def test_parse_decision_confidence_huge_clamps_to_one():
    decision = parse_response_decision(
        '{"should_respond": true, "confidence": 9000, "response_text": "x"}'
    )
    assert decision.confidence == 1.0


def test_parse_decision_empty_response_text_becomes_none():
    decision = parse_response_decision('{"should_respond": true, "response_text": ""}')
    assert decision.response_text is None


def test_parse_decision_sanitizes_leaked_json_tail_in_response_text():
    # The exact broken-JSON shape seen in the wild: the response_text value
    # swallowed the next key. The leaked tail must be cut off.
    text = (
        '{"should_respond": true, "response_text": '
        "\"another one \\ud83d\\ude2d','reply_to_message_id':156,\"}"
    )
    decision = parse_response_decision(text)
    assert decision.response_text == "another one \U0001f62d"
    assert "reply_to_message_id" not in (decision.response_text or "")


def test_parse_decision_plan_falls_back_to_reasoning_when_absent():
    decision = parse_response_decision(
        '{"should_respond": true, "response_text": "x", "reasoning": "because vibes"}'
    )
    assert decision.plan == "because vibes"


def test_parse_decision_coerces_non_string_text_fields():
    # If the model emits non-strings for fields typed as str, they are coerced
    # rather than raising a pydantic validation error.
    text = (
        '{"should_respond": true, "response_text": "x", '
        '"semantic_risk": 5, "annoying_reason": null}'
    )
    decision = parse_response_decision(text)
    assert decision.semantic_risk == "5"
    assert decision.annoying_reason == ""


def test_parse_decision_intent_tag_normalized_when_valid():
    decision = parse_response_decision(
        '{"should_respond": true, "response_text": "x", "intent_tag": "ROAST"}'
    )
    assert decision.intent_tag == "roast"


def test_parse_decision_intent_tag_dropped_when_unknown():
    decision = parse_response_decision(
        '{"should_respond": true, "response_text": "x", "intent_tag": "not_a_real_tag"}'
    )
    assert decision.intent_tag is None


def test_parse_decision_drops_legacy_signals_object():
    # Old quantitative outputs included a "signals" object the new model no
    # longer emits; it must be popped defensively rather than crash validation.
    text = (
        '{"should_respond": true, "response_text": "x", '
        '"signals": {"velocity": 0.5, "fatigue": 0.1}}'
    )
    decision = parse_response_decision(text)
    assert decision.should_respond is True
    assert not hasattr(decision, "signals")


def test_parse_decision_non_dict_stances_reset_to_empty():
    decision = parse_response_decision(
        '{"should_respond": true, "response_text": "x", "stances": "nope"}'
    )
    assert decision.stances == {}


def test_parse_decision_handles_markdown_fenced_payload():
    text = '```json\n{"should_respond": true, "response_text": "fenced"}\n```'
    decision = parse_response_decision(text)
    assert decision.response_text == "fenced"


def test_parse_decision_garbage_raises_value_error():
    # No JSON at all propagates as ValueError from extract_json_object -- a
    # well-typed failure the caller can catch, NOT an unexpected crash.
    with pytest.raises(ValueError):
        parse_response_decision("the model said nothing useful")


# ---------------------------------------------------------------------------
# parse_context_summary: optional enrichment -> safe-default on bad input
# ---------------------------------------------------------------------------


def test_parse_context_summary_garbage_returns_safe_default():
    # Unlike the decision parser, the perception/context parser must NEVER
    # raise: bad input degrades to an empty, not-relevant summary so the cycle
    # proceeds.
    summary = parse_context_summary("totally not json")
    assert isinstance(summary, ContextSummary)
    assert summary.relevant_context is False
    assert summary.summary == ""
    assert summary.compressed_relevant_context == ""
    assert summary.context_message_ids == []


def test_parse_context_summary_non_object_json_returns_safe_default():
    # Valid JSON but not an object (a bare list) also degrades, not raises.
    summary = parse_context_summary("[1, 2, 3]")
    assert summary.relevant_context is False
    assert summary.summary == ""


def test_parse_context_summary_relevant_true_populates_fields():
    text = json.dumps(
        {
            "relevant_context": True,
            "summary": "they were arguing about the foo deal",
            "context_message_ids": [10, 11, 12],
        }
    )
    summary = parse_context_summary(text)
    assert summary.relevant_context is True
    assert summary.summary == "they were arguing about the foo deal"
    assert summary.context_message_ids == [10, 11, 12]
    # Legacy summary is mirrored into the new compressed field when that is empty.
    assert summary.compressed_relevant_context == "they were arguing about the foo deal"


def test_parse_context_summary_relevant_false_clears_all_content():
    # When the model marks context irrelevant, every content field is wiped so
    # nothing stale leaks downstream even if the model still filled them in.
    text = json.dumps(
        {
            "relevant_context": False,
            "summary": "should be cleared",
            "compressed_relevant_context": "also cleared",
            "context_message_ids": [1, 2],
            "high_level_included": True,
            "direct_mention_or_continuation": True,
        }
    )
    summary = parse_context_summary(text)
    assert summary.relevant_context is False
    assert summary.summary == ""
    assert summary.compressed_relevant_context == ""
    assert summary.context_message_ids == []
    assert summary.high_level_included is False
    assert summary.direct_mention_or_continuation is False


def test_parse_context_summary_coerces_string_bools():
    # _coerce_bool accepts "true"/"yes"/"1"; a stringly-typed flag is honored.
    text = json.dumps({"relevant_context": "true", "summary": "hi"})
    summary = parse_context_summary(text)
    assert summary.relevant_context is True
    assert summary.summary == "hi"


def test_parse_context_summary_missing_optional_fields_default():
    # Only relevant_context present: all the optional list/str/bool fields take
    # their defaults rather than raising.
    summary = parse_context_summary('{"relevant_context": false}')
    assert summary.summary == ""
    assert summary.reasoning == ""
    assert summary.context_message_ids == []
    assert summary.high_level_included is False
    assert summary.direct_mention_or_continuation is False


# ---------------------------------------------------------------------------
# _sanitize_response_text: the focused JSON-tail scrubber used above
# ---------------------------------------------------------------------------


def test_sanitize_response_text_passthrough_for_clean_value():
    assert _sanitize_response_text("just a normal reply") == "just a normal reply"


def test_sanitize_response_text_none_passthrough():
    assert _sanitize_response_text(None) is None


def test_sanitize_response_text_coerces_non_string_non_none():
    # A stray int is stringified (callers expect str | None).
    assert _sanitize_response_text(42) == "42"


def test_sanitize_response_text_cuts_leaked_key_tail():
    raw = "lmao that's wild','reply_to_message_id':99,"
    assert _sanitize_response_text(raw) == "lmao that's wild"


def test_sanitize_response_text_all_tail_becomes_none():
    # If after cutting the leaked tail nothing meaningful remains, return None
    # (an empty string would be a silent send of nothing).
    assert _sanitize_response_text("'reasoning': 'because'") is None
