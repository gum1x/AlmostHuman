"""Offline, deterministic tests for the wired decision path.

Covers conversation_engine.context_builder.build_request2_constraints (feedback
wording branches, preferred-tone surfacing, reflection clipping, empty path) and
conversation_engine.prompts.build_decide_and_draft_prompt (constraints injection
and the empty no-op). Duck-typed fakes only -- no DB, no network.
"""

from __future__ import annotations

from types import SimpleNamespace

from conversation_engine.context_builder import ContextBundle, build_request2_constraints
from conversation_engine.prompts import build_decide_and_draft_prompt


class FakePersona:
    def __init__(self, identity_summary: str):
        self.identity_summary = identity_summary


class FakeReflection:
    def __init__(self, updated_summary: str):
        self.updated_summary = updated_summary


class FakeProfile:
    def __init__(self, user_id: int, notes: str | None):
        self.user_id = user_id
        self.notes = notes


def _context_bundle(context: str = "target: 7 user_2 reply_to=none: gm") -> ContextBundle:
    return ContextBundle(
        context=context,
        candidate_user_ids=[2],
        relationship_profiles=[],
        avg_feedback_score=0.0,
    )


# --- build_request2_constraints: feedback wording branches ---


def test_feedback_branch_landed_well():
    out = build_request2_constraints(
        current_persona=FakePersona("long-time member, low ego"),
        latest_reflection=None,
        relationship_profiles=[],
        avg_feedback_score=0.5,
    )
    assert "landed well" in out
    assert "+0.50" in out
    assert "landed flat/poorly" not in out
    assert "roughly neutral" not in out


def test_feedback_branch_flat_poorly():
    out = build_request2_constraints(
        current_persona=FakePersona("long-time member, low ego"),
        latest_reflection=None,
        relationship_profiles=[],
        avg_feedback_score=-0.5,
    )
    assert "landed flat/poorly" in out
    assert "-0.50" in out
    assert "landed well" not in out
    assert "roughly neutral" not in out


def test_feedback_branch_neutral_at_boundaries():
    # 0.15 and -0.15 are not strictly past the thresholds -> neutral wording.
    for score in (0.0, 0.15, -0.15):
        out = build_request2_constraints(
            current_persona=FakePersona("long-time member"),
            latest_reflection=None,
            relationship_profiles=[],
            avg_feedback_score=score,
        )
        assert "roughly neutral" in out, score
        assert "landed well" not in out
        assert "landed flat/poorly" not in out


# --- build_request2_constraints: relationship preferred-tone ---


def test_preferred_tone_surfaces_relationship_tone_line():
    profile = FakeProfile(42, "knows their stuff\nPreferred tone: dry and blunt")
    out = build_request2_constraints(
        current_persona=FakePersona("long-time member"),
        latest_reflection=None,
        relationship_profiles=[profile],
        avg_feedback_score=0.0,
    )
    assert "=== RELATIONSHIP TONE ===" in out
    assert "user_42: preferred_tone=dry and blunt" in out


def test_no_preferred_tone_omits_relationship_tone_section():
    profile = FakeProfile(42, "just some notes, no tone here")
    out = build_request2_constraints(
        current_persona=FakePersona("long-time member"),
        latest_reflection=None,
        relationship_profiles=[profile],
        avg_feedback_score=0.0,
    )
    assert "RELATIONSHIP TONE" not in out


# --- build_request2_constraints: reflection clipping ---


def test_reflection_summary_is_clipped():
    long_summary = "x" * 400
    out = build_request2_constraints(
        current_persona=FakePersona("long-time member"),
        latest_reflection=FakeReflection(long_summary),
        relationship_profiles=[],
        avg_feedback_score=0.0,
    )
    assert "What I've learned in this chat:" in out
    learned_line = next(
        line for line in out.splitlines() if line.startswith("What I've learned in this chat:")
    )
    # _clip caps the summary at 240 chars and appends an ellipsis.
    assert "..." in learned_line
    assert long_summary not in out
    # 240-char clip means far fewer than the original 400 'x' chars survive.
    assert learned_line.count("x") < 400


# --- build_request2_constraints: None / empty path ---


def test_empty_path_still_returns_usable_string():
    out = build_request2_constraints(
        current_persona=None,
        latest_reflection=None,
        relationship_profiles=[],
        avg_feedback_score=0.0,
    )
    assert out.strip()
    assert "=== PERSONA ALIGNMENT ===" in out
    assert "Core identity: unknown" in out
    assert "roughly neutral" in out
    assert "RELATIONSHIP TONE" not in out


def test_target_message_block_prepended_when_present():
    out = build_request2_constraints(
        current_persona=FakePersona("long-time member"),
        latest_reflection=None,
        relationship_profiles=[],
        avg_feedback_score=0.0,
        target_message_block="target: 7 user_2 reply_to=none: gm",
    )
    assert out.startswith("target: 7 user_2 reply_to=none: gm")


# --- build_decide_and_draft_prompt: constraints injection vs no-op ---


def test_prompt_injects_nonempty_constraints():
    bundle = _context_bundle()
    constraints = "=== FEEDBACK LEARNING ===\nbe sharper or stay silent more"
    prompt, system = build_decide_and_draft_prompt(bundle, SimpleNamespace(), constraints)
    assert constraints in prompt
    assert prompt.startswith(bundle.context)
    # The system prompt is returned alongside the user prompt.
    assert isinstance(system, str) and system


def test_prompt_empty_constraints_is_noop():
    bundle = _context_bundle()
    prompt_with_empty, _ = build_decide_and_draft_prompt(bundle, SimpleNamespace(), "")
    prompt_with_none, _ = build_decide_and_draft_prompt(bundle, SimpleNamespace(), None)
    # No stray injected block: empty and None behave identically.
    assert prompt_with_empty == prompt_with_none
    # Prompt begins with the context, immediately followed by the standing
    # instruction line -- no blank constraints block wedged between them.
    assert prompt_with_empty.startswith(bundle.context)
    assert f"{bundle.context}\n\nNew messages arrived." in prompt_with_empty


def test_prompt_whitespace_only_constraints_is_noop():
    bundle = _context_bundle()
    prompt_ws, _ = build_decide_and_draft_prompt(bundle, SimpleNamespace(), "   \n  ")
    prompt_none, _ = build_decide_and_draft_prompt(bundle, SimpleNamespace(), None)
    assert prompt_ws == prompt_none
