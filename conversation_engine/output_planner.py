"""Output planner: turn "what to say" into human-shaped send actions.

The decision pipeline produces a single block of text. A real group member does
not always answer with a sentence: they often just drop a reaction, fire off a
sticker, or split one thought across two quick messages. This module reshapes a
finished decision into an ordered list of ``Action``s the scheduler can execute,
each carrying its own pre-send ``delay_before_s`` (computed by the humanizer).

Why a self-burst splitter is the headline feature: the behavioral baseline
(data/prod_export/analysis/behavioral_baseline.json) puts the group's burst_rate
band at [0.12, 0.31] — a meaningful share of real consecutive same-sender turns
are a single thought spread over multiple messages — while the bot self-bursts
only 0.011 of the time. That gap is the bot's single worst behavioral tell.
Splitting a naturally-multi-clause reply into 2 (rarely 3) messages ~2-4s apart
closes it.

Everything here is pure and deterministic: all randomness flows through an
injected ``random.Random`` and all send delays come from an injected humanizer,
so the same seed always yields the same plan. Nothing sleeps; delays are returned
as values for the caller to await *outside* the send transaction.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

# Follow-up burst messages land in a tight window after the lead message.
_BURST_GAP_MIN_S = 2.0
_BURST_GAP_MAX_S = 4.0

# Below this word count a message is a one-liner — never split it.
_MIN_SPLITTABLE_WORDS = 6


class Humanizer(Protocol):
    """Minimal contract the planner needs from the humanizer module.

    Matches ``conversation_engine.humanizer.compute_send_delay`` exactly so the
    scheduler can pass the module object straight through. Tests inject a fake
    that returns a fixed delay so plans stay deterministic.
    """

    def compute_send_delay(  # pragma: no cover - protocol
        self, text: str, rng, *, intent_tag: str | None = None
    ) -> float:
        ...


@dataclass
class Action:
    kind: str  # 'text' | 'react' | 'media' | 'typing'
    text: str | None = None
    emoji: str | None = None
    sticker_source_message_id: int | None = None
    reply_to_message_id: int | None = None
    delay_before_s: float = 0.0


@dataclass
class OutputPlan:
    actions: list[Action] = field(default_factory=list)
    suppressed: bool = False
    suppressed_reason: str | None = None


# Clause boundaries we split a thought on, in priority order: sentence-ending
# punctuation, then " and " / " but ", then a comma. The first boundary that
# yields two non-trivial halves wins.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CONJ_SPLIT = re.compile(r"\s+(?:and|but)\s+", re.IGNORECASE)
_COMMA_SPLIT = re.compile(r"\s*,\s+")


def _word_count(text: str) -> int:
    return len(text.split())


def _compute_send_delay(humanizer: Humanizer, text: str, rng, intent_tag: str | None) -> float:
    """First-action (reply-latency) delay from the humanizer for ``text``."""
    return float(humanizer.compute_send_delay(text, rng, intent_tag=intent_tag))


def _short_gap(rng) -> float:
    """A 2-4s inter-burst gap for a follow-up message."""
    return rng.uniform(_BURST_GAP_MIN_S, _BURST_GAP_MAX_S)


def _split_into_burst(text: str, rng, *, max_parts: int = 3) -> list[str]:
    """Split ``text`` into 2 (rarely 3) clause-aligned parts.

    Returns a single-element list when the text has no natural seam (the caller
    treats that as "do not burst").
    """
    text = text.strip()
    if _word_count(text) < 3:
        return [text]

    parts: list[str] = []
    for pattern in (_SENTENCE_SPLIT, _CONJ_SPLIT, _COMMA_SPLIT):
        pieces = [p.strip() for p in pattern.split(text) if p.strip()]
        if len(pieces) >= 2:
            parts = pieces
            break

    if len(parts) < 2:
        return [text]

    # Usually a 2-way split; occasionally keep a third part if one exists.
    if len(parts) >= 3 and rng.random() < 0.25:
        head, mid, *tail = parts
        return [head, mid, " ".join(tail)][:max_parts]
    head, *rest = parts
    return [head, " ".join(rest)]


def _is_splittable(text: str) -> bool:
    """True when the text is long/multi-clause enough to plausibly burst.

    Never split a 1-2 word message. Otherwise require either >= 6 words or at
    least one clause seam (sentence end, ' and '/' but ', or comma).
    """
    if _word_count(text) < 3:
        return False
    if _word_count(text) >= _MIN_SPLITTABLE_WORDS:
        return True
    if _SENTENCE_SPLIT.search(text) or _CONJ_SPLIT.search(text) or _COMMA_SPLIT.search(text):
        return True
    return False


def plan_output(
    *,
    text: str,
    reply_to_message_id: int | None,
    rng,
    humanizer: Humanizer,
    intent_tag: str | None = None,
    react_emoji: str | None = None,
    sticker_source_message_id: int | None = None,
    allow_media: bool = False,
    burst_rate_target: float = 0.20,
) -> OutputPlan:
    """Reshape a finished reply into a sequence of human-shaped send actions.

    Returns one of:
      * a single ``react`` action (react_only intent, or empty text + an emoji),
      * a single ``media`` action (opt-in: ``allow_media`` + media intent/seeded),
      * 1-3 ``text`` actions, the first carrying ``reply_to`` and any burst
        follow-ups free-floating,
      * a suppressed plan when there is nothing to send.

    Determinism: every branch that rolls a die uses ``rng``; every text delay
    comes from ``humanizer.compute_send_delay`` (lead) or a 2-4s gap helper
    (follow-ups). No wall-clock, no sleeping.
    """
    text = (text or "").strip()

    # --- react-only: a reaction in place of a sentence ----------------------
    # Many gate-passing moments want an emoji, not prose; this also stops the
    # bot from answering every single message with text.
    if react_emoji and (intent_tag == "react_only" or not text):
        return OutputPlan(
            actions=[
                Action(
                    kind="react",
                    emoji=react_emoji,
                    reply_to_message_id=reply_to_message_id,
                )
            ]
        )

    # --- media: opt-in only (live media send is unverified) -----------------
    media_requested = intent_tag == "media" or sticker_source_message_id is not None
    if allow_media and media_requested:
        # An explicit media intent always fires; a bare sticker source is a
        # seeded coin-flip so it stays occasional, not every turn.
        fire = intent_tag == "media" or rng.random() < 0.5
        if fire:
            return OutputPlan(
                actions=[
                    Action(
                        kind="media",
                        sticker_source_message_id=sticker_source_message_id,
                        reply_to_message_id=reply_to_message_id,
                    )
                ]
            )

    # --- nothing left to send ----------------------------------------------
    if not text:
        return OutputPlan(suppressed=True, suppressed_reason="empty")

    # --- self-burst splitter (the #1 tell fix) ------------------------------
    parts: list[str] = [text]
    if _is_splittable(text) and rng.random() < burst_rate_target:
        candidate = _split_into_burst(text, rng)
        if len(candidate) >= 2:
            parts = candidate

    actions: list[Action] = []
    for i, part in enumerate(parts):
        if i == 0:
            # Lead message: full reply latency, carries the reply target.
            actions.append(
                Action(
                    kind="text",
                    text=part,
                    reply_to_message_id=reply_to_message_id,
                    delay_before_s=_compute_send_delay(humanizer, part, rng, intent_tag),
                )
            )
        else:
            # Burst follow-up: short gap, free-floating (no reply_to).
            actions.append(
                Action(
                    kind="text",
                    text=part,
                    reply_to_message_id=None,
                    delay_before_s=_short_gap(rng),
                )
            )

    return OutputPlan(actions=actions)
