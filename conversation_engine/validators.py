from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from conversation_engine.ai_client import ResponseDecision
from conversation_engine.config import EngineConfig


# ---------------------------------------------------------------------------
# Redundancy detection
# ---------------------------------------------------------------------------

def _normalize_for_dedup(text: str) -> str:
    """Lowercase, strip punctuation/whitespace so trivial variations still count as identical."""
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def is_duplicate_response(text: str, recent_texts: Iterable[str]) -> bool:
    """True if `text` is an exact (normalized) match of any recent bot message."""
    norm = _normalize_for_dedup(text)
    if not norm:
        return False
    return any(_normalize_for_dedup(prev) == norm for prev in recent_texts)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def is_similar_response(
    text: str,
    recent_texts: Iterable[str],
    *,
    jaccard_threshold: float = 0.6,
    opener_words: int = 4,
) -> bool:
    """True if `text` is *too similar* to a recent bot message.

    Catches near-duplicates that exact matching misses: same content reworded,
    or the same opening run of words (the bot's habit of starting many replies
    the same way, e.g. "bro went from ...", "bro said ... like it's ...").
    """
    cur_tokens = _tokens(text)
    if not cur_tokens:
        return False
    cur_set = set(cur_tokens)
    cur_opener = tuple(cur_tokens[:opener_words])
    for prev in recent_texts:
        prev_tokens = _tokens(prev)
        if not prev_tokens:
            continue
        # High token overlap => reworded duplicate.
        if _jaccard(cur_set, set(prev_tokens)) >= jaccard_threshold:
            return True
        # Same multi-word opening => formulaic repeat.
        if opener_words >= 3 and len(cur_tokens) >= opener_words and len(prev_tokens) >= opener_words:
            if cur_opener == tuple(prev_tokens[:opener_words]):
                return True
    return False


# Formulaic phrases the model overuses as openers/closers. These are dead
# giveaways when repeated across messages ("pick a lane", "pick a struggle",
# "bro went from X to Y", "the duality of man", etc.). We block a reply that
# leans on one of these *if it already used the same tic recently*.
_TIC_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bpick a lane\b"),
    re.compile(r"\bpick a struggle\b"),
    re.compile(r"\bpick one\b"),
    re.compile(r"\bread the room\b"),
    re.compile(r"\bbro (?:went|going) from .* to .*"),
    re.compile(r"\bbro said .* like it'?s\b"),
    re.compile(r"\b(?:the )?duality of man\b"),
    re.compile(r"\bspeedrun any%\b"),
    re.compile(r"\bcope is real\b"),
    re.compile(r"\bfamous last words\b"),
    re.compile(r"\bwe'?re done here\b"),
)


def matched_tics(text: str) -> list[str]:
    low = (text or "").lower()
    return [p.pattern for p in _TIC_PATTERNS if p.search(low)]


def reuses_recent_tic(text: str, recent_texts: Iterable[str]) -> bool:
    """True if `text` uses a formulaic tic that also appears in a recent message."""
    cur = set(matched_tics(text))
    if not cur:
        return False
    for prev in recent_texts:
        if cur & set(matched_tics(prev)):
            return True
    return False


# ---------------------------------------------------------------------------
# Emoji limiting
# ---------------------------------------------------------------------------

# Broad emoji/pictograph ranges (covers the 😭 💀 😱 family used in chat).
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # symbols, pictographs, supplemental, emoticons
    "\U00002600-\U000027BF"  # misc symbols + dingbats
    "\U0001F1E6-\U0001F1FF"  # regional indicators (flags)
    "\U00002B00-\U00002BFF"  # arrows/stars
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U0000200D"             # zero-width joiner
    "]+",
    flags=re.UNICODE,
)


def count_emojis(text: str) -> int:
    return sum(len(m.group()) for m in _EMOJI_RE.finditer(text or ""))


def strip_emojis(text: str) -> str:
    cleaned = _EMOJI_RE.sub("", text or "")
    # Collapse the double-spaces / dangling spaces left behind.
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()


def enforce_emoji_budget(
    text: str,
    recent_texts: Sequence[str],
    *,
    window: int = 5,
) -> str:
    """Return `text` with emojis limited across consecutive messages.

    Rule (per user spec):
    - Many emojis in a *single* message are fine — no per-message cap.
    - Emojis in consecutive messages are not. If any of the last
      ``window - 1`` bot messages contained an emoji, strip *all* emojis from
      this one; otherwise leave the text completely untouched.

    An emoji-only message follows the same rule: it sends as-is when recent
    messages were emoji-free, but becomes empty (and is dropped upstream) when
    a recent message already used emojis.
    """
    if not text:
        return text
    recent_window = list(recent_texts)[:max(0, window - 1)]
    recent_emoji_used = any(count_emojis(t) > 0 for t in recent_window)
    if recent_emoji_used:
        return strip_emojis(text)
    return text


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------

def validate(
    decision: ResponseDecision,
    config: EngineConfig,
    recent_bot_texts: Iterable[str] | None = None,
) -> tuple[bool, str | None]:
    """Validate (and lightly normalize) a decision before sending.

    May mutate ``decision.response_text`` to enforce the emoji budget. Rejects
    empty, over-length, avoided-user, low-confidence, and redundant responses
    (exact duplicate, near-duplicate/formulaic, or reused verbal tic).
    """
    if not decision.should_respond:
        return False, "decision_should_not_respond"
    if decision.confidence < config.ai.min_confidence_to_send:
        return False, f"low_confidence:{decision.confidence}"
    if decision.reply_to_user_id in set(config.prompt.avoid_users):
        return False, f"avoided_user:{decision.reply_to_user_id}"
    if not decision.response_text or not decision.response_text.strip():
        return False, "empty_response"

    recent = list(recent_bot_texts) if recent_bot_texts else []

    # Enforce the emoji budget first (mutates the outgoing text). Done before
    # the length check so the stripped text is what we measure and send.
    if config.emoji_window > 0:
        decision.response_text = enforce_emoji_budget(
            decision.response_text,
            recent,
            window=config.emoji_window,
        )
        if not decision.response_text.strip():
            return False, "empty_after_emoji_strip"

    if len(decision.response_text) > 4096:
        return False, "telegram_message_too_long"

    if recent:
        if is_duplicate_response(decision.response_text, recent):
            return False, "duplicate_of_recent_response"
        if is_similar_response(decision.response_text, recent):
            return False, "too_similar_to_recent_response"
        if reuses_recent_tic(decision.response_text, recent):
            return False, "reused_verbal_tic"

    return True, None
