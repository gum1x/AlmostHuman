from __future__ import annotations

import random
import re
from collections.abc import Iterable, Sequence

from conversation_engine.ai_client import ResponseDecision
from conversation_engine.config import EngineConfig
from conversation_engine.handles import HandleMap

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
        if (
            opener_words >= 3
            and len(cur_tokens) >= opener_words
            and len(prev_tokens) >= opener_words
        ):
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
    "\U0001f300-\U0001faff"  # symbols, pictographs, supplemental, emoticons
    "\U00002600-\U000027bf"  # misc symbols + dingbats
    "\U0001f1e6-\U0001f1ff"  # regional indicators (flags)
    "\U00002b00-\U00002bff"  # arrows/stars
    "\U0000fe00-\U0000fe0f"  # variation selectors
    "\U0000200d"  # zero-width joiner
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
    recent_window = list(recent_texts)[: max(0, window - 1)]
    recent_emoji_used = any(count_emojis(t) > 0 for t in recent_window)
    if recent_emoji_used:
        return strip_emojis(text)
    return text


# ---------------------------------------------------------------------------
# Donor-voice shaping (Phase 2 helpers — pure; NOT yet wired into validate())
# ---------------------------------------------------------------------------
#
# These reproduce a donor regular's surface statistics:
# bimodal casing (lowercase_rate 0.964), rare terminal punctuation
# (terminal_punct_rate ~0.0585), near-zero emoji, and very short messages.
# Phase 3 integrates them into the send path; for now they are standalone so
# callers can opt in. validate()'s existing behavior is untouched.

# Preserve URLs and @handles from being lowercased — these are the only
# tokens where casing is semantically load-bearing (handles are case-insensitive
# on Telegram but we keep the donor's literal form rather than mangle them).
_PRESERVE_CASE_RE = re.compile(r"(https?://\S+|@\w+)", flags=re.IGNORECASE)


def apply_donor_casing(text: str, rng: random.Random, lowercase_rate: float = 0.964) -> str:
    """Reproduce the donor's bimodal casing with a per-MESSAGE coin flip.

    With probability ``lowercase_rate`` the whole message is lowercased; otherwise
    it is left exactly as-is. This matches the donor's measured ``lowercase_rate``
    (0.964) — they almost always type lowercase but occasionally don't — instead of
    the blunt "always lowercase" transform, which is itself a tell.

    URLs and @handles keep their original casing (they are spliced back in after
    lowercasing the rest).
    """
    if not text:
        return text
    if rng.random() >= lowercase_rate:
        return text
    # Lowercase everything except URLs/@handles, which we restore verbatim.
    parts = _PRESERVE_CASE_RE.split(text)
    out: list[str] = []
    for index, part in enumerate(parts):
        # re.split with one capture group yields: [text, match, text, match, ...]
        # so odd indices are the preserved tokens.
        out.append(part if index % 2 == 1 else part.lower())
    return "".join(out)


def strip_terminal_period(text: str) -> str:
    """Strip a single trailing '.' (deterministically — the CALLER decides when).

    Keeps '?' and '!' (the donor still asks/exclaims) and keeps an ellipsis
    '...' intact (a trailing-period strip there would mangle the trailing-off
    tone the donor uses). This shapes toward the donor's low
    ``terminal_punct_rate`` (~0.0585); applying it probabilistically is left to
    the caller so this helper stays deterministic and easy to test.
    """
    if not text:
        return text
    stripped = text.rstrip()
    if stripped.endswith("...") or not stripped.endswith("."):
        return text
    # Single trailing period only (not part of an ellipsis, guarded above).
    trailing_ws = text[len(stripped) :]
    return stripped[:-1] + trailing_ws


# Donor terminal-punctuation habit: terminal_punct_rate ~0.0585, question_rate
# ~0.0471 — i.e. the donor almost never ends a message with punctuation and rarely
# exclaims. The model over-punctuates, pushing the terminal_punct KS just out of
# band. This brings the realized rate onto the donor's: '.' always stripped, '!'
# almost always, '?' often (questions are content, so it's kept more than '!').
_STRIP_EXCLAIM_P = 0.85
_STRIP_QUESTION_P = 0.5


def apply_donor_terminal_punct(text: str, rng: random.Random) -> str:
    """Probabilistically strip a single trailing ! or ? (after the deterministic
    '.' strip) to match the donor's low terminal-punctuation rate. Keeps ellipsis."""
    text = strip_terminal_period(text)
    if not text:
        return text
    stripped = text.rstrip()
    if stripped.endswith("...") or not stripped:
        return text
    last = stripped[-1]
    if last == "!" and rng.random() < _STRIP_EXCLAIM_P:
        return stripped[:-1].rstrip() + text[len(stripped) :]
    if last == "?" and rng.random() < _STRIP_QUESTION_P:
        return stripped[:-1].rstrip() + text[len(stripped) :]
    return text


# Assistant-register "AI tells" — phrasing that immediately reads as a chatbot
# rather than a degen-group regular. violates_ai_tell() flags the first match so
# callers can reject/regenerate. Patterns are lowercased-substring or regex.
_AI_TELL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("certainly", re.compile(r"\bcertainly\b")),
    ("i'd be happy to", re.compile(r"\bi('?d| would) be happy to\b")),
    ("happy to help", re.compile(r"\bhappy to help\b")),
    ("as an ai", re.compile(r"\bas an ai\b")),
    ("as a language model", re.compile(r"\bas an? (?:large )?language model\b")),
    ("great question", re.compile(r"\bgreat question\b")),
    ("delve", re.compile(r"\bdelve\b")),
    ("it's worth noting", re.compile(r"\bit'?s worth noting\b")),
    ("in conclusion", re.compile(r"\bin conclusion\b")),
    ("to summarize", re.compile(r"\bto summari[sz]e\b")),
    ("i hope this helps", re.compile(r"\bi hope (?:this|that) helps\b")),
    ("let me know if", re.compile(r"\blet me know if\b")),
    ("feel free to", re.compile(r"\bfeel free to\b")),
    ("rest assured", re.compile(r"\brest assured\b")),
    ("it is important to note", re.compile(r"\bit is important to note\b")),
    ("i cannot", re.compile(r"\bi (?:cannot|can'?t) (?:assist|help) with\b")),
    ("tapestry", re.compile(r"\btapestry\b")),
    ("navigating", re.compile(r"\bnavigating the\b")),
    # Em-dash-heavy constructions (multiple em-dashes) read as polished AI prose.
    ("em-dash-heavy", re.compile(r"—.*—")),
)

# Convenience: the documented banlist of tell labels, in pattern order.
AI_TELL_BANLIST: tuple[str, ...] = tuple(label for label, _ in _AI_TELL_PATTERNS)


def violates_ai_tell(text: str) -> str | None:
    """Return the label of the first AI-register tell found in ``text``, else None."""
    low = (text or "").lower()
    for label, pattern in _AI_TELL_PATTERNS:
        if pattern.search(low):
            return label
    return None


def inject_mention(
    text: str,
    target_sender_id: int,
    handle_map: HandleMap,
    rng: random.Random,
    *,
    p: float = 0.0,
) -> str:
    """With probability ``p``, prepend ``'@handle '`` for the target sender.

    The donor occasionally @-mentions the person they're replying to; injecting a
    mention some fraction of the time (the caller sets ``p`` toward the observed
    @-mention rate) makes replies feel addressed rather than broadcast. Default
    ``p=0`` keeps this OFF unless a caller opts in.

    No-op when: the roll fails, no handle exists for the target, the text is
    empty, or the handle is already mentioned in the text.
    """
    if not text or rng.random() >= p:
        return text
    handle = handle_map.handle_for(target_sender_id)
    if not handle:
        return text
    if re.search(rf"(?<!\w){re.escape(handle)}\b", text, flags=re.IGNORECASE):
        return text
    return f"{handle} {text}"


def enforce_ack_dedup(
    text: str,
    todays_bot_texts: list[str],
    *,
    max_per_day: int = 2,
) -> bool:
    """True if this trivial "ack" was already sent ``>= max_per_day`` times TODAY.

    Short acknowledgements ('lol', 'same', 'fr', 'gm', ...) are fine occasionally
    but become a tell when repeated all day. The caller passes *today's* bot texts
    (from BotMemory) so the count is a persisted per-day counter — a more precise
    replacement for the blunt 6-message dedup window for these trivial messages.
    Returns False (i.e. allowed) for non-ack / longer messages so this only gates
    the trivial case.
    """
    norm = _normalize_for_dedup(text)
    if not norm:
        return False
    matches = sum(1 for prev in todays_bot_texts if _normalize_for_dedup(prev) == norm)
    return matches >= max_per_day


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
