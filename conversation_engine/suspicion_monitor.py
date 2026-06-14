from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Suspicion monitor
# ---------------------------------------------------------------------------
#
# When the room starts asking "are you a bot?", the single worst thing the bot
# can do is respond. Arguing — defending itself, denying, getting clever — is
# exactly what a cornered bot does, and every reply is fresh evidence for the
# accuser. What a real human does when half-jokingly called a bot is shrug it
# off and keep living; the cheap, robust tell-killer is to simply GO DARK for a
# while and let the moment pass. Silence reads as a person who didn't care
# enough to answer; a paragraph of denial reads as a script.
#
# So this module does two things: (1) detect accusations in recent traffic, and
# (2) hand back a jittered quiet duration. It never generates a rebuttal.


@dataclass(frozen=True)
class SuspicionResult:
    accused: bool
    severity: str  # 'none' | 'soft' | 'hard'
    matched: list[str] = field(default_factory=list)


# HARD accusations: direct, fairly unambiguous "you/this is a bot/AI" calls,
# tolerant of degen spelling (ur/youre/u, no apostrophes, "ai" vs "a.i."),
# plus the classic "chatgpt ass reply" / "bot detected" memes.
_HARD_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "are you a bot/ai", "r u a bot", "u a bot?"
    re.compile(r"\b(?:are|r)\s*(?:you|u|ya)\s+(?:a\s+)?(?:bot|ai|gpt|llm|chatgpt)\b"),
    # "you're a bot", "ur a bot", "youre an ai", "u are a bot"
    re.compile(
        r"\b(?:you'?re|ur|youre|u\s*r|u\s+are|you\s+are)\s+(?:a|an)?\s*(?:bot|ai|gpt|llm|chatgpt)\b"
    ),
    # "this is ai", "thats a bot", "that's ai", "it's a bot", "this guy is a bot"
    re.compile(
        r"\b(?:this|that'?s|thats|its|it'?s|that|it)\s+"
        r"(?:(?:guy|account|one)\s+)?(?:is\s+|was\s+)?(?:a\s+|an\s+)?"
        r"(?:bot|ai|gpt|llm|chatgpt)\b"
    ),
    # meme: "chatgpt ass reply/answer/response"
    re.compile(r"\b(?:chatgpt|gpt|ai)\s+ass\b"),
    # "bot detected", "ai detected", "bot confirmed"
    re.compile(r"\b(?:bot|ai|gpt)\s+(?:detected|confirmed|alert|spotted)\b"),
    # "literally a bot", "actual bot", "fucking bot/ai"
    re.compile(
        r"\b(?:literally|actual(?:ly)?|fuckin'?g?|legit)\s+(?:a\s+|an\s+)?(?:bot|ai|chatgpt)\b"
    ),
    # "talking to a bot", "responding like a bot"
    re.compile(
        r"\b(?:talking|replying|responding|typing)\s+(?:to\s+)?(?:like\s+)?(?:a\s+)?(?:bot|ai)\b"
    ),
)

# SOFT accusations: hedged / question-y suspicion, "is this even a real person",
# "sus", "kinda ai", "smells like a bot" — real but deniable.
_SOFT_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "is this a real person", "are you even real", "are you human"
    re.compile(
        r"\b(?:are|r)\s*(?:you|u|ya)\s+(?:even\s+)?(?:real|human|a\s+real\s+(?:person|human|guy))\b"
    ),
    re.compile(
        r"\bis\s+(?:this|that|he|she|it)\s+(?:even\s+)?(?:a\s+)?real\s+(?:person|human|guy|one)\b"
    ),
    # "sounds/feels/smells/reads like a bot/ai"
    re.compile(
        r"\b(?:sounds?|feels?|smells?|reads?|looks?)\s+like\s+(?:a\s+|an\s+)?(?:bot|ai|chatgpt|gpt)\b"
    ),
    # "kinda/sorta bot", "lowkey ai", "sus bot"
    re.compile(r"\b(?:kinda|sorta|lowkey|highkey|sus(?:\s+as\s+fuck)?)\s+(?:a\s+)?(?:bot|ai)\b"),
    # "bot vibes", "ai vibes", "npc vibes"
    re.compile(r"\b(?:bot|ai|npc)\s+(?:vibes?|energy|coded)\b"),
    # bare "npc" call-out
    re.compile(r"\b(?:you'?re|ur|youre|u)\s+(?:an?\s+)?npc\b"),
)


def _norm(text: str) -> str:
    """Lowercase and collapse whitespace so spacing quirks don't break matches."""
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def scan_for_accusation(
    recent_texts: list[str],
    *,
    bot_username: str | None = None,
) -> SuspicionResult:
    """Scan recent messages for accusations that the bot/an account is a bot/AI.

    Case-insensitive and tolerant of degen spelling. ``bot_username`` is accepted
    so callers can pass the bot's handle (used to prioritise directly-addressed
    accusations); matching is phrase-based and works with or without it.

    Returns a :class:`SuspicionResult`. ``severity`` is ``'hard'`` if any direct
    accusation matched, ``'soft'`` if only hedged suspicion matched, else
    ``'none'``. ``matched`` holds the offending (normalised) message snippets.
    """
    hard_hits: list[str] = []
    soft_hits: list[str] = []
    for raw in recent_texts:
        text = _norm(raw)
        if not text:
            continue
        if any(p.search(text) for p in _HARD_PATTERNS):
            hard_hits.append(text)
        elif any(p.search(text) for p in _SOFT_PATTERNS):
            soft_hits.append(text)

    if hard_hits:
        return SuspicionResult(accused=True, severity="hard", matched=hard_hits + soft_hits)
    if soft_hits:
        return SuspicionResult(accused=True, severity="soft", matched=soft_hits)
    return SuspicionResult(accused=False, severity="none", matched=[])


# Quiet-period bands (seconds). Hard accusation => go dark for hours; soft => go
# quiet for tens of minutes. Both jittered so the cooldown isn't a fixed,
# fingerprintable interval.
_HARD_DARK_RANGE = (90 * 60.0, 240 * 60.0)  # 1.5h .. 4h
_SOFT_DARK_RANGE = (20 * 60.0, 60 * 60.0)  # 20min .. 1h


def go_dark_seconds(result: SuspicionResult, rng: random.Random) -> float:
    """How long to stay silent after an accusation, in seconds.

    Returns ``0.0`` when not accused. Otherwise a jittered quiet period: hours
    for a hard accusation, tens of minutes for a soft one. Going dark (rather
    than replying) is the point — arguing is what a bot does; a human just stops
    answering and moves on.
    """
    if not result.accused:
        return 0.0
    lo, hi = _HARD_DARK_RANGE if result.severity == "hard" else _SOFT_DARK_RANGE
    return rng.uniform(lo, hi)
