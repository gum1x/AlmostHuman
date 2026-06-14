from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import Any

from conversation_engine.config import EngineConfig
from conversation_engine.handles import HandleMap
from conversation_engine.validators import (
    apply_donor_casing,
    apply_donor_terminal_punct,
    count_emojis,
    strip_emojis,
    violates_ai_tell,
)

# Default donor profile location (matches the Phase-1/2 analysis artifact).
_DEFAULT_DONOR_PROFILE = "data/prod_export/analysis/donor_profile.json"

# Strata order in the donor exemplar_pool, shortest -> longest.
_STRATA = ("1w", "2-4w", "5-8w", "9-15w")

# Absolute ceiling on output length (the donor's longest register); the per-message
# budget sampled from the donor distribution does the real shaping, this just caps
# the rare long-tail draw.
_WORD_CLAMP = 18

# Inverse-CDF anchor points (cumulative_prob, word_count) reconstructing the donor's
# word-count distribution: 26.9% one-word, p50=3, p75=5, p90=8, tail up to the clamp.
# A per-message budget drawn from this pulls the generator's length distribution onto
# the donor's instead of letting kimi default to fuller sentences (the residual
# char_len/word_count tell the n=237 A/B exposed).
_LEN_CDF_ANCHORS = (
    (0.0, 1.0),
    (0.269, 1.0),
    (0.5, 3.0),
    (0.75, 5.0),
    (0.9, 8.0),
    (1.0, float(_WORD_CLAMP)),
)


def _sample_word_budget(rng: random.Random, anchors=_LEN_CDF_ANCHORS) -> int:
    """Draw a per-message word budget from the donor's empirical length CDF."""
    u = rng.random()
    for (c0, w0), (c1, w1) in zip(anchors, anchors[1:]):
        if u <= c1:
            span = c1 - c0
            frac = 0.0 if span <= 0 else (u - c0) / span
            return max(1, round(w0 + (w1 - w0) * frac))
    return int(anchors[-1][1])


@dataclass
class ToneCapsule:
    register: str = "peer"
    relationship_stage: str = "regular"
    address_form: str | None = None


def build_recent_lines(
    enriched_messages: Any,
    handle_map: HandleMap | None = None,
    max_lines: int = 10,
) -> list[str]:
    """Format the recent window as donor-style "uXXXX: text" lines.

    Mirrors the donor training format (``u<sender_id>: <text>``) but prefers
    ``@handle: text`` when a HandleMap supplies a username for the sender.
    Accepts EnrichedMessage-like objects (attrs) or plain dicts; skips rows
    with no sender or no text. Returns at most ``max_lines`` most-recent lines.
    """
    lines: list[str] = []
    for m in enriched_messages or []:
        if isinstance(m, dict):
            sid = m.get("sender_id")
            txt = m.get("cleaned_text") or m.get("text") or ""
        else:
            sid = getattr(m, "sender_id", None)
            txt = getattr(m, "cleaned_text", None) or getattr(m, "text", None) or ""
        txt = (txt or "").strip()
        if sid is None or not txt:
            continue
        label = None
        if handle_map is not None:
            label = handle_map.handle_for(int(sid))
        if not label:
            label = f"u{sid}"
        lines.append(f"{label}: {txt}")
    return lines[-max_lines:]


class WordGenerator:
    """API word-generator: kimi-k2 writes the actual message end-to-end.

    Rich recent context + a balanced donor few-shot + a tone capsule go in; a
    single short message comes out, shaped by the donor's surface statistics and
    cleaned through the validator stack. It does NOT decide whether to speak (the
    brain does); it only writes WORDS.
    """

    def __init__(
        self,
        ai_client: Any,
        config: EngineConfig,
        *,
        donor_profile_path: str = _DEFAULT_DONOR_PROFILE,
        exemplar_k: int = 16,
    ):
        self.ai_client = ai_client
        self.config = config
        self.exemplar_k = exemplar_k
        # A missing/unreadable donor profile should degrade gracefully (empty
        # exemplar pool + stats) rather than crash construction; the generator
        # still works, just without donor few-shots/surface stats.
        try:
            with open(donor_profile_path, encoding="utf-8") as fh:
                profile = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            profile = {}
        self._pool: dict[str, list[str]] = profile.get("exemplar_pool", {}) or {}
        self.donor_stats: dict[str, Any] = profile.get("stats", {}) or {}
        seed = getattr(config, "behavioral_rng_seed", None)
        self.exemplars = self._sample_exemplars(random.Random(seed), exemplar_k)

    def _sample_exemplars(self, rng: random.Random, k: int) -> list[str]:
        """Sample a balanced set of ``k`` donor exemplars across the strata.

        Round-robins one stratum at a time (shortest -> longest) so the few-shot
        is weighted toward the donor's short messages without dropping the longer
        registers, sampling without replacement within each stratum. Deterministic
        given ``rng``.
        """
        # Pre-shuffle each stratum once with the seeded rng (deterministic).
        shuffled: dict[str, list[str]] = {}
        for stratum in _STRATA:
            items = list(self._pool.get(stratum, []))
            rng.shuffle(items)
            shuffled[stratum] = items
        chosen: list[str] = []
        cursors = {s: 0 for s in _STRATA}
        # Round-robin until we have k or every stratum is exhausted.
        while len(chosen) < k:
            progressed = False
            for stratum in _STRATA:
                if len(chosen) >= k:
                    break
                items = shuffled[stratum]
                idx = cursors[stratum]
                if idx < len(items):
                    chosen.append(items[idx])
                    cursors[stratum] = idx + 1
                    progressed = True
            if not progressed:
                break
        return chosen

    def _system_prompt(self, tone: ToneCapsule, intent_tag: str | None) -> str:
        stats = self.donor_stats
        lc = stats.get("lowercase_rate", 0.964)
        med = (stats.get("word_count_p25_50_75_90") or [1, 3, 5, 8])[1]
        few_shot = "\n".join(f"- {ex}" for ex in self.exemplars)
        tone_line = f"register: {tone.register}; relationship: {tone.relationship_stage}"
        if tone.address_form:
            tone_line += f"; address them as: {tone.address_form}"
        intent_line = f"\nwhat you want to get across: {intent_tag}" if intent_tag else ""
        return (
            "you are a regular in a chaotic crypto/telegram group chat. you are not "
            "an assistant. write ONE short chat message the way this specific person "
            "writes:\n"
            f"- almost always lowercase (~{int(round(lc * 100))}% of the time)\n"
            f"- very short, around {med} words, often 1-3 words\n"
            "- blunt, reactive, dry; sometimes mean, sometimes warm\n"
            "- rarely asks questions, rarely uses emoji\n"
            "- no greetings, no sign-offs, no explanations, no quotes around it\n"
            "- typos and lazy spelling are fine and authentic\n\n"
            "this is exactly how you talk (match the voice, do not copy verbatim):\n"
            f"{few_shot}\n\n"
            f"tone for this message -> {tone_line}{intent_line}\n\n"
            "output ONLY the message text. nothing else."
        )

    def _user_message(
        self,
        recent_lines: list[str],
        reply_target_line: str | None,
        word_budget: int | None = None,
    ) -> str:
        window = "\n".join(recent_lines) if recent_lines else "(no recent messages)"
        out = f"recent chat:\n{window}"
        if reply_target_line:
            out += f"\n\nyou are replying to this line:\n{reply_target_line}"
        if word_budget is not None:
            limit = "1 word" if word_budget == 1 else f"{word_budget} words or fewer"
            out += f"\n\nkeep it to {limit}."
        out += "\n\nyour message:"
        return out

    async def generate(
        self,
        *,
        recent_lines: list[str],
        tone: ToneCapsule | None = None,
        intent_tag: str | None = None,
        rng: random.Random,
        reply_target_line: str | None = None,
    ) -> str:
        """Generate one donor-shaped chat message. Returns '' on failure/suppression."""
        tone = tone or ToneCapsule()
        burst = bool(intent_tag and "burst" in intent_tag.lower())
        # Donor-matched per-message length: instruct toward it and clamp as backstop.
        word_budget = _sample_word_budget(rng)

        messages = [
            {"role": "system", "content": self._system_prompt(tone, intent_tag)},
            {
                "role": "user",
                "content": self._user_message(recent_lines, reply_target_line, word_budget),
            },
        ]

        text = await self._call(messages)
        cleaned = self._post_process(text, rng, burst=burst, word_budget=word_budget)
        if cleaned and violates_ai_tell(cleaned) is not None:
            # One retry with an explicit "say it like a real person" nudge.
            retry_messages = messages + [
                {"role": "assistant", "content": text},
                {
                    "role": "user",
                    "content": (
                        "that sounded like an AI. say it like a real person in this "
                        "chat would — short, blunt, lowercase, no assistant phrasing."
                    ),
                },
            ]
            text = await self._call(retry_messages)
            cleaned = self._post_process(text, rng, burst=burst, word_budget=word_budget)
            if cleaned and violates_ai_tell(cleaned) is not None:
                return ""
        return cleaned

    async def _call(self, messages: list[dict[str, str]]) -> str:
        """Call the lowest-friction raw-chat path on the AI client."""
        call_raw = getattr(self.ai_client, "call_raw", None)
        if call_raw is not None:
            result = await call_raw(messages, temperature=0.8)
            return getattr(result, "text", "") or ""
        # Fallback for clients without call_raw: fold the system + window into a
        # single decision-model prompt.
        system = next((m["content"] for m in messages if m["role"] == "system"), None)
        user = "\n\n".join(m["content"] for m in messages if m["role"] != "system")
        result = await self.ai_client.call_decision_model(user, system)
        return getattr(result, "text", "") or ""

    def _post_process(
        self, text: str, rng: random.Random, *, burst: bool, word_budget: int | None = None
    ) -> str:
        """Strip wrapping, clamp length, apply donor casing + emoji budget."""
        cleaned = _strip_wrapping(text)
        if not cleaned:
            return ""
        if not burst:
            # Take the first non-empty line unless this is an intentional burst.
            for line in cleaned.splitlines():
                if line.strip():
                    cleaned = line.strip()
                    break
        clamp = _WORD_CLAMP if word_budget is None else min(word_budget, _WORD_CLAMP)
        cleaned = _clamp_words(cleaned, clamp)
        cleaned = self._enforce_emoji_budget(cleaned)
        cleaned = apply_donor_casing(
            cleaned, rng, getattr(self.config, "behavioral_donor_lowercase_rate", 0.964)
        )
        cleaned = apply_donor_terminal_punct(cleaned, rng)
        return cleaned.strip()

    def _enforce_emoji_budget(self, text: str) -> str:
        """Donor rarely uses emoji (emoji_msg_rate ~0.0045): keep at most one."""
        if count_emojis(text) <= 1:
            return text
        return strip_emojis(text)


# ---------------------------------------------------------------------------
# Module-level cleaning helpers
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?|\n?```$")


def _strip_wrapping(text: str) -> str:
    """Strip markdown fences and a single layer of wrapping quotes."""
    s = (text or "").strip()
    if not s:
        return ""
    s = _FENCE_RE.sub("", s).strip()
    # Strip one symmetric layer of surrounding quotes (straight or curly).
    for open_q, close_q in (('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’")):
        if len(s) >= 2 and s.startswith(open_q) and s.endswith(close_q):
            s = s[1:-1].strip()
            break
    return s


def _clamp_words(text: str, max_words: int) -> str:
    """Clamp to at most ``max_words`` whitespace-delimited words."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])
