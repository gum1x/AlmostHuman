"""Narrow READ channel from a per-person dossier into the two destinations.

The dossier is a flat JSONB blob on the relationship profile (see
``.plans/voice-memory-redesign.md``). This module renders it into exactly two
shapes, enforcing the witnessed filter and the daily callback budget:

- ``render_decision_capsule`` -> a <=120-token DISPOSITION line for the DECISION
  BRAIN (WHAT/WHO/WHETHER). May include ``witnessed=false`` entries as *silent*
  context (the brain can reason on pre-join knowledge; it just never types it).
- ``select_voice_tone`` -> a ``ToneCapsule`` for the WORD-GENERATOR (HOW). No
  facts. ``witnessed=false`` material is invisible here.
- ``select_callback`` -> the ONE specific entity/token injected into the
  word-generator on a budgeted callback. The only path by which a concrete
  memory reaches the words; gated on budget + witnessed + recency.

Pure: no DB, no network. The scheduler passes already-fetched dossier dicts.

A dossier dict is shaped::

    {
      "entries": [
        {"type": "running_joke", "content": "...", "confidence": 0.8,
         "corroboration_count": 3, "evidence_msg_ids": [...],
         "t_valid": ..., "t_invalid": None, "witnessed": True,
         "last_referenced_at": <epoch-or-iso-or-None>, "use_count": 2},
        ...
      ],
      "tone": {"register_tag": "peer", "banter_permission": "high",
               "address_form": "mara", "relationship_stage": "regular"},
      "aliases": ["mara", "maraxxx"],
    }
"""

from __future__ import annotations

import random
from typing import Any

from conversation_engine.word_generator import ToneCapsule

# Token-count approximation: whitespace words * this factor (sub-word tokens).
_TOKENS_PER_WORD = 1.3

# A claim is promoted to an assertable fact at this many corroborations (the
# corroboration ladder; see the write policy). Below this it is a low-confidence
# claim and is NOT stated as fact to the brain.
_ASSERTABLE_CORROBORATION = 2

# Entry types that carry a concrete, voiceable entity for a budgeted callback,
# in preference order (running jokes / shared history first).
_CALLBACK_PREFERENCE = ("running_joke", "history_with_me", "nickname", "bag_held")


def _entries(dossier: Any) -> list[dict]:
    if not isinstance(dossier, dict):
        return []
    raw = dossier.get("entries") or []
    return [e for e in raw if isinstance(e, dict)]


def _witnessed(entry: dict) -> bool:
    """An entry is witnessed only when explicitly flagged True.

    Missing/None ``witnessed`` is treated as NOT witnessed (fail-closed): an
    entry whose provenance we can't confirm must never be voiced.
    """
    return entry.get("witnessed") is True


def _voiceable(entry: dict) -> bool:
    """Single gate shared by both voice-facing functions.

    An entry may reach the WORD-GENERATOR (tone or callback) only if it was
    witnessed (post-ingestion-horizon) -- the hard, code-level pre-join filter:
    "how do you even know that" is an unrecoverable expose. ``witnessed=false``
    is therefore never voiceable. Entries with empty content are also rejected.
    """
    if not isinstance(entry, dict):
        return False
    if not _witnessed(entry):
        return False
    return bool((entry.get("content") or "").strip())


def _approx_tokens(text: str) -> float:
    return len(text.split()) * _TOKENS_PER_WORD


def _is_assertable(entry: dict) -> bool:
    """A bio_fact / stance is assertable to the brain only once corroborated."""
    return int(entry.get("corroboration_count") or 0) >= _ASSERTABLE_CORROBORATION


def _last_ref_key(entry: dict) -> tuple[int, Any]:
    """Sort key for least-recently-referenced first.

    Never-referenced entries (``last_referenced_at`` is None) sort *before* any
    referenced entry, so a fresh callback is preferred over re-using an old one.
    """
    ref = entry.get("last_referenced_at")
    if ref is None:
        return (0, 0)
    return (1, ref)


def render_decision_capsule(
    dossier: Any, display_name: str, max_tokens: int = 120
) -> str:
    """Render a <=``max_tokens`` DISPOSITION line for the DECISION BRAIN.

    Disposition, not a fact dump: leads with relationship stance, then the
    assertable facts/stances and running bits as *don't-bring-up-unless-relevant*
    guidance. Includes ``witnessed=false`` entries as SILENT context (allowed for
    the brain only). Clamped to the token budget (whitespace-word approximation).
    """
    entries = _entries(dossier)
    tone = (dossier or {}).get("tone") if isinstance(dossier, dict) else None
    tone = tone if isinstance(tone, dict) else {}

    parts: list[str] = []
    stage = tone.get("relationship_stage")
    if stage:
        parts.append(str(stage))
    banter = tone.get("banter_permission")
    if banter:
        parts.append(f"banter: {banter}")

    for entry in entries:
        content = (entry.get("content") or "").strip()
        if not content:
            continue
        etype = entry.get("type") or "note"
        # Facts/stances only enter as disposition once corroborated; jokes and
        # relational entries are dispositional by nature.
        if etype in ("bio_fact", "stance") and not _is_assertable(entry):
            continue
        tag = etype.replace("_", " ")
        marker = "" if _witnessed(entry) else " (silent: don't voice)"
        parts.append(f"{tag}: {content}{marker}")

    head = f"{display_name}:" if display_name else ""
    body = "; ".join(parts) if parts else "no read yet"
    line = f"{head} {body}".strip()

    # Clamp to the token budget by dropping trailing words.
    words = line.split()
    if not words:
        return line
    max_words = max(1, int(max_tokens / _TOKENS_PER_WORD))
    if len(words) > max_words:
        line = " ".join(words[:max_words])
    return line


def select_voice_tone(dossier: Any) -> ToneCapsule:
    """Map the dossier ``tone`` block to a ``ToneCapsule`` for the WORD-GENERATOR.

    HOW channel only: register / relationship_stage / address_form. Carries NO
    facts, stances, or jokes -- the generator physically cannot info-dump because
    it never receives the fact list. ``witnessed`` does not gate the tone block
    itself (tone is an aggregate disposition, not a voiced fact), but no entry
    content crosses into it.
    """
    tone = (dossier or {}).get("tone") if isinstance(dossier, dict) else None
    tone = tone if isinstance(tone, dict) else {}
    capsule = ToneCapsule()
    register = tone.get("register_tag")
    if register:
        capsule.register = str(register)
    stage = tone.get("relationship_stage")
    if stage:
        capsule.relationship_stage = str(stage)
    address = tone.get("address_form")
    capsule.address_form = str(address) if address else None
    return capsule


def select_callback(
    dossier: Any, budget_remaining: int, rng: random.Random
) -> str | None:
    """Pick the ONE specific entity/token to inject into the word-gen context.

    Returns ``None`` unless ``budget_remaining > 0`` AND there is a witnessed,
    voiceable, not-recently-used entry. Prefers running jokes / shared history,
    then picks the least-recently-referenced eligible entry (deterministic given
    ``rng`` for ties). This is the budgeted exception that lets a real callback
    through the otherwise-blunt wall.
    """
    if budget_remaining <= 0:
        return None
    eligible = [e for e in _entries(dossier) if _voiceable(e)]
    if not eligible:
        return None

    def pref_rank(entry: dict) -> int:
        etype = entry.get("type")
        try:
            return _CALLBACK_PREFERENCE.index(etype)
        except ValueError:
            return len(_CALLBACK_PREFERENCE)

    # Least-recently-referenced first, within the most-preferred type present.
    # rng breaks exact ties deterministically.
    chosen = min(
        eligible,
        key=lambda e: (pref_rank(e), _last_ref_key(e), rng.random()),
    )
    return (chosen.get("content") or "").strip() or None


def enforce_daily_budget(callbacks_used_today: int, daily_budget: int = 1) -> int:
    """budget_remaining = max(0, daily_budget - used). Hard ~1/day default."""
    return max(0, daily_budget - int(callbacks_used_today or 0))
