"""Pure dossier data model for the per-person memory redesign.

No DB, no I/O — just dataclasses plus (de)serialization for the flat JSONB
dossier shape and the corroboration-ladder write policy from
``.plans/voice-memory-redesign.md``.

The corroboration ladder beats a sarcasm classifier in this register:
everything enters as a low-confidence *claim*; a claim only becomes an
assertable *fact* once it has been corroborated on multiple *distinct days*
(different-days rule). Single-source / same-day-repeat material never gets
promoted, and anything not actually witnessed by the account (evidence before
the ingestion horizon) is never voiced — only used as silent tone context.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

# Allowed dossier entry types (flat blob — no knowledge graph).
ENTRY_TYPES = {
    "bio_fact",
    "stance",
    "running_joke",
    "nickname",
    "bag_held",
    "history_with_me",
}

# Types that may ever be asserted as a fact (running_joke is deliberately excluded:
# it is voiced as a joke, never typed as a fact).
_ASSERTABLE_FACT_TYPES = {
    "bio_fact",
    "stance",
    "nickname",
    "bag_held",
    "history_with_me",
}


@dataclass
class DossierEntry:
    type: str
    content: str
    confidence: float = 0.0
    corroboration_count: int = 1
    evidence_msg_ids: list[int] = field(default_factory=list)
    t_valid: str | None = None
    t_invalid: str | None = None
    witnessed: bool = False
    last_referenced_at: str | None = None
    use_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DossierEntry":
        return cls(
            type=data.get("type"),
            content=data.get("content", ""),
            confidence=float(data.get("confidence", 0.0)),
            corroboration_count=int(data.get("corroboration_count", 1)),
            evidence_msg_ids=[int(x) for x in (data.get("evidence_msg_ids") or [])],
            t_valid=data.get("t_valid"),
            t_invalid=data.get("t_invalid"),
            witnessed=bool(data.get("witnessed", False)),
            last_referenced_at=data.get("last_referenced_at"),
            use_count=int(data.get("use_count", 0)),
        )


@dataclass
class ToneCapsuleData:
    register_tag: str | None = None
    banter_permission: bool = False
    address_form: str | None = None
    relationship_stage: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "ToneCapsuleData":
        data = data or {}
        return cls(
            register_tag=data.get("register_tag"),
            banter_permission=bool(data.get("banter_permission", False)),
            address_form=data.get("address_form"),
            relationship_stage=data.get("relationship_stage"),
        )


@dataclass
class Dossier:
    entries: list[DossierEntry] = field(default_factory=list)
    tone: ToneCapsuleData = field(default_factory=ToneCapsuleData)
    aliases: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "entries": [e.to_dict() for e in self.entries],
            "tone": self.tone.to_dict(),
            "aliases": list(self.aliases),
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "Dossier":
        """Load a dossier from a plain dict; drop unknown entry types, never crash."""
        data = data or {}
        entries: list[DossierEntry] = []
        for raw in data.get("entries") or []:
            if not isinstance(raw, dict):
                continue
            if raw.get("type") not in ENTRY_TYPES:
                continue
            entries.append(DossierEntry.from_dict(raw))
        return cls(
            entries=entries,
            tone=ToneCapsuleData.from_dict(data.get("tone")),
            aliases=[str(a) for a in (data.get("aliases") or [])],
        )


def _normalize(content: str) -> str:
    """Lowercased, whitespace-collapsed content key for matching claims."""
    return " ".join((content or "").lower().split())


def _evidence_day(ts: str | None) -> str | None:
    """Day key (YYYY-MM-DD) from an ISO-ish timestamp string; None if unusable."""
    if not ts:
        return None
    return str(ts).strip()[:10] or None


def is_assertable(entry: DossierEntry, min_corroboration: int = 2) -> bool:
    """True only when the entry may be voiced AS A FACT.

    Requires: witnessed by the account, corroborated on at least
    ``min_corroboration`` distinct days, and a fact-type (running_joke is never
    assertable as a fact — it is voiced as a joke via ``is_voiceable_joke``).
    """
    if not entry.witnessed:
        return False
    if entry.corroboration_count < min_corroboration:
        return False
    return entry.type in _ASSERTABLE_FACT_TYPES


def is_voiceable_joke(entry: DossierEntry) -> bool:
    """True when a running_joke may be voiced AS A JOKE (witnessed + corroborated)."""
    return (
        entry.type == "running_joke"
        and entry.witnessed
        and entry.corroboration_count >= 2
    )


def corroborate(
    entries: list[DossierEntry],
    new_claim: DossierEntry,
    day_of: str,
) -> list[DossierEntry]:
    """Apply the corroboration ladder for ``new_claim`` observed on ``day_of``.

    Matching is by (same type, normalized content). The different-days rule:
    ``corroboration_count`` is bumped ONLY when the new evidence lands on a day
    that is distinct from every day already recorded for the matched entry —
    same-day repeats add evidence ids (provenance) but do NOT promote the claim.
    If no entry matches, the claim is appended fresh as a low-confidence claim.

    The set of distinct days seen for an entry is tracked on a transient
    ``_seen_days`` attribute (seeded from ``t_valid`` so it survives a reload),
    keeping the day-set out of the persisted JSONB schema. Mutates and returns
    ``entries``.
    """
    new_day = _evidence_day(day_of)
    key = (new_claim.type, _normalize(new_claim.content))
    for entry in entries:
        if (entry.type, _normalize(entry.content)) != key:
            continue
        seen_days = _seen_days(entry)
        # Always record the new evidence ids (provenance).
        for mid in new_claim.evidence_msg_ids:
            if mid not in entry.evidence_msg_ids:
                entry.evidence_msg_ids.append(mid)
        if new_day is not None and new_day not in seen_days:
            # New distinct day -> promote one rung up the ladder.
            seen_days.add(new_day)
            entry.corroboration_count += 1
            entry.confidence = min(1.0, entry.confidence + 0.25)
        _set_seen_days(entry, seen_days)
        return entries

    # No match: append as a fresh low-confidence claim.
    fresh = DossierEntry.from_dict(new_claim.to_dict())
    if fresh.corroboration_count < 1:
        fresh.corroboration_count = 1
    _set_seen_days(fresh, {new_day} if new_day else set())
    entries.append(fresh)
    return entries


# Distinct-day provenance is tracked on a transient attribute so it survives a
# corroborate() chain without bloating the persisted JSONB schema. It is seeded
# from t_valid (the first witnessed day) when absent.
def _seen_days(entry: DossierEntry) -> set[str]:
    days = getattr(entry, "_seen_days", None)
    if days is None:
        days = set()
        first = _evidence_day(entry.t_valid)
        if first:
            days.add(first)
    return days


def _set_seen_days(entry: DossierEntry, days: set[str]) -> None:
    object.__setattr__(entry, "_seen_days", set(days))
    if days and entry.t_valid is None:
        entry.t_valid = min(days)
