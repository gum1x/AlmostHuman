#!/usr/bin/env python3
"""Build the timing-classifier training table from a raw messages JSONL export.

Reads a messages export (one JSON object per line, oldest -> newest preferred but
re-sorted by timestamp/message_id defensively) with at least:

    message_id, chat_id, sender_id, text, reply_to_message_id, timestamp

and emits one feature row per *regular*-authored, non-empty message, with the 11
serving features (in conversation_engine.timing_classifier feature_order) plus the
binary label "a later message in the export replied to this one" (a real member
bothered to respond).

This is the TRAIN side of the train==serve contract. Every feature here MUST match
conversation_engine/timing_classifier.py exactly (regexes copied verbatim below;
history features mirror compute_regulars / history_feature_inputs):

  is_mention, is_reply, reply_to_regular, msg_len_words, msg_len_bucket, has_number,
  has_claim_token, is_question, sender_is_regular, idx_gap_since_sender, is_botlike

Definitions (kept in lockstep with timing_classifier.py and the parity test):
  regulars             top-K most active senders by message count (--regular-top-k,
                       default 60), counted AFTER dropping empty-text rows.
  reply_to_regular     parent message is in the export AND its author is a regular.
  sender_is_regular    sender in regulars.
  idx_gap_since_sender i - last_spoke_idx[sender] over the kept rows (last_spoke_idx
                       updated for EVERY sender, regular or not); -1 if not seen.
  label                1 if any later kept message replies to this message_id, else 0.

Only regular-authored rows are emitted (matches serving, which only scores regulars);
non-regular rows still advance the index bookkeeping so gaps line up with serving.

Usage:
    python scripts/build_timing_dataset.py \
        --source data/prod_export/messages.jsonl \
        --out data/timing_dataset.jsonl \
        --regular-top-k 60
    # then feed --out to scripts/train_timing_classifier.py
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

# --- feature regexes: copied verbatim into conversation_engine/timing_classifier.py ---
MENTION = re.compile(r"@[A-Za-z0-9_]{3,}")
NUMBER = re.compile(r"\d")
CLAIM = re.compile(
    r"\b(larp|sold|buy|sell|selling|buying|scam|\$|\d+k|\d+\$|price|paid|offer|wts|wtb|cop|copped)\b",
    re.I,
)
WH = re.compile(
    r"^(who|what|when|where|why|how|which|anyone|any1|anybody|does|is|are|can|could|should)\b",
    re.I,
)
BOTLIKE = re.compile(
    r"(🛍|Sold ✅|Price:.*🪙|Off-chain|Marketapp|^/[a-z]|🎲|👑|⚠️|🔈|Rent\. Gifts|"
    r"✅️\s*$|joined the group|left the group|💬\s*ban|🔈\s*mut)",
    re.I,
)

FEATURE_ORDER = [
    "is_mention",
    "is_reply",
    "reply_to_regular",
    "msg_len_words",
    "msg_len_bucket",
    "has_number",
    "has_claim_token",
    "is_question",
    "sender_is_regular",
    "idx_gap_since_sender",
    "is_botlike",
]


def _len_bucket(wc: int) -> int:
    if wc <= 1:
        return 0
    if wc <= 3:
        return 1
    if wc <= 6:
        return 2
    if wc <= 12:
        return 3
    return 4


def _is_botlike(text: str) -> bool:
    if BOTLIKE.search(text):
        return True
    if text.count("\n") >= 3 and len(text) > 120:
        return True
    return False


def _text_of(m: dict) -> str:
    """Export field tolerance: prefer `text`, fall back to text_cleaned/text_raw."""
    for key in ("text", "text_cleaned", "text_raw"):
        val = m.get(key)
        if val:
            return str(val)
    return ""


def text_features(text: str) -> dict:
    """The text-derived features, byte-identical to TimingClassifier._features."""
    wc = len(text.split())
    return {
        "is_mention": int(bool(MENTION.search(text))),
        "msg_len_words": wc,
        "msg_len_bucket": _len_bucket(wc),
        "has_number": int(bool(NUMBER.search(text))),
        "has_claim_token": int(bool(CLAIM.search(text))),
        "is_question": int(bool(text.rstrip().endswith("?") or WH.search(text))),
        "is_botlike": int(_is_botlike(text)),
    }


def load_ordered(source: str | Path) -> list[dict]:
    """Load the export, drop deleted/empty-text/no-sender rows, sort oldest->newest.

    Empty-text messages are dropped BEFORE any counting/indexing so the regulars
    population and idx gaps match serving (timing_classifier history features filter
    the same way). Each kept row is normalised to: id, sender, parent, text.
    """
    rows: list[dict] = []
    with open(source, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                m = json.loads(line)
            except json.JSONDecodeError:
                continue
            if m.get("is_deleted") or m.get("sender_id") is None:
                continue
            text = _text_of(m).strip()
            if not text:
                continue
            rows.append(
                {
                    "id": m.get("message_id"),
                    "sender": m["sender_id"],
                    "parent": m.get("reply_to_message_id"),
                    "text": text,
                    "ts": m.get("timestamp"),
                }
            )
    # Stable oldest->newest: timestamp first, message_id as tiebreaker.
    rows.sort(key=lambda r: ((r["ts"] is None, r["ts"]), (r["id"] is None, r["id"])))
    return rows


def compute_regulars(sender_ids: Iterable[int | None], top_k: int) -> set[int]:
    """Top-K most active senders by message count (training's 'regulars')."""
    counts: dict[int, int] = defaultdict(int)
    for sender in sender_ids:
        if sender:
            counts[sender] += 1
    return {u for u, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_k]}


def build_rows(ordered: list[dict], top_k: int) -> tuple[list[dict], set[int]]:
    """Emit one labeled feature row per regular-authored message.

    Mirrors timing_classifier serving: regulars = top-K active senders; last_spoke_idx
    updated for every sender (regulars and non-regulars alike); idx_gap = i - last_idx;
    reply_to_regular = parent in export and its author is a regular. The label looks
    FORWARD only (later messages replying to this id) — no future leakage into features.
    """
    regulars = compute_regulars((r["sender"] for r in ordered), top_k)
    sender_of = {r["id"]: r["sender"] for r in ordered if r["id"] is not None}
    # Forward-looking label: which message_ids were replied to by a later kept message.
    replied_to: set = {r["parent"] for r in ordered if r["parent"] is not None}

    last_spoke_idx: dict[int, int] = {}
    out: list[dict] = []
    for i, r in enumerate(ordered):
        sender = r["sender"]
        if sender not in regulars:
            if sender is not None:
                last_spoke_idx[sender] = i
            continue
        gap = i - last_spoke_idx[sender] if sender in last_spoke_idx else -1
        if sender is not None:
            last_spoke_idx[sender] = i
        parent = r["parent"]
        feats = text_features(r["text"])
        feats.update(
            {
                "is_reply": int(parent is not None),
                "reply_to_regular": int(parent in sender_of and sender_of.get(parent) in regulars)
                if parent
                else 0,
                "sender_is_regular": int(sender in regulars),
                "idx_gap_since_sender": gap,
            }
        )
        label = int(r["id"] is not None and r["id"] in replied_to)
        out.append(
            {
                "message_id": r["id"],
                "features": {k: feats[k] for k in FEATURE_ORDER},
                "label": label,
            }
        )
    return out, regulars


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--source", required=True, help="messages JSONL export (oldest->newest)")
    ap.add_argument("--out", required=True, help="destination JSONL of labeled feature rows")
    ap.add_argument(
        "--regular-top-k",
        type=int,
        default=60,
        help="top-K most active senders treated as 'regulars' (default 60)",
    )
    args = ap.parse_args(argv)

    ordered = load_ordered(args.source)
    rows, regulars = build_rows(ordered, args.regular_top_k)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        # Header line: metadata the trainer reuses (frozen regulars, feature order, top-k).
        fh.write(
            json.dumps(
                {
                    "_meta": {
                        "feature_order": FEATURE_ORDER,
                        "regular_top_k": args.regular_top_k,
                        "regulars": sorted(regulars),
                        "label_key": "label",
                        "n_rows": len(rows),
                    }
                }
            )
            + "\n"
        )
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    pos = sum(r["label"] for r in rows)
    print(
        f"kept {len(ordered):,} ordered messages -> {len(rows):,} regular rows "
        f"({pos:,} positive, {pos / (len(rows) or 1):.2%}); "
        f"{len(regulars)} regulars; wrote {out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
