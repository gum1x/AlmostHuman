#!/usr/bin/env python3
"""Replay the historical firehose through the timing model; print the unprompted
pass rate at a sweep of thresholds so we can pick the calibrated operating point.

Reuses the EXACT serve-time scorer (conversation_engine.timing_classifier.TimingClassifier)
so train==serve — no re-derived feature math. Offline; stdlib + the model JSON only."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Mock external dependencies to avoid import bloat in offline script
class MockLogger:
    def info(self, *args, **kwargs): pass
    def warning(self, *args, **kwargs): pass

class MockStructlog:
    @staticmethod
    def get_logger():
        return MockLogger()

sys.modules["structlog"] = MockStructlog()

from conversation_engine.timing_classifier import TimingClassifier, compute_regulars  # noqa: E402

DEFAULT_SOURCE = "data/prod_export/messages.jsonl"
DEFAULT_MODEL = "models/timing_classifier_v2.json"
THRESHOLDS = [0.5, 0.6, 0.7, 0.745, 0.8, 0.825, 0.85, 0.9]


def score_rows(classifier, rows):
    """rows: dicts with text/is_reply/reply_to_regular/sender_is_regular/idx_gap_since_sender.
    Returns the model probability per row (botlike => 0.0), via the real serve scorer."""
    return [
        classifier.score(
            text=r["text"],
            is_reply=r["is_reply"],
            reply_to_regular=r["reply_to_regular"],
            sender_is_regular=r["sender_is_regular"],
            idx_gap_since_sender=r["idx_gap_since_sender"],
        ).score
        for r in rows
    ]


def rate_table(scores, thresholds):
    n = len(scores) or 1
    return [(t, sum(1 for s in scores if s >= t) / n) for t in thresholds]


def _load_rows(source, regulars_override):
    """Single pass over the export, computing history features train==serve style
    (regulars = top-K active senders; idx gap = message-index gap since sender last spoke)."""
    raw = []
    for line in open(source):
        try:
            m = json.loads(line)
        except json.JSONDecodeError:
            continue
        if m.get("is_deleted") or m.get("sender_id") is None:
            continue
        text = (m.get("text_cleaned") or m.get("text_raw") or "").strip()
        if not text:
            continue
        raw.append({"id": m.get("message_id"), "sender": m["sender_id"],
                    "parent": m.get("reply_to_message_id"), "text": text})
    regulars = regulars_override or compute_regulars(r["sender"] for r in raw)
    sender_of, last_spoke, rows = {}, {}, []
    for i, r in enumerate(raw):
        parent = r["parent"]
        gap = i - last_spoke[r["sender"]] if r["sender"] in last_spoke else -1
        rows.append({
            "text": r["text"],
            "is_reply": parent is not None,
            "reply_to_regular": bool(parent and sender_of.get(parent) in regulars),
            "sender_is_regular": r["sender"] in regulars,
            "idx_gap_since_sender": gap,
        })
        if r["id"] is not None:
            sender_of[r["id"]] = r["sender"]
        last_spoke[r["sender"]] = i
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args(argv)
    classifier = TimingClassifier(model_path=Path(args.model))
    rows = _load_rows(args.source, classifier.regulars)
    scores = score_rows(classifier, rows)
    nonzero = [s for s in scores if s > 0.0]
    print(f"messages scored: {len(rows):,}  (non-botlike: {len(nonzero):,})")
    print(f"model threshold: {classifier.threshold}")
    print(f"\n{'threshold':>10} {'unprompted pass rate':>22}")
    for t, rate in rate_table(scores, THRESHOLDS):
        print(f"{t:>10.3f} {rate:>21.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
