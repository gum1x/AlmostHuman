# When-to-Respond Timing Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Calibrate and turn on the already-built timing classifier so the bot speaks at a controlled, human-plausible rate on the unprompted firehose, while always engaging when addressed.

**Architecture:** The serve-time `TimingClassifier` and its scheduler pre-gate already exist (`conversation_engine/timing_classifier.py`, `scheduler.py:476-529`); they are flag-off on the v1 model. We (1) build an offline replay to pick the threshold, (2) add a log-only **shadow mode** to confirm the live rate before enforcing, (3) add a rate **monitor**, then (4) flip it on against the v2 model at the calibrated threshold. Addressed messages keep bypassing the classifier (always engage); the classifier governs only unprompted messages.

**Tech Stack:** Python 3.11, pure-numpy/stdlib logistic regression (no torch — VPS GPU is CPU-only sm_61), pytest (`asyncio_mode=auto`), structlog, SQLAlchemy async (Postgres). Models are JSON in `models/`.

## Global Constraints

- **No new ML deps / no neural nets** — the model is the existing JSON logreg; scoring is stdlib `math`. CPU-only.
- **train == serve feature parity** — the 11 features must match `scripts/build_timing_dataset.py` and `conversation_engine/timing_classifier.py` exactly. Do not redefine them; import/reuse.
- **Flag-gated, default OFF** — every new behavior is behind a config flag defaulting to today's behavior. Rollback = one env var.
- **Flagship chat id:** `-1002705709115`. Prod export: `data/prod_export/messages.jsonl` (git-excluded; fields: `message_id, sender_id, reply_to_message_id, text_cleaned, text_raw, is_deleted, timestamp`).
- **v2 model:** `models/timing_classifier_v2.json` (60 frozen regulars, `chosen_threshold=0.825`, isotonic calibration map, `target_response_rate=0.06`).
- **Tests:** offline only — no network, no DB, synthetic fixtures. Run with `PYTHONPATH=.` from repo root.

---

### Task 1: Offline rate-report / calibration tool

Replays the historical firehose through the model and prints the unprompted pass rate at a sweep of thresholds, so we can pick the calibrated threshold. No engine change.

**Files:**
- Create: `scripts/timing_rate_report.py`
- Test: `tests/unit/test_timing_rate_report.py`

**Interfaces:**
- Consumes: `conversation_engine.timing_classifier` (`MENTION/NUMBER/CLAIM/WH/BOTLIKE` regexes, `_len_bucket`, `_is_botlike`, `compute_regulars`) — reuse, do not re-derive.
- Produces:
  - `score_rows(model: dict, rows: list[dict]) -> list[float]` — prob per non-botlike message (botlike → score 0.0), in input order.
  - `rate_table(scores: list[float], thresholds: list[float]) -> list[tuple[float, float]]` — `(threshold, pass_rate)` pairs.
  - `main(argv=None) -> int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_timing_rate_report.py
from __future__ import annotations
import json, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import timing_rate_report as rr  # noqa: E402
from conversation_engine.timing_classifier import TimingClassifier  # noqa: E402


def _clf(tmp_path):
    """Real TimingClassifier loaded from a temp 1-feature model (weight on is_question)."""
    model = {
        "feature_order": ["is_mention", "is_reply", "reply_to_regular", "msg_len_words",
                          "msg_len_bucket", "has_number", "has_claim_token", "is_question",
                          "sender_is_regular", "idx_gap_since_sender", "is_botlike"],
        "weights": [0, 0, 0, 0, 0, 0, 0, 5.0, 0, 0, 0],
        "bias": -2.5,
        "feature_mean": [0]*11, "feature_std": [1]*11,
        "chosen_threshold": 0.5,
    }
    p = tmp_path / "m.json"
    p.write_text(json.dumps(model))
    return TimingClassifier(model_path=p)


def test_botlike_scores_zero(tmp_path):
    rows = [{"text": "/start", "is_reply": False, "reply_to_regular": False,
             "sender_is_regular": True, "idx_gap_since_sender": -1}]
    scores = rr.score_rows(_clf(tmp_path), rows)
    assert scores == [0.0]  # botlike "/..." forced to 0 by the real classifier


def test_question_scores_high(tmp_path):
    rows = [{"text": "anyone selling?", "is_reply": False, "reply_to_regular": False,
             "sender_is_regular": True, "idx_gap_since_sender": -1}]
    scores = rr.score_rows(_clf(tmp_path), rows)
    assert scores[0] > 0.9


def test_rate_table_counts_pass_fraction():
    scores = [0.9, 0.9, 0.1, 0.1]  # 2 of 4 above 0.5
    table = dict(rr.rate_table(scores, [0.5, 0.95]))
    assert table[0.5] == 0.5
    assert table[0.95] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/unit/test_timing_rate_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'timing_rate_report'`.

- [ ] **Step 3: Write the script**

```python
# scripts/timing_rate_report.py
#!/usr/bin/env python3
"""Replay the historical firehose through the timing model; print the unprompted
pass rate at a sweep of thresholds so we can pick the calibrated operating point.

Reuses the EXACT serve-time feature definitions from conversation_engine.timing_classifier
(train==serve). Offline, stdlib + the model JSON only. No DB, no network."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from conversation_engine.timing_classifier import (  # noqa: E402
    TimingClassifier, compute_regulars,
)

DEFAULT_SOURCE = "data/prod_export/messages.jsonl"
DEFAULT_MODEL = "models/timing_classifier_v2.json"
THRESHOLDS = [0.5, 0.6, 0.7, 0.745, 0.8, 0.825, 0.85, 0.9]


def score_rows(clf, rows):
    """Score each row's probability via the REAL serve-time classifier (train==serve —
    no re-derived feature/scoring math). rows: dicts with text/is_reply/reply_to_regular/
    sender_is_regular/idx_gap_since_sender. Botlike messages come back 0.0 (the classifier
    force-skips them)."""
    out = []
    for r in rows:
        ts = clf.score(
            text=r["text"],
            is_reply=r["is_reply"],
            reply_to_regular=r["reply_to_regular"],
            sender_is_regular=r["sender_is_regular"],
            idx_gap_since_sender=r["idx_gap_since_sender"],
        )
        out.append(ts.score)
    return out


def rate_table(scores, thresholds):
    n = len(scores) or 1
    return [(t, sum(1 for s in scores if s >= t) / n) for t in thresholds]


def _load_rows(source, regulars_override):
    """Single pass over the export, computing history features train==serve style."""
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
    sender_of = {}
    last_spoke = {}
    rows = []
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
    clf = TimingClassifier(model_path=Path(args.model))
    rows = _load_rows(args.source, clf.regulars)
    scores = score_rows(clf, rows)
    nonzero = [s for s in scores if s > 0.0]
    print(f"messages scored: {len(rows):,}  (non-botlike: {len(nonzero):,})")
    print(f"model chosen_threshold: {clf.threshold}")
    print(f"\n{'threshold':>10} {'unprompted pass rate':>22}")
    for t, rate in rate_table(scores, THRESHOLDS):
        print(f"{t:>10.3f} {rate:>21.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/unit/test_timing_rate_report.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Run the report on real data (manual, records the calibration input)**

Run: `PYTHONPATH=. python scripts/timing_rate_report.py`
Expected: a table of threshold → unprompted pass rate. **Record the threshold whose rate matches the chattier-leaning target** (compare against the model's 0.825/~6% default; lower threshold = chattier). This number feeds Task 4.

- [ ] **Step 6: Commit**

```bash
git add scripts/timing_rate_report.py tests/unit/test_timing_rate_report.py
git commit -m "feat(timing): offline rate-report to calibrate the unprompted threshold"
```

---

### Task 2: Shadow mode (log-only) in the scheduler

Lets the classifier score live traffic and log what it *would* do, without changing behavior — so we confirm the live rate matches Task 1's replay before enforcing.

**Files:**
- Modify: `conversation_engine/config.py:124-126` and `:195-199` (add the flag)
- Modify: `conversation_engine/scheduler.py:158-164` (instantiate when shadow too) and `:476-529` (score-and-log without skipping in shadow; attach telemetry to gate factors)
- Test: `tests/unit/test_timing_shadow.py`

**Interfaces:**
- Consumes: existing `TimingClassifier`, `history_feature_inputs`, `GateResult`.
- Produces: `EngineConfig.timing_classifier_shadow: bool`; scheduler behavior — in shadow, the timing pre-gate never returns early, and `gate.gate_factors` carries `timing_p`, `timing_would_pass`, `timing_is_direct` on every classifier-eligible cycle.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_timing_shadow.py
from __future__ import annotations
import os
from conversation_engine.config import load_engine_config


def test_shadow_flag_parses_from_env(monkeypatch):
    monkeypatch.setenv("TIMING_CLASSIFIER_SHADOW", "true")
    cfg = load_engine_config()
    assert cfg.timing_classifier_shadow is True


def test_shadow_flag_defaults_false(monkeypatch):
    monkeypatch.delenv("TIMING_CLASSIFIER_SHADOW", raising=False)
    cfg = load_engine_config()
    assert cfg.timing_classifier_shadow is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/unit/test_timing_shadow.py -v`
Expected: FAIL — `AttributeError: 'EngineConfig' object has no attribute 'timing_classifier_shadow'`.

- [ ] **Step 3: Add the config flag**

In `conversation_engine/config.py`, after line 126 (`timing_classifier_threshold`):

```python
    timing_classifier_shadow: bool = False  # score+log "would-fire" without acting (measure first)
```

In `load_engine_config(...)`, after the `timing_classifier_threshold=...` line (~199):

```python
        timing_classifier_shadow=os.getenv("TIMING_CLASSIFIER_SHADOW", "false").lower() == "true",
```

- [ ] **Step 4: Run config test to verify it passes**

Run: `PYTHONPATH=. pytest tests/unit/test_timing_shadow.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Extract the skip decision into a pure, testable helper (TDD)**

The skip-vs-proceed rule is the real logic worth testing. Extract it so it can be tested without a DB/scheduler. Add to `conversation_engine/timing_classifier.py` (module level, after `TimingScore`):

```python
def timing_should_skip(*, passes: bool, enforcing: bool) -> bool:
    """True only when the classifier rejected the message AND we are enforcing.
    In shadow mode (enforcing=False) we never skip — we only observe."""
    return (not passes) and enforcing
```

Write the failing test:

```python
# append to tests/unit/test_timing_shadow.py
from conversation_engine.timing_classifier import timing_should_skip


def test_enforcing_skips_on_reject():
    assert timing_should_skip(passes=False, enforcing=True) is True

def test_shadow_never_skips():
    assert timing_should_skip(passes=False, enforcing=False) is False

def test_pass_never_skips():
    assert timing_should_skip(passes=True, enforcing=True) is False
    assert timing_should_skip(passes=True, enforcing=False) is False
```

Run: `PYTHONPATH=. pytest tests/unit/test_timing_shadow.py -v` → FAIL (`ImportError: cannot import name 'timing_should_skip'`), then add the helper, re-run → PASS.

- [ ] **Step 6: Wire shadow + the helper into the scheduler**

In `scheduler.py:159`, load the classifier in shadow too:

```python
        if getattr(config, "timing_classifier_enabled", False) or getattr(
            config, "timing_classifier_shadow", False
        ):
```

Import the helper at the top alongside the existing timing imports (`scheduler.py:48-51`):

```python
from conversation_engine.timing_classifier import (
    TimingClassifier,
    history_feature_inputs,
    timing_should_skip,
)
```

In the timing pre-gate block (`scheduler.py:476-529`), after `ts = self.timing_classifier.score(...)` (~line 498): attach telemetry to the gate factors, compute `enforcing`, and gate the existing skip block on the helper:

```python
            gate = GateResult(
                gate_score=gate.gate_score,
                gate_factors={
                    **gate.gate_factors,
                    "timing_p": round(ts.score, 3),
                    "timing_would_pass": ts.passes,
                    "timing_is_direct": False,
                },
                should_proceed=gate.should_proceed,
            )
            enforcing = (
                self.config.timing_classifier_enabled
                and not self.config.timing_classifier_shadow
            )
            if timing_should_skip(passes=ts.passes, enforcing=enforcing):
                # ... existing skip block (insert_ai_decision + log + return) unchanged ...
            elif not ts.passes:
                await log.ainfo(
                    "timing_classifier_shadow",
                    chat_id=chat_id, p=round(ts.score, 3),
                    threshold=self.timing_classifier.threshold,
                    would_pass=ts.passes,
                    message_id=getattr(target_for_direct, "message_id", None),
                )
```

(The existing skip block at 499-529 moves under the `if timing_should_skip(...)`. In shadow, no early return — the cycle proceeds and records its normal decision, now carrying the `timing_*` factors.)

- [ ] **Step 7: Run all timing tests**

Run: `PYTHONPATH=. pytest tests/unit/test_timing_shadow.py tests/unit/test_timing_feature_parity.py -v`
Expected: PASS (all).

- [ ] **Step 8: Commit**

```bash
git add conversation_engine/config.py conversation_engine/scheduler.py conversation_engine/timing_classifier.py tests/unit/test_timing_shadow.py
git commit -m "feat(timing): add shadow mode — score+log would-fire without acting"
```

---

### Task 3: Live rate monitor

Reads recent recorded decisions and reports rolling unprompted / addressed / overall rates, exiting nonzero if the rate is out of band. Used to verify shadow vs replay, and as the post-enable alarm.

**Files:**
- Create: `scripts/timing_rate_monitor.py`
- Test: `tests/unit/test_timing_rate_monitor.py`

**Interfaces:**
- Consumes: rows of `{"timing_p": float, "timing_would_pass": bool, "timing_is_direct": bool, "should_respond": bool}` (the `gate_factors` + decision shape written by Task 2). The DB query is thin; the testable core is pure.
- Produces:
  - `summarize(rows: list[dict]) -> dict` → `{"n": int, "unprompted_pass_rate": float, "addressed_frac": float, "send_rate": float}`.
  - `check_band(summary: dict, lo: float, hi: float) -> tuple[bool, str]`.
  - `main(argv=None) -> int` (exit 0 in band, 1 out of band).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_timing_rate_monitor.py
from __future__ import annotations
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT)); sys.path.insert(0, str(REPO_ROOT / "scripts"))
import timing_rate_monitor as mon  # noqa: E402


def _row(p, would_pass, is_direct, sent):
    return {"timing_p": p, "timing_would_pass": would_pass,
            "timing_is_direct": is_direct, "should_respond": sent}


def test_summarize_splits_addressed_and_unprompted():
    rows = [_row(0.9, True, False, True), _row(0.1, False, False, False),
            _row(0.0, False, True, True), _row(0.0, False, True, True)]
    s = mon.summarize(rows)
    assert s["n"] == 4
    assert s["addressed_frac"] == 0.5            # 2 of 4 addressed
    assert s["unprompted_pass_rate"] == 0.5      # 1 of 2 unprompted passed
    assert s["send_rate"] == 0.75                # 3 of 4 sent


def test_check_band():
    ok, _ = mon.check_band({"unprompted_pass_rate": 0.08}, 0.05, 0.15)
    assert ok is True
    bad, msg = mon.check_band({"unprompted_pass_rate": 0.40}, 0.05, 0.15)
    assert bad is False and "0.40" in msg
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/unit/test_timing_rate_monitor.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'timing_rate_monitor'`.

- [ ] **Step 3: Write the monitor**

```python
# scripts/timing_rate_monitor.py
#!/usr/bin/env python3
"""Report rolling timing rates from recorded decisions; exit nonzero if the
unprompted pass rate leaves the alarm band. Pure aggregation core + a thin DB read."""
from __future__ import annotations
import argparse, sys


def summarize(rows):
    n = len(rows)
    if n == 0:
        return {"n": 0, "unprompted_pass_rate": 0.0, "addressed_frac": 0.0, "send_rate": 0.0}
    addressed = [r for r in rows if r.get("timing_is_direct")]
    unprompted = [r for r in rows if not r.get("timing_is_direct")]
    up_pass = sum(1 for r in unprompted if r.get("timing_would_pass")) / (len(unprompted) or 1)
    sent = sum(1 for r in rows if r.get("should_respond")) / n
    return {"n": n, "unprompted_pass_rate": up_pass,
            "addressed_frac": len(addressed) / n, "send_rate": sent}


def check_band(summary, lo, hi):
    r = summary["unprompted_pass_rate"]
    if lo <= r <= hi:
        return True, f"unprompted pass rate {r:.2%} in band [{lo:.0%}, {hi:.0%}]"
    return False, f"unprompted pass rate {r:.2f} OUT OF BAND [{lo:.2f}, {hi:.2f}]"


async def _load_rows(hours):
    """Read recent AiDecision rows via the engine's memory manager. Imported lazily so
    the pure core (above) stays testable without a DB."""
    from conversation_engine.config import load_engine_config
    from conversation_engine.memory_manager import ConversationMemoryManager
    cfg = load_engine_config()
    mm = ConversationMemoryManager(cfg)
    return await mm.recent_timing_decisions(hours=hours)  # returns dicts as in summarize()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=168.0)
    ap.add_argument("--lo", type=float, default=0.05)
    ap.add_argument("--hi", type=float, default=0.15)
    args = ap.parse_args(argv)
    import asyncio
    rows = asyncio.run(_load_rows(args.hours))
    s = summarize(rows)
    print(f"n={s['n']}  unprompted_pass={s['unprompted_pass_rate']:.2%}  "
          f"addressed={s['addressed_frac']:.2%}  send_rate={s['send_rate']:.2%}")
    ok, msg = check_band(s, args.lo, args.hi)
    print(msg)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/unit/test_timing_rate_monitor.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Add the `recent_timing_decisions` read to the memory manager**

In `conversation_engine/memory_manager.py`, add this method (mirrors `get_avg_feedback_score`'s
session/query style; `select`, `func`, `utcnow`, `timedelta`, and `AiDecision` are already imported
there). Read-only; no schema change — `gate_factors` is existing JSONB:

```python
    async def recent_timing_decisions(self, hours: float = 168.0) -> list[dict]:
        """Recent AiDecision rows carrying timing telemetry, projected into the shape
        scripts/timing_rate_monitor.summarize() expects. Returns [] if none."""
        since = utcnow() - timedelta(hours=hours)
        result = await self.session.execute(
            select(AiDecision.gate_factors, AiDecision.should_respond).where(
                AiDecision.evaluated_at >= since,
                AiDecision.gate_factors.has_key("timing_p"),  # JSONB ? operator (Postgres)
            )
        )
        rows = []
        for gate_factors, should_respond in result.all():
            gf = gate_factors or {}
            rows.append({
                "timing_p": gf.get("timing_p"),
                "timing_would_pass": gf.get("timing_would_pass"),
                "timing_is_direct": gf.get("timing_is_direct"),
                "should_respond": should_respond,
            })
        return rows
```

This DB read needs Postgres, so it isn't unit-tested here; its output shape is the same dict the
`summarize()` tests already pin, and it's exercised live in Task 4 Step 4 (the monitor run). If
`utcnow`/`timedelta` are not already imported in this file, add them (check the existing imports).

- [ ] **Step 6: Commit**

```bash
git add scripts/timing_rate_monitor.py tests/unit/test_timing_rate_monitor.py conversation_engine/memory_manager.py
git commit -m "feat(timing): rolling rate monitor with alarm band"
```

---

### Task 4: Calibrate, shadow-verify, enable (ops — VPS, owner-run)

No code. Deploy + verify, gated and reversible. Owner runs VPS commands (stage-only).

- [ ] **Step 1: Pick the threshold** from Task 1's table — the value whose unprompted pass rate matches the chattier-leaning target. Record it in the spec's open-items and as `T`.

- [ ] **Step 2: Deploy code** — rsync the three changed files + scripts to the VPS, rebuild the engine:
  `rsync -avz -R conversation_engine/config.py conversation_engine/scheduler.py conversation_engine/memory_manager.py scripts/timing_rate_report.py scripts/timing_rate_monitor.py vps:/home/x/Research/` then `ssh vps "cd /home/x/Research && docker compose up -d --build conversation-engine"`.

- [ ] **Step 3: Shadow first** — VPS `.env`: `TIMING_CLASSIFIER_MODEL_PATH=models/timing_classifier_v2.json`, `TIMING_CLASSIFIER_SHADOW=true`, `TIMING_CLASSIFIER_ENABLED=false`, `TIMING_CLASSIFIER_THRESHOLD=T`. Recreate the engine. No behavior change.

- [ ] **Step 4: Measure (after ~2-3 days)** — `PYTHONPATH=. python scripts/timing_rate_monitor.py --hours 72`. Confirm the live unprompted pass rate ≈ Task 1's replay prediction at `T`. Re-pick `T` if they diverge.

- [ ] **Step 5: Enforce** — VPS `.env`: `TIMING_CLASSIFIER_ENABLED=true`, `TIMING_CLASSIFIER_SHADOW=false`. Recreate the engine. The classifier now gates the unprompted firehose; addressed messages still bypass.

- [ ] **Step 6: Verify live + rollback drill** — run the monitor over the next window; confirm the rate holds in band and addressed messages still always engage. Confirm rollback: set `TIMING_CLASSIFIER_ENABLED=false`, recreate, behavior returns to today's.

---

## Self-Review

**1. Spec coverage:**
- §3 use trained classifier → Task 4 (v2 model, enable). ✓
- §4 addressed=always-engage → no change (existing bypass), asserted in Task 2 telemetry (`timing_is_direct`). ✓
- §4 unprompted selective + monitored ceiling → Task 1 (calibrate), Task 3 (monitor). ✓
- §5 replace-not-stack → the classifier remains the firehose pre-gate; the hand gate stays only as a secondary/ safety check (unchanged). *Note:* the spec's stronger "retire the 9 hand weights" is **not** done here — deferred, since the gate currently also serves direct-mention force-proceed and DM logic; ripping it out is a separate change. Flagged as a follow-up, not silently dropped.
- §6 shadow→measure→calibrate→enforce → Tasks 1-4 map 1:1. ✓
- §7 offline eval → Task 1 replay. ✓
- §8 safety/rollback → Task 4 Step 6 rollback drill; guardrails untouched. ✓
- §9 phasing → Tasks ordered shadow→enforce. ✓

**2. Placeholder scan:** Clean after the pre-flight fixes (2026-06-16). Task 1 now reuses the real
`TimingClassifier` (no re-derived scoring math — satisfies the train==serve constraint); Task 2's
skip logic is an extracted `timing_should_skip` helper with real unit tests (no self-referential
test); Task 3 Step 5 has complete code. The only un-unit-tested code is the Task 3 DB read (needs
Postgres), verified live in Task 4.

**3. Type consistency:** `score_rows`/`rate_table` (Task 1), `summarize`/`check_band` (Task 3) signatures match their tests. The `gate_factors` keys written in Task 2 (`timing_p`, `timing_would_pass`, `timing_is_direct`) are exactly the keys `summarize` reads in Task 3. ✓

## Open follow-ups (not in this plan)
- Retire the 9 hand-tuned gate weights entirely (spec §5 full version) once the classifier is trusted.
- Persona-specific retrain on `8384923892`'s reply decisions (spec §9 fast-follow) if the room-engageability proxy isn't selective enough.
