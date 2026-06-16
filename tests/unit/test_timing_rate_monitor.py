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
