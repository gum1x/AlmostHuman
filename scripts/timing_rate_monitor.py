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
