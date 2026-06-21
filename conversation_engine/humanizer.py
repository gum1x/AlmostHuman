"""Timing / circadian humanizer (pure, deterministic, no I/O).

The bot's worst behavioral tells are *temporal*, not lexical. Against the
regular-member band (measured from real chat history):

  * reply latency p50: bot 39s vs regular band [13.9, 30.1]s  -> bot too SLOW
  * latency p90:       bot 144.3s, regular band [51.8, 286.44]s (in band)
  * share replied >1h: bot 0.0,   regular band [0.0, 0.01]    -> bot NEVER late
  * the bot is online 24/7 (no human sleep gap), a glaring non-human signature.

So this module produces:
  * a per-reply send delay drawn from a length-scaled log-normal calibrated to
    land p50 in the regular band (~20s) with a fat right tail (the occasional
    "saw it late, replied minutes later" and, rarely, "replied hours later"),
  * a mandatory daily dead window (>=6h of "sleep") whose edges jitter day to
    day so it is not a fixed clock signature,
  * a self-monitor that flags when the bot's emitted latencies have collapsed
    to a too-uniform distribution (a giveaway of a fixed timer).

Nothing here sleeps or reads a clock on its own. Delays are returned as VALUES
to be awaited outside the send transaction; every time-dependent function takes
``now``/``date`` as an argument.
"""

from __future__ import annotations

import math
import random
from datetime import date, datetime, time, timedelta, timezone

# ---------------------------------------------------------------------------
# Send-delay constants (log-normal reply latency, in seconds)
# ---------------------------------------------------------------------------
# We model reply latency as log-normal because human reply gaps are heavy-tailed
# and strictly positive: most replies are quick, a long tail of "saw it late".
# For a log-normal, median == exp(mu). The regular p50 band is [13.9, 30.1]s, so
# we aim the *median* of a typical-length message at ~20s -> mu = ln(20).
#   median ~= exp(2.996) ~= 20.0s
# sigma controls tail fatness. With sigma ~= 0.85 the p90 multiplier is
# exp(1.2816 * 0.85) ~= 2.96, i.e. a typical p90 near 20 * 2.96 ~= 59s, which
# sits inside the regular p90 band [51.8, 286.44]s while keeping p50 in band.
_DELAY_MU = math.log(20.0)  # median ~20s for a "typical" length message
_DELAY_SIGMA = 0.85  # tail fatness; tuned to keep p90 in band

# Length scaling: a regular skims a one-word reply fast and takes longer to
# read+type a long one. We nudge mu by message length around a reference length
# (donor word_count p50 = 3). Short msgs => slightly faster, long => slower.
_REF_WORD_COUNT = 3.0
_LEN_MU_GAIN = 0.18  # ln-seconds added per ln(words/ref); gentle
_LEN_MU_CLAMP = 0.7  # cap the length nudge so it can't dominate

# Normal-range clamp. 1s floor (no instant bot reflexes); 1800s (30min) ceiling
# for the ordinary distribution.
_DELAY_FLOOR = 1.0
_DELAY_CEIL = 1800.0

# Rare long tail: with small probability, mimic "replied a while later". The
# regular share replied >1h is tiny ([0.0, 0.01]), so the tail must be sparse
# AND mostly land in the 30min-1h band, with only a sliver crossing 1h. We draw
# uniformly on [1801, 5400]s (30min-90min): with _LONG_TAIL_PROB=0.015 the
# empirical share>30min is ~0.015 and share>1h is ~0.0075, both in band.
_LONG_TAIL_PROB = 0.015
_LONG_TAIL_MIN = 1801.0
_LONG_TAIL_MAX = 5400.0  # 90min; most tail mass stays under 1h

# ---------------------------------------------------------------------------
# Dead-window (daily "sleep") constants
# ---------------------------------------------------------------------------
# The bot must go dark for a human-length block each day. Base window is a
# ~7h block in UTC overlapping the donor's quietest hours; the regular's active
# hours show a clear overnight lull. >=6h is enforced after jitter.
_DEAD_BASE_START_HOUR = 3.0  # ~03:00 UTC nominal "bedtime"
_DEAD_BASE_END_HOUR = 10.0  # ~10:00 UTC nominal "wake" (7h block)
# Per-date jitter of +/- ~1.5h on the START so the edge is never a fixed clock
# tick. The END is derived from start + a (jittered) duration that stays >=6h.
_DEAD_START_JITTER_H = 1.5
_DEAD_DURATION_MIN_H = 6.0  # hard floor on sleep length
_DEAD_DURATION_BASE_H = _DEAD_BASE_END_HOUR - _DEAD_BASE_START_HOUR  # 7.0
_DEAD_DURATION_JITTER_H = 0.75  # +/- on duration, never dropping below the floor

# ---------------------------------------------------------------------------
# Latency self-monitor constants
# ---------------------------------------------------------------------------
# A fixed/near-fixed timer collapses latency variance. Real regulars have a
# heavy-tailed spread: coefficient of variation (std/mean) well above ~0.5 and
# a non-trivial IQR. We alarm when the emitted distribution is too uniform.
_CV_MIN_HEALTHY = 0.45  # below this CV => suspiciously uniform
_IQR_RATIO_MIN_HEALTHY = 0.20  # (p75-p25)/p50 below this => too tight
_ALARM_MIN_SAMPLES = 8  # need enough samples to judge


# ---------------------------------------------------------------------------
# Send delay
# ---------------------------------------------------------------------------


def _word_count(text: str) -> int:
    return len((text or "").split())


def compute_send_delay(
    text: str,
    rng: random.Random,
    *,
    intent_tag: str | None = None,
) -> float:
    """Return a humanized reply delay in SECONDS for ``text``.

    Drawn from a length-scaled log-normal calibrated to the regular latency
    band (p50 ~20s, fat right tail). Short messages resolve faster, long ones
    slower. The bulk is clamped to ``[1.0, 1800.0]``; with probability
    ``_LONG_TAIL_PROB`` a rare "replied a while later" draw up to 90min is
    allowed (sparse enough to keep the share replied >1h inside the band).

    ``intent_tag`` is accepted for forward-compat (Phase 3) and currently does
    not alter timing; defaults to ``None``.

    Pure: all randomness flows through the injected ``rng``.
    """
    # Rare long tail first: a sparse "saw it hours later" reply.
    if rng.random() < _LONG_TAIL_PROB:
        return rng.uniform(_LONG_TAIL_MIN, _LONG_TAIL_MAX)

    words = max(1, _word_count(text))
    # Length nudge to mu: ln-scaled around the donor reference length, clamped
    # so a very long message can't blow past the normal ceiling on its own.
    len_nudge = _LEN_MU_GAIN * math.log(words / _REF_WORD_COUNT)
    len_nudge = max(-_LEN_MU_CLAMP, min(_LEN_MU_CLAMP, len_nudge))
    mu = _DELAY_MU + len_nudge

    delay = rng.lognormvariate(mu, _DELAY_SIGMA)
    return max(_DELAY_FLOOR, min(_DELAY_CEIL, delay))


# ---------------------------------------------------------------------------
# Dead window (daily sleep)
# ---------------------------------------------------------------------------


def _date_jitter(d: date, seed: int, *, salt: int) -> float:
    """Deterministic [-1, 1] jitter for a (date, seed) pair.

    Uses the date's proleptic ordinal so it is stable for a fixed date+seed but
    varies day to day. ``salt`` separates independent jitters (start vs duration).
    """
    # Combine into a single deterministic int seed (random.Random rejects tuples
    # on this runtime). Large multipliers de-correlate adjacent dates/salts.
    combined = (seed * 1_000_003) ^ (d.toordinal() * 97) ^ (salt * 1_000_000_007)
    rng = random.Random(combined)
    return rng.uniform(-1.0, 1.0)


def dead_window_for_date(d: date, seed: int) -> tuple[float, float]:
    """Resolve the (start_hour, end_hour) of the dead window for a UTC date.

    Both are fractional UTC hours. ``end_hour`` may exceed 24 to denote a
    window crossing midnight into the next day; callers normalize with mod 24.
    The window is always >= ``_DEAD_DURATION_MIN_H`` hours long.
    """
    start = _DEAD_BASE_START_HOUR + _DEAD_START_JITTER_H * _date_jitter(d, seed, salt=1)
    duration = _DEAD_DURATION_BASE_H + _DEAD_DURATION_JITTER_H * _date_jitter(d, seed, salt=2)
    duration = max(_DEAD_DURATION_MIN_H, duration)
    return start, start + duration


def _window_interval(d: date, seed: int) -> tuple[datetime, datetime]:
    """Concrete UTC [start_dt, end_dt) for the dead window *owned by* date ``d``.

    ``end_dt`` may land on the following calendar day (a late-starting, long
    window crosses midnight). Modeling the window as a wall-clock interval owned
    by one date — rather than a recurring hour-of-day pattern — means a window
    that does NOT cross midnight cannot spuriously match the next day's clock.
    """
    start_h, end_h = dead_window_for_date(d, seed)
    day_start = datetime.combine(d, time(0, 0), tzinfo=timezone.utc)
    return day_start + timedelta(hours=start_h), day_start + timedelta(hours=end_h)


def is_dead_window(now: datetime, *, seed: int) -> bool:
    """True if ``now`` (UTC) lands inside the bot's mandatory daily sleep window.

    The window's start and duration jitter per UTC date (stable for a fixed
    date+seed) but it is always >=6h. We check the concrete interval owned by
    ``now``'s date and the previous date, since a window that starts late and
    crosses midnight is "owned" by the prior date.
    """
    now = _as_utc(now)
    today = now.date()
    for d in (today, today - timedelta(days=1)):
        start_dt, end_dt = _window_interval(d, seed)
        if start_dt <= now < end_dt:
            return True
    return False


def next_active_time(now: datetime, seed: int) -> datetime:
    """When the bot is next awake.

    If ``now`` is inside the dead window, return the window's wake (end) time;
    otherwise return ``now`` unchanged. Adjacent days' windows can overlap, so
    we advance past every interval the cursor is continuously inside.
    """
    now = _as_utc(now)
    cursor = now
    # Bounded loop: at most a couple of overlapping windows can chain.
    for _ in range(4):
        cursor_date = cursor.date()
        end_candidates = [
            end_dt
            for d in (cursor_date - timedelta(days=1), cursor_date)
            for start_dt, end_dt in (_window_interval(d, seed),)
            if start_dt <= cursor < end_dt
        ]
        if not end_candidates:
            break
        cursor = max(end_candidates)
    return cursor


def _as_utc(now: datetime) -> datetime:
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# Latency self-monitor
# ---------------------------------------------------------------------------


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolation percentile on a pre-sorted list (q in [0, 1])."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def latency_cdf_alarm(observed_latencies: list[float], band: tuple[float, float]) -> bool:
    """True if the bot's recent latency distribution has collapsed (too uniform).

    Rule: with enough samples, flag the distribution when EITHER
      * coefficient of variation std/mean < ``_CV_MIN_HEALTHY`` (the spread is
        too narrow for a human), OR
      * interquartile spread (p75 - p25) / p50 < ``_IQR_RATIO_MIN_HEALTHY``
        (the middle of the distribution is too tight, e.g. a fixed timer).

    ``band`` is the healthy regular latency band (p_low, p_high), accepted as a
    documentation-level reference for the caller; the shape tests (CV and IQR)
    are scale-free, so the decision does not depend on ``band``. A realistic
    heavy-tailed spread (the regular band itself) does not alarm.
    """
    vals = [float(v) for v in observed_latencies if v is not None and v >= 0.0]
    if len(vals) < _ALARM_MIN_SAMPLES:
        return False

    mean = sum(vals) / len(vals)
    if mean <= 0.0:
        return True  # a pile of zeros is maximally degenerate

    variance = sum((v - mean) ** 2 for v in vals) / len(vals)
    cv = math.sqrt(variance) / mean
    if cv < _CV_MIN_HEALTHY:
        return True

    s = sorted(vals)
    p25 = _percentile(s, 0.25)
    p50 = _percentile(s, 0.50)
    p75 = _percentile(s, 0.75)
    if p50 > 0.0 and (p75 - p25) / p50 < _IQR_RATIO_MIN_HEALTHY:
        return True

    return False
