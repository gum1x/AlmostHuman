from __future__ import annotations

import random

# ---------------------------------------------------------------------------
# Volume governor
# ---------------------------------------------------------------------------
#
# Volume is the #1 catch signal for chat bots. GPT-4chan was unmasked not by
# any single bad message but by sheer *throughput*: it posted ~15,000 times in a
# couple of days, far above any human's share of the board's traffic. Humans in
# a busy group contribute a small, lumpy fraction of total messages; an
# always-on account that answers a meaningful slice of everything is a statistical
# outlier that's trivial to flag with a simple counter. So we cap two things:
#   1. the bot's *share* of recent traffic (don't be a measurable fraction), and
#   2. absolute hourly / 10-minute counts (don't machine-gun a quiet room).
#
# Constants are deliberately conservative — a real low-ego regular lurks far more
# than it speaks.

# Jitter applied to the share ceiling so the cutoff isn't a crisp, detectable
# line at exactly max_share. We pull the effective ceiling DOWN by a small random
# margin, so the bot starts backing off a touch early and unpredictably.
_SHARE_JITTER_FRAC = 0.25  # up to 25% of max_share shaved off as margin


def should_suppress(
    *,
    bot_sends_last_hour: int,
    group_msgs_last_hour: int,
    bot_sends_last_10min: int,
    rng: random.Random,
    max_share: float = 0.02,
    hourly_ceiling: int = 12,
    ten_min_ceiling: int = 3,
) -> tuple[bool, str]:
    """Decide whether to suppress an otherwise-approved send to stay invisible.

    Returns ``(suppress, reason)``. ``reason`` is a short tag for telemetry.

    Suppression triggers, in priority order:
    - ``ten_min_ceiling`` — too many sends in the last 10 minutes (bursty machine-gun).
    - ``hourly_ceiling`` — too many sends in the last hour (always-on outlier).
    - ``share`` — the bot's share of recent traffic would exceed a jittered
      fraction of ``max_share``. The jitter shaves a small random margin off the
      ceiling so there's no crisp, reverse-engineerable line at exactly 2%.

    The share is computed as if THIS send already happened, so we suppress the
    message that would push us over rather than discovering it after the fact.
    """
    if bot_sends_last_10min >= ten_min_ceiling:
        return True, "ten_min_ceiling"
    if bot_sends_last_hour >= hourly_ceiling:
        return True, "hourly_ceiling"

    # Count this prospective send in both the numerator and denominator: a bot
    # message is also a group message.
    projected_bot = bot_sends_last_hour + 1
    projected_total = group_msgs_last_hour + 1
    share = projected_bot / projected_total

    # Shave a small jittered margin off the ceiling so backoff begins a touch
    # early and the threshold isn't a hard detectable line.
    margin = rng.uniform(0.0, _SHARE_JITTER_FRAC) * max_share
    effective_ceiling = max_share - margin
    if share >= effective_ceiling:
        return True, "share"

    return False, "ok"
