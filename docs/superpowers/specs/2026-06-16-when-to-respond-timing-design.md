# When-to-Respond Timing Model — Design Spec

**Date:** 2026-06-16
**Status:** Approved design, pending implementation plan
**Author:** brainstormed with the owner

## 1. Problem

The bot decides *when* to speak with two parallel, partly redundant systems: a hand-weighted
`engagement_gate` and a trained-but-unused logistic-regression timing classifier
(`models/timing_classifier_v2.json`, `TIMING_CLASSIFIER_ENABLED=false`). The gate's weights are
guessed, not learned. The result is poorly calibrated: the bot sends ~4 messages/day and its rate
isn't tied to how a real member behaves.

## 2. Objective

Make the bot's *timing* imitate a real regular: speak at a human-plausible rate and on the same
kinds of messages a regular speaks on. Pure imitation — not engagement-maximizing, not a stealth
optimizer. Stealth and safety are enforced as guardrails, not learned.

## 3. Locked decisions (from brainstorming)

- **Objective:** blend in by rate (imitation), matching both rate and message selection.
- **Voice persona:** donor `u8035098195` (unchanged; from Com_Chat).
- **Timing persona:** a *live-group* regular, default `8384923892` — because the voice donor has
  **zero** messages in the live group (`-1002705709115`), so his in-group timing can't be learned.
  The live-group audience never knew the donor, so a "talks-like-X, times-like-Y" split is
  invisible to them, and live regulars give 8k+ in-context reply decisions to learn from.
- **Model:** the existing trained logistic regression. Not a neural network — CPU-only constraint
  (VPS GPU is sm_61, no modern PyTorch kernels), interpretable, sufficient for ~11 tabular features.
- **Architecture:** the classifier *replaces* the gate's timing role; the LLM stays as a backstop.

## 4. The rate target (the key insight)

A member's "response rate" has two very different meanings. Measured on the live group
(`data/prod_export`, 163,648 msgs):

| Regular | Overall (of *all* msgs, fraction they reply to) | When *addressed* (someone replies to / @s them → they reply back) |
|---|---|---|
| `8384923892` | 1.9% | **65%** |
| `5755932997` | 2.7% | 39% |
| `5564587165` | 1.3% | 63% |

"Responds ~60% of the time" is the **addressed** rate, not the overall rate. Setting the *overall*
rate to 35% would mean replying to 1 in 3 of *all* messages — ~17× a real human — instant spam.

**Target (decision A, 2026-06-16):**
- **Addressed** (bot is @mentioned, someone replies to a bot message, or an active bot thread):
  **always engage** — let the LLM + validators decide quality. Ignoring people who talk to you is
  itself a bot tell, so near-always answering is the human move. *This is already the code's
  behavior* — addressed messages bypass the timing classifier (`scheduler.py:476-480`); no change.
- **Unprompted firehose** (messages not aimed at the bot): the classifier's selective filter. This
  is where being choosy reads as natural. Calibrate the threshold here (chattier-leaning, per owner)
  and monitor the resulting overall rate as a spam ceiling.

So the classifier governs *only the unprompted stream*. The "~35%/60%" discussion settled the
addressed policy (always engage); the tunable number is the unprompted pass rate.

## 5. Architecture — replace, don't stack

Stacking the classifier on top of the existing gate multiplies their rates → uncontrollable and far
too quiet. The classifier replaces the gate's *timing* role. The gate's *safety* factors (anti-flame
tension, fatigue/volume ceilings) move to the guardrail layer.

```
new message
   │
   ▼
[timing_classifier_v2]  ── score < threshold ──►  stay silent   (the large majority)
   │ score ≥ threshold
   ▼
[LLM decision (kimi)]   ── "no" ──►  stay silent   (precision backstop: scams, flame, low-value)
   │ should_respond
   ▼
[voice model writes]  →  [validators + humanizer + suspicion/volume guardrails]  →  send
```

Components, each independently testable:
- **Timing classifier** — input: 11 features for the candidate message; output: raw sigmoid score
  and a calibrated probability. Decides candidacy against a threshold. Pure function of features.
- **LLM backstop** — unchanged kimi decision; final yes/no + meaning.
- **Voice renderer** — unchanged (the donor reply-pairs model, already serving).
- **Guardrails** — anti-flame, volume governor, suspicion monitor, dedup. Hard filters *after* the
  classifier; a learned score never overrides them.

## 6. How we guarantee the rate (the real work)

The classifier's offline "6%" will not equal the live rate, because (a) the LLM backstop vetoes some
candidates and (b) live traffic drifts from training. So:

1. **Shadow mode.** Wire the classifier in but log "would-fire" decisions only — send nothing. Run
   on live traffic for several days.
2. **Measure** the realized would-fire rate, split by addressed vs unprompted, and the post-LLM
   would-actually-send rate.
3. **Calibrate** the threshold to the chosen *unprompted-firehose* pass rate (adjust the raw
   threshold from its 0.825 default — chattier-leaning per owner; the model's isotonic calibration
   map turns a target rate into a threshold lookup, then verify on the shadow/replay data). Addressed
   messages always engage regardless of threshold.
4. **Enforce** — flip to live at that threshold. Keep measuring; alarm if the 7-day addressed rate
   drifts outside a band (e.g. [25%, 50%]) or the overall rate exceeds a ceiling.

This loop delivers the rate instead of hoping for it, and de-risks the live behavior change: we see
exactly what the bot *would* do before it does anything.

## 7. Evaluation

- **Offline replay** on held-out live-group history: precision/recall at the chosen threshold, plus
  an eyeball check that fired-on messages are sensible spots to speak (not just AUC).
- **Per-rate verification:** addressed rate ≈ 35%, overall rate within ceiling.
- The current replay harness excludes the timing classifier; closing that gap is part of the work.

## 8. Safety & rollback

- Guardrails (anti-flame, volume ceiling, suspicion monitor, dedup) remain hard filters after the
  classifier.
- **Rollback is one env flag:** `TIMING_CLASSIFIER_ENABLED=false` restores today's behavior. Zero-risk.

## 9. Phasing

- **Phase 0 — shadow + eval.** Wire classifier in shadow mode; build the missing replay eval; log
  scores + would-fire + outcomes. No behavior change.
- **Phase 1 — calibrate + enforce.** Tune threshold to ~35% addressed; replace the gate's timing
  role; flip to live behind the flag; keep the rate alarm running.
- **Fast-follow (optional) — persona-specific retrain.** The current model's label is "did the
  *room* respond," a close proxy. If it isn't persona-specific enough, relabel on "did `8384923892`
  reply" and retrain the same logreg.

## 10. Success criteria

- Addressed messages near-always engage (bypass preserved); unprompted-firehose pass rate holds at
  the calibrated target (within the alarm band) over a rolling 7-day window.
- Overall rate stays under the spam ceiling.
- Fired-on unprompted messages look like reasonable places for a regular to speak (human eyeball).
- One-flag rollback verified.

## 11. Open items

- Exact unprompted-firehose target rate + alarm bands (set during calibration from replay/shadow data).
- Whether to keep the LLM backstop long-term (arch A) or move to fully-local timing+voice (arch B)
  once the classifier is trusted.

**Resolved during planning (2026-06-16):**
- Addressed-message policy → **always engage** (decision A); selectivity applies to the unprompted
  stream only.
- The classifier + serve-time scorer are **already built and wired** (`timing_classifier.py`,
  `scheduler.py:476-529`), flag-off on the v1 model path. The remaining work is shadow measurement,
  threshold calibration, and flipping it on against the v2 model — not building from scratch.
