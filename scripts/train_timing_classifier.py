#!/usr/bin/env python3
"""Train the tiny timing logistic-regression model from a built dataset.

Consumes the JSONL produced by scripts/build_timing_dataset.py (a `_meta` header line
+ one {"message_id", "features", "label"} object per row, oldest->newest) and writes a
models/<name>.json that conversation_engine.timing_classifier.TimingClassifier loads.

Pipeline (numpy + stdlib only — no sklearn):
  1. TIME-ORDERED 60/20/20 split (train / cal / test). Rows are already chronological,
     so slicing by position avoids any future leakage into the standardizer or weights.
  2. Standardize features on TRAIN stats (mean/std), then fit logistic regression by
     full-batch gradient descent with L2.
  3. Isotonic calibration (pool-adjacent-violators) of raw sigmoid -> empirical reply
     rate on the CAL split; emitted as knots for any consumer wanting a true probability.
  4. Pick chosen_threshold in RAW-SIGMOID space (what serving thresholds) as the lowest
     raw score whose CAL pass-rate <= --target-response-rate (default 0.06 cadence).

Output schema matches models/timing_classifier_v2.json: feature_order, weights, bias,
feature_mean, feature_std, chosen_threshold, calibration{knots_x,knots_y}, time_split,
label_key, regulars, target_response_rate.

Usage:
    python scripts/train_timing_classifier.py \
        --dataset data/timing_dataset.jsonl \
        --out models/timing_classifier_next.json \
        --target-response-rate 0.06
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

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


def load_dataset(path: str | Path):
    """Return (X, y, meta) preserving the file's chronological row order."""
    meta: dict = {}
    rows: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "_meta" in obj:
                meta = obj["_meta"]
                continue
            rows.append(obj)
    order = meta.get("feature_order", FEATURE_ORDER)
    X = np.array([[float(r["features"][k]) for k in order] for r in rows], dtype=float)
    y = np.array([float(r["label"]) for r in rows], dtype=float)
    return X, y, meta, order


def fit_logreg(X: np.ndarray, y: np.ndarray, *, lr=0.1, epochs=4000, l2=1e-3):
    """Full-batch gradient-descent logistic regression on standardized X.

    Returns (weights, bias). Standardization happens upstream; columns with zero
    variance (e.g. is_botlike, all 0 after the hard filter) contribute a 0 weight.
    """
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(epochs):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        err = p - y
        w -= lr * (X.T @ err / n + l2 * w)
        b -= lr * float(err.mean())
    return w, b


def isotonic_pav(x: np.ndarray, y: np.ndarray):
    """Pool-adjacent-violators isotonic regression. x sorted ascending; returns the
    fitted non-decreasing y-hat aligned to the sorted x (knots for a step map)."""
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order].astype(float)
    yhat = ys.copy()
    weight = np.ones_like(yhat)
    i = 0
    while i < len(yhat) - 1:
        if yhat[i] > yhat[i + 1]:
            pooled = (yhat[i] * weight[i] + yhat[i + 1] * weight[i + 1]) / (
                weight[i] + weight[i + 1]
            )
            yhat[i] = yhat[i + 1] = pooled
            weight[i] = weight[i + 1] = weight[i] + weight[i + 1]
            # Back up to repair any new upstream violation.
            while i > 0 and yhat[i - 1] > yhat[i]:
                pooled = (yhat[i - 1] * weight[i - 1] + yhat[i] * weight[i]) / (
                    weight[i - 1] + weight[i]
                )
                yhat[i - 1] = yhat[i] = pooled
                weight[i - 1] = weight[i] = weight[i - 1] + weight[i]
                i -= 1
        else:
            i += 1
    return xs, yhat


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z)) if not isinstance(z, np.ndarray) else 1.0 / (1.0 + np.exp(-z))


def pick_threshold_for_rate(cal_scores: np.ndarray, target_rate: float) -> float:
    """Lowest raw-sigmoid threshold whose CAL pass-rate is <= target_rate.

    Sending fewer than target is acceptable (cadence cap); we want the most permissive
    threshold that still respects the cap, so we scan high->low and stop when the rate
    would exceed target."""
    n = len(cal_scores) or 1
    chosen = 1.0
    for t in sorted(set(cal_scores.tolist()), reverse=True):
        rate = float((cal_scores >= t).sum()) / n
        if rate <= target_rate:
            chosen = float(t)
        else:
            break
    return chosen


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", required=True, help="JSONL from build_timing_dataset.py")
    ap.add_argument("--out", required=True, help="destination models/<name>.json")
    ap.add_argument("--target-response-rate", type=float, default=0.06)
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--cal-frac", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--l2", type=float, default=1e-3)
    args = ap.parse_args(argv)

    X, y, meta, order = load_dataset(args.dataset)
    if len(X) < 3:
        raise SystemExit(f"need >=3 rows to make a 60/20/20 split, got {len(X)}")

    n = len(X)
    n_train = max(1, int(n * args.train_frac))
    n_cal = max(1, int(n * args.cal_frac))
    n_train = min(n_train, n - 2)  # leave >=1 each for cal and test
    n_cal = min(n_cal, n - n_train - 1)
    Xtr, ytr = X[:n_train], y[:n_train]
    Xcal, ycal = X[n_train : n_train + n_cal], y[n_train : n_train + n_cal]
    # Test split (Xte) is held out for an honest cadence readout; not used to fit anything.
    Xte = X[n_train + n_cal :]

    mean = Xtr.mean(axis=0)
    std = Xtr.std(axis=0)
    std_safe = np.where(std == 0, 1.0, std)

    def standardize(M):
        return (M - mean) / std_safe

    w, b = fit_logreg(standardize(Xtr), ytr, lr=args.lr, epochs=args.epochs, l2=args.l2)

    cal_scores = sigmoid(standardize(Xcal) @ w + b)
    knots_x, knots_y = isotonic_pav(cal_scores, ycal)
    chosen_threshold = pick_threshold_for_rate(cal_scores, args.target_response_rate)

    # Honest test-split readout (not used for fitting).
    te_scores = sigmoid(standardize(Xte) @ w + b) if len(Xte) else np.array([])
    te_rate = float((te_scores >= chosen_threshold).mean()) if len(te_scores) else 0.0

    model = {
        "feature_order": order,
        "weights": w.tolist(),
        "bias": float(b),
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "chosen_threshold": float(chosen_threshold),
        "regulars": meta.get("regulars", []),
        "target_response_rate": args.target_response_rate,
        "notes": (
            "Score: sigmoid(((x-mean)/std)@weights + bias). Hard-filter is_botlike=1 to 0 "
            "before scoring. chosen_threshold is in RAW-SIGMOID space (what the serving "
            "loader thresholds) at the target cadence on the cal split. calibration.knots_* "
            "give the isotonic raw->probability map. Trained by scripts/train_timing_classifier.py."
        ),
        "calibration": {
            "method": "isotonic_pav",
            "knots_x": [float(v) for v in knots_x],
            "knots_y": [float(v) for v in knots_y],
        },
        "time_split": {"train_frac": args.train_frac, "cal_frac": args.cal_frac},
        "label_key": meta.get("label_key", "label"),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(model, indent=2))

    print(
        f"trained on {len(Xtr)}/{len(Xcal)}/{len(Xte)} (train/cal/test) rows; "
        f"chosen_threshold={chosen_threshold:.4f}; test pass-rate={te_rate:.2%}; "
        f"wrote {out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
