"""
TIMING classifier scorer (the advisor's Part 2, served inside the engine).

Loads the tiny logistic-regression model trained by scripts/train_timing_classifier.py
(models/timing_classifier.json) and scores an incoming message:

    "given this message, would a regular actually bother to respond?"

The feature extraction here MUST match scripts/build_timing_dataset.py exactly, or the
standardized weights won't mean anything. Kept in lockstep:
  is_mention, is_reply, reply_to_regular, msg_len_words, msg_len_bucket, has_number,
  has_claim_token, is_question, sender_is_regular, idx_gap_since_sender, is_botlike

Scoring: sigmoid(((x - mean) / std) @ weights + bias). If is_botlike, force-skip.
Above chosen_threshold => worth a (potential) response; the smart model still decides
the actual WHETHER/WHAT. This only cheaply filters the firehose down to the ~6% of
messages that realistically earn a reply, before any paid LLM call.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import structlog

log = structlog.get_logger()

# --- feature regexes: copied verbatim from scripts/build_timing_dataset.py ---
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

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "timing_classifier.json"


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


@dataclass
class TimingScore:
    score: float
    passes: bool
    is_botlike: bool
    features: dict


class TimingClassifier:
    """Self-contained scorer. Loads once; score() is pure-python and cheap."""

    def __init__(self, model_path: Optional[Path] = None):
        self.ok = False
        self.threshold = 0.8
        path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        try:
            data = json.loads(Path(path).read_text())
            self.feature_order = data["feature_order"]
            self.weights = data["weights"]
            self.bias = float(data["bias"])
            self.mean = data["feature_mean"]
            self.std = data["feature_std"]
            self.threshold = float(data.get("chosen_threshold", 0.8))
            self.ok = True
            log.info(
                "timing_classifier_loaded",
                path=str(path),
                threshold=self.threshold,
                n_features=len(self.feature_order),
            )
        except Exception as exc:  # missing/corrupt model => disabled, fail open
            log.warning("timing_classifier_load_failed", path=str(path), error=str(exc))

    def _features(
        self,
        *,
        text: str,
        is_reply: bool,
        reply_to_regular: bool,
        sender_is_regular: bool,
        idx_gap_since_sender: int,
    ) -> dict:
        wc = len(text.split())
        bot = _is_botlike(text)
        return {
            "is_mention": int(bool(MENTION.search(text))),
            "is_reply": int(is_reply),
            "reply_to_regular": int(reply_to_regular),
            "msg_len_words": wc,
            "msg_len_bucket": _len_bucket(wc),
            "has_number": int(bool(NUMBER.search(text))),
            "has_claim_token": int(bool(CLAIM.search(text))),
            "is_question": int(bool(text.rstrip().endswith("?") or WH.search(text))),
            "sender_is_regular": int(sender_is_regular),
            "idx_gap_since_sender": idx_gap_since_sender,
            "is_botlike": int(bot),
        }

    def score(
        self,
        *,
        text: str,
        is_reply: bool = False,
        reply_to_regular: bool = False,
        sender_is_regular: bool = True,
        idx_gap_since_sender: int = -1,
    ) -> TimingScore:
        feats = self._features(
            text=text,
            is_reply=is_reply,
            reply_to_regular=reply_to_regular,
            sender_is_regular=sender_is_regular,
            idx_gap_since_sender=idx_gap_since_sender,
        )
        # Hard filter: bot/feed/command messages never earn a response.
        if feats["is_botlike"]:
            return TimingScore(score=0.0, passes=False, is_botlike=True, features=feats)
        if not self.ok:
            # Fail open: if the model didn't load, don't block anything.
            return TimingScore(score=1.0, passes=True, is_botlike=False, features=feats)

        z = self.bias
        for i, name in enumerate(self.feature_order):
            std = self.std[i] or 1.0
            z += ((feats[name] - self.mean[i]) / std) * self.weights[i]
        prob = 1.0 / (1.0 + math.exp(-z))
        return TimingScore(
            score=prob,
            passes=prob >= self.threshold,
            is_botlike=False,
            features=feats,
        )
