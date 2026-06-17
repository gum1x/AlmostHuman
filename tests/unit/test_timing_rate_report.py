from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import timing_rate_report as rr  # noqa: E402

from conversation_engine.timing_classifier import TimingClassifier  # noqa: E402


def _toy_classifier(tmp_path):
    model = {
        "feature_order": ["is_mention","is_reply","reply_to_regular","msg_len_words",
                          "msg_len_bucket","has_number","has_claim_token","is_question",
                          "sender_is_regular","idx_gap_since_sender","is_botlike"],
        "weights": [0,0,0,0,0,0,0,5.0,0,0,0], "bias": -2.5,
        "feature_mean": [0]*11, "feature_std": [1]*11, "chosen_threshold": 0.5,
    }
    p = tmp_path / "toy.json"
    p.write_text(json.dumps(model))
    return TimingClassifier(model_path=p)


def test_botlike_scores_zero(tmp_path):
    clf = _toy_classifier(tmp_path)
    rows = [{"text": "/start", "is_reply": False, "reply_to_regular": False,
             "sender_is_regular": True, "idx_gap_since_sender": -1}]
    assert rr.score_rows(clf, rows) == [0.0]


def test_question_scores_high(tmp_path):
    clf = _toy_classifier(tmp_path)
    rows = [{"text": "anyone selling?", "is_reply": False, "reply_to_regular": False,
             "sender_is_regular": True, "idx_gap_since_sender": -1}]
    assert rr.score_rows(clf, rows)[0] > 0.9


def test_rate_table_counts_pass_fraction():
    table = dict(rr.rate_table([0.9, 0.9, 0.1, 0.1], [0.5, 0.95]))
    assert table[0.5] == 0.5
    assert table[0.95] == 0.0
