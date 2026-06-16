"""Train==serve parity for the timing classifier features (TASK A1).

The serve-time feature computation (conversation_engine/timing_classifier.py
compute_regulars / history_feature_inputs + TimingClassifier._features) must match
the TRAINING definitions in scripts/build_timing_dataset.py exactly:

  regulars             top-K most active senders by message count
                       (`sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_k]`)
  is_reply             `int(r["parent"] is not None)`
  reply_to_regular     `int(r["parent"] in sender_of and sender_of.get(r["parent"]) in regulars)
                        if r["parent"] else 0`  — parent message is in the export AND its
                       author is a regular. NOT "replies to the bot".
  sender_is_regular    `int(sender in regulars)`
  idx_gap_since_sender `i - last_spoke_idx[sender]` (message-index gap; last_spoke_idx is
                       updated for EVERY sender, including non-regulars), -1 if not seen
  (text features)      MENTION/NUMBER/CLAIM/WH/BOTLIKE regexes + len buckets

load_ordered() in the script drops empty-text messages BEFORE any counting/indexing,
so the serve side must too.

`train_features()` below is a line-for-line port of the script's main() loop, used as
the parity reference against synthetic message fixtures.
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("ALLOW_FAKE_EMBEDDER", "true")

from conversation_engine.config import (
    AiConfig,
    CircuitBreakerConfig,
    EngagementGateConfig,
    EngineConfig,
    FeedbackLoopConfig,
    PersonaConfig,
    PersonaEngineConfig,
    PromptConfig,
    SchedulerConfig,
)
from conversation_engine.persona_engine import load_embedder
from conversation_engine.scheduler import (
    TIMING_REGULARS_HISTORY_LIMIT,
    ConversationScheduler,
)
from conversation_engine.timing_classifier import (
    TimingClassifier,
    TimingScore,
    compute_regulars,
    history_feature_inputs,
)

load_embedder("test")


@dataclass
class Msg:
    """Stand-in for both EnrichedMessage (text/...) and the Message ORM row
    (text_cleaned/text_raw/...), so one fixture drives every code path."""

    message_id: int
    sender_id: int | None
    reply_to_message_id: int | None
    text: str
    chat_id: int = -100
    timestamp: datetime = datetime(2026, 6, 10, tzinfo=timezone.utc)

    @property
    def text_cleaned(self):
        return self.text

    @property
    def text_raw(self):
        return self.text

    @property
    def cleaned_text(self):
        return self.text

    @property
    def raw_text(self):
        return self.text


def train_features(rows: list[Msg], top_k: int):
    """Reference port of scripts/build_timing_dataset.py main() (history features only).

    Mirrors load_ordered() (empty-text rows already dropped by the caller's filter),
    the regulars computation, sender_of, last_spoke_idx bookkeeping (updated for
    non-regulars in the `continue` branch too), and the feat dict expressions.
    Rows from non-regulars are skipped, exactly as in training.
    """
    rows = [r for r in rows if r.text.strip()]  # load_ordered() drops empty text
    counts: dict[int, int] = defaultdict(int)
    for r in rows:
        if r.sender_id:
            counts[r.sender_id] += 1
    regulars = {u for u, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_k]}
    sender_of = {r.message_id: r.sender_id for r in rows}
    last_spoke_idx: dict[int, int] = {}
    feats: dict[int, dict] = {}
    for i, r in enumerate(rows):
        sender = r.sender_id
        if sender not in regulars:
            if sender is not None:
                last_spoke_idx[sender] = i
            continue
        mins_since = None
        if sender in last_spoke_idx:
            mins_since = i - last_spoke_idx[sender]  # index gap as time proxy
        if sender is not None:
            last_spoke_idx[sender] = i
        parent = r.reply_to_message_id
        feats[r.message_id] = {
            "is_reply": int(parent is not None),
            "reply_to_regular": int(parent in sender_of and sender_of.get(parent) in regulars)
            if parent
            else 0,
            "sender_is_regular": int(sender in regulars),
            "idx_gap_since_sender": mins_since if mins_since is not None else -1,
        }
    return regulars, feats


# Synthetic chat. With top_k=3: u1 (5 msgs), u2 (4), u3 (3) are regulars;
# u4 (2) and u5 (1, empty text only) are not.
U1, U2, U3, U4, U5 = 1, 2, 3, 4, 5
FIXTURE = [
    Msg(1, U1, None, "gm everyone"),
    Msg(2, U2, None, "gm"),
    Msg(3, U4, None, "yo whats the price"),
    Msg(4, U1, 3, "like 5k"),  # reply to a NON-regular -> reply_to_regular 0
    Msg(5, U3, None, "anyone selling?"),
    Msg(6, U2, 5, "i might"),  # reply to a regular -> reply_to_regular 1
    Msg(7, U5, None, "   "),  # empty text: dropped before indexing (load_ordered)
    Msg(8, U1, 999, "what"),  # reply to a parent NOT in history -> reply_to_regular 0
    Msg(9, U3, None, "@someuser u around?"),
    Msg(10, U1, None, "lol"),
    Msg(11, U2, None, "sold it for $4k"),
    Msg(12, U4, None, "vouch"),
    Msg(13, U3, 12, "who are you again?"),  # reply to non-regular u4 -> 0
    Msg(14, U1, 6, "u never sell"),  # reply to regular u2 -> 1
    Msg(15, U2, 14, "cope"),  # reply to regular u1 -> 1
]
TOP_K = 3


def serve_features(target_id: int, history=FIXTURE, top_k: int = TOP_K) -> dict:
    regulars = compute_regulars((m.sender_id for m in history if m.text.strip()), top_k=top_k)
    return history_feature_inputs(target_message_id=target_id, history=history, regulars=regulars)


def test_regulars_match_training_definition():
    train_regulars, _ = train_features(FIXTURE, TOP_K)
    serve_regulars = compute_regulars((m.sender_id for m in FIXTURE if m.text.strip()), top_k=TOP_K)
    assert serve_regulars == train_regulars == {U1, U2, U3}
    # u5 only ever sent empty text: never counted (load_ordered drops the row).
    assert U5 not in compute_regulars((m.sender_id for m in FIXTURE if m.text.strip()), top_k=5)


def test_every_history_feature_matches_training_for_every_regular_message():
    _, train = train_features(FIXTURE, TOP_K)
    assert train  # sanity: fixture produced training rows
    for message_id, expected in train.items():
        got = serve_features(message_id)
        for key, value in expected.items():
            assert int(got[key]) == value, f"msg {message_id} feature {key}: serve={got[key]} train={value}"


def test_reply_to_regular_is_author_based_not_reply_to_bot():
    """Regression for the silence trap: training defines reply_to_regular as 'parent's
    author is a top-K regular' (build_timing_dataset.py line ~174), NOT 'replies to one
    of the bot's messages'. A reply to a regular must score 1 even if the bot has been
    silent for days (bot_sent_ids empty)."""
    got = serve_features(6)  # u2 replying to regular u3's msg 5
    assert got["is_reply"] is True
    assert got["reply_to_regular"] is True
    # Reply to a non-regular's message -> 0 (msg 4 replies to u4's msg 3).
    assert serve_features(4)["reply_to_regular"] is False
    # Reply to a parent outside the history window -> 0 (training: parent not in sender_of).
    assert serve_features(8)["is_reply"] is True
    assert serve_features(8)["reply_to_regular"] is False


def test_sender_is_regular_computed_for_real():
    assert serve_features(14)["sender_is_regular"] is True  # u1 is top-K
    # u4 is not a regular at top_k=3. Training never emits rows for non-regulars, but
    # serve-time must compute the real value instead of hardcoding True.
    assert serve_features(12)["sender_is_regular"] is False
    # With top_k=5 (everyone with non-empty text is a regular) u4 flips to True,
    # matching `sender in regulars` from the script.
    assert serve_features(12, top_k=5)["sender_is_regular"] is True


def test_idx_gap_since_sender_matches_training_index_gap():
    """Training: `i - last_spoke_idx[sender]` over load_ordered() rows (empty text
    dropped; last_spoke_idx updated for every sender, regulars or not); -1 when the
    sender has not spoken before (build_timing_dataset.py lines ~165-169, 181)."""
    # First message from u1 -> -1.
    assert serve_features(1)["idx_gap_since_sender"] == -1
    # Indices after dropping msg 7: [1,2,3,4,5,6,8,9,10,11,12,13,14,15].
    # u1 spoke at idx 0 (msg 1) and idx 3 (msg 4): gap 3.
    assert serve_features(4)["idx_gap_since_sender"] == 3
    # msg 8 is u1 again at idx 6: gap 6-3=3 (msg 7 not counted because it was dropped).
    assert serve_features(8)["idx_gap_since_sender"] == 3
    # msg 10 (u1, idx 8): gap 8-6=2.
    assert serve_features(10)["idx_gap_since_sender"] == 2
    # Cross-check the whole fixture against the training port.
    _, train = train_features(FIXTURE, TOP_K)
    for message_id, expected in train.items():
        assert serve_features(message_id)["idx_gap_since_sender"] == expected["idx_gap_since_sender"]


def test_unknown_target_falls_back_to_defaults():
    got = serve_features(424242)
    assert got == {
        "is_reply": False,
        "reply_to_regular": False,
        "sender_is_regular": False,
        "idx_gap_since_sender": -1,
    }


def test_text_features_match_training_regexes():
    """The text-derived features in TimingClassifier._features must match the script's
    expressions (MENTION/NUMBER/CLAIM/WH regexes, len_bucket, is_botlike — copied
    verbatim per the module docstring). Spot-check representative texts."""
    import re

    # Ports of scripts/build_timing_dataset.py lines 36-64.
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

    def train_len_bucket(wc):
        if wc <= 1:
            return 0
        if wc <= 3:
            return 1
        if wc <= 6:
            return 2
        if wc <= 12:
            return 3
        return 4

    tc = TimingClassifier(model_path=Path("/nonexistent/timing_classifier.json"))
    texts = [
        "gm",
        "@someuser u around?",
        "sold it for $4k",
        "anyone selling?",
        "who is this guy",
        "🛍 Sold ✅ Price: 5 🪙",
        "just a longer message that runs past twelve words to land in the top bucket ok",
    ]
    for text in texts:
        feats = tc._features(
            text=text,
            is_reply=False,
            reply_to_regular=False,
            sender_is_regular=True,
            idx_gap_since_sender=-1,
        )
        wc = len(text.split())
        assert feats["is_mention"] == int(bool(MENTION.search(text)))
        assert feats["msg_len_words"] == wc
        assert feats["msg_len_bucket"] == train_len_bucket(wc)
        assert feats["has_number"] == int(bool(NUMBER.search(text)))
        assert feats["has_claim_token"] == int(bool(CLAIM.search(text)))
        assert feats["is_question"] == int(bool(text.rstrip().endswith("?") or WH.search(text)))


# --- scheduler wiring: _prepare_cycle must feed the REAL features into score() ---


def make_config(**gate_overrides) -> EngineConfig:
    timing_classifier_enabled = gate_overrides.pop("timing_classifier_enabled", False)
    return EngineConfig(
        active_chat_ids=[],
        xai_api_key="",
        xai_base_url="",
        conversation_tg_session_name="test",
        persona=PersonaConfig(),
        ai=AiConfig(),
        prompt=PromptConfig(),
        scheduler=SchedulerConfig(),
        circuit_breaker=CircuitBreakerConfig(),
        persona_engine=PersonaEngineConfig(),
        feedback_loop=FeedbackLoopConfig(),
        engagement_gate=EngagementGateConfig(**gate_overrides),
        timing_classifier_enabled=timing_classifier_enabled,
    )


def make_scheduler(config=None) -> ConversationScheduler:
    return ConversationScheduler(
        config or make_config(),
        ai_client=SimpleNamespace(),
        sender=SimpleNamespace(),
        feedback_loop=SimpleNamespace(),
        bot_user_id=9999,
        bot_username="thebot",
    )


class FakeMemory:
    def __init__(self, messages, bot_mem=None, responses_10min=0, responses_60min=0):
        self.messages = list(messages)  # chronological
        self.bot_mem = list(bot_mem or [])  # newest first
        self.responses_10min = responses_10min
        self.responses_60min = responses_60min
        self.decisions = []
        self.recent_message_calls = []

    async def get_latest_ai_decision(self, chat_id):
        return None

    async def count_messages_after_snapshot(self, chat_id, snapshot_message_id):
        return len(self.messages)

    async def get_recent_bot_memory(self, chat_id, limit=50):
        return self.bot_mem[:limit]

    async def get_recent_messages(self, chat_id, limit=100):
        self.recent_message_calls.append(limit)
        return self.messages[-limit:]

    async def seed_persona_if_empty(self, **kwargs):
        return None

    async def get_avg_feedback_score(self, chat_id, window_hours=24):
        return 0.0

    async def latest_message_id(self, chat_id):
        return self.messages[-1].message_id if self.messages else None

    async def upsert_activity_pattern(self, chat_id, hour_of_day, day_of_week, velocity, tension):
        return None

    async def count_messages_in_window(self, chat_id, minutes):
        return len(self.messages)

    async def count_bot_responses(self, chat_id, window_minutes):
        return self.responses_10min if window_minutes == 10 else self.responses_60min

    async def avg_relationship_strength(self, chat_id, user_ids):
        return 0.0

    async def get_activity_pattern(self, chat_id, hour, day=None):
        return None

    async def count_bot_responses_in_threads(self, chat_id, thread_message_ids, window_minutes):
        return 0

    async def insert_ai_decision(self, **kwargs):
        self.decisions.append(kwargs)
        return SimpleNamespace(id=len(self.decisions))

    async def record_cycle_success(self, chat_id):
        return None


class RecordingClassifier:
    threshold = 0.9

    def __init__(self):
        self.calls = []

    def score(self, **kwargs):
        self.calls.append(kwargs)
        return TimingScore(score=0.1, passes=False, is_botlike=False, features={})


async def test_prepare_cycle_feeds_train_parity_features_to_classifier():
    # timing_classifier_enabled=True so the enforcing skip path is active.
    scheduler = make_scheduler(make_config(timing_classifier_enabled=True))
    classifier = RecordingClassifier()
    scheduler.timing_classifier = classifier
    memory = FakeMemory(messages=FIXTURE)  # no bot memory: bot silent "for days"

    result = await scheduler._prepare_cycle(memory, chat_id=-100, is_private_dm=False, previous_interval=30)

    # Classifier said skip -> cycle ends with a backoff interval + a recorded skip decision.
    assert isinstance(result, int)
    assert len(classifier.calls) == 1
    call = classifier.calls[0]
    # Target is the last message (15: u2 replying to regular u1's msg 14).
    expected = serve_features(15, top_k=60)  # scheduler uses the training default top-K
    assert call["is_reply"] == expected["is_reply"] is True
    # Regression: reply_to_regular True with ZERO bot activity (author-based, not reply-to-bot).
    assert call["reply_to_regular"] == expected["reply_to_regular"] is True
    assert call["sender_is_regular"] == expected["sender_is_regular"] is True
    assert call["idx_gap_since_sender"] == expected["idx_gap_since_sender"]
    assert call["idx_gap_since_sender"] != -1  # computed for real, not hardcoded
    # Skip decision persisted with the full gate factors + timing_p.
    assert memory.decisions and "timing_p" in memory.decisions[0]["gate_factors"]


async def test_timing_regulars_cached_per_chat():
    scheduler = make_scheduler()
    memory = FakeMemory(messages=FIXTURE)
    first = await scheduler._get_timing_regulars(memory, chat_id=-100)
    second = await scheduler._get_timing_regulars(memory, chat_id=-100)
    assert first == second
    assert memory.recent_message_calls.count(TIMING_REGULARS_HISTORY_LIMIT) == 1


# --- frozen regulars: v2 models embed the trained regulars set; serving must use it ---

V1_MODEL = {
    "feature_order": ["is_mention"],
    "weights": [1.0],
    "bias": 0.0,
    "feature_mean": [0.0],
    "feature_std": [1.0],
    "chosen_threshold": 0.6,
}


def write_model(tmp_path: Path, extra: dict) -> Path:
    import json

    path = tmp_path / "model.json"
    path.write_text(json.dumps({**V1_MODEL, **extra}))
    return path


def test_classifier_exposes_frozen_regulars_when_present(tmp_path):
    tc = TimingClassifier(model_path=write_model(tmp_path, {"regulars": [1, 2, 3]}))
    assert tc.ok
    assert tc.regulars == {1, 2, 3}


def test_classifier_regulars_none_for_v1_model(tmp_path):
    tc = TimingClassifier(model_path=write_model(tmp_path, {}))
    assert tc.ok
    assert tc.regulars is None


async def test_scheduler_uses_frozen_regulars_not_recent_window(tmp_path):
    scheduler = make_scheduler()
    scheduler.timing_classifier = TimingClassifier(
        model_path=write_model(tmp_path, {"regulars": [U2, U4]})
    )
    memory = FakeMemory(messages=FIXTURE)
    regulars = await scheduler._get_timing_regulars(memory, chat_id=-100)
    # Frozen set used verbatim (recent-window top-K would be {U1, U2, U3}), no DB read.
    assert regulars == {U2, U4}
    assert memory.recent_message_calls == []


async def test_scheduler_falls_back_to_recent_window_without_frozen_regulars(tmp_path):
    scheduler = make_scheduler()
    scheduler.timing_classifier = TimingClassifier(model_path=write_model(tmp_path, {}))
    memory = FakeMemory(messages=FIXTURE)
    regulars = await scheduler._get_timing_regulars(memory, chat_id=-100)
    assert regulars == compute_regulars(
        (m.sender_id for m in FIXTURE if m.text.strip()), top_k=60
    )
    assert memory.recent_message_calls == [TIMING_REGULARS_HISTORY_LIMIT]
