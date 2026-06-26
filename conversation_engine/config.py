from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def _split_ints(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class PersonaConfig:
    identity: str = "You are a bold, old timer in these specific groups. Sharp pattern-noticing brain + genuine enjoyment of the chaos, absurdity, and unhinged energy. You mostly only speak when there's a moment, but you sometimes just speak to hate or to pick someone's side or to give your opinion. Can be mean, spiky, opinionated. Not a normal or safe person."
    core_beliefs: list[str] = field(default_factory=list)
    speaking_style: str = "Bold and direct. Sharp economy. Mostly speak only when there's a moment, but sometimes just to hate, pick a side or give opinion. Reactive, topic-jumping, non-sequiturs normal. Can do hyperbolic shitposts, direct cruelty or praise, dry larp calls, random game spam, or warm/funny energy depending on mode and who I'm talking to. Typos and lazy spelling are authentic."


@dataclass(frozen=True)
class AiConfig:
    perception_model: str = "grok-4.3"
    decision_model: str = "grok-4.3"
    total_context_token_budget: int = 6_000
    max_output_tokens: int = 700
    min_confidence_to_send: float = 0.6
    prompt_version: str = "v1.1-grok-compact"
    persona_top_k: int = 4


@dataclass(frozen=True)
class PromptConfig:
    engagement_style: str = "bold_engaged"
    max_responses_per_hour: int = 8
    topics_of_interest: list[str] = field(default_factory=list)
    avoid_users: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class SchedulerConfig:
    initial_interval_seconds: int = 30
    max_interval_seconds: int = 300
    backoff_multiplier: float = 2.0
    new_message_threshold: int = 3
    worker_pool_size: int = 5
    monitor_private_dms: bool = True
    dm_discovery_interval_seconds: int = 15
    dm_new_message_threshold: int = 1
    dm_recent_message_limit: int = 20
    dm_max_active_chats: int = 25
    high_level_message_limit: int = 200
    recent_context_limit: int = 10


@dataclass(frozen=True)
class CircuitBreakerConfig:
    failure_threshold: int = 5
    pause_duration_minutes: int = 10


@dataclass(frozen=True)
class PersonaEngineConfig:
    self_reflection_interval_hours: int = 6
    self_reflection_message_threshold: int = 50
    embedding_model: str = "all-MiniLM-L6-v2"


@dataclass(frozen=True)
class FeedbackLoopConfig:
    observation_window_minutes: int = 45
    meta_reflection_interval_hours: int = 12


@dataclass(frozen=True)
class EngagementGateConfig:
    anti_flame_tension_threshold: float = 0.75
    velocity_window_minutes: int = 10
    thread_repeat_penalty_count: int = 2
    max_fatigue_score: float = 1.0
    min_gate_score_to_send: float = 0.25
    max_group_responses_per_10min: int = 3
    same_thread_cooldown_minutes: int = 30
    # Bait-loop cap: after this many consecutive bot replies to the same user (no other
    # human turn in between), that user's direct replies stop force-proceeding the gate.
    max_consecutive_replies_per_user: int = 2


@dataclass(frozen=True)
class EngineConfig:
    active_chat_ids: list[int]
    xai_api_key: str
    xai_base_url: str
    conversation_tg_session_name: str
    persona: PersonaConfig
    ai: AiConfig
    prompt: PromptConfig
    scheduler: SchedulerConfig
    circuit_breaker: CircuitBreakerConfig
    persona_engine: PersonaEngineConfig
    feedback_loop: FeedbackLoopConfig
    engagement_gate: EngagementGateConfig
    local_style_rewrite_enabled: bool = False
    local_style_python: str = "python3"
    local_style_chat_script: str = ""
    local_style_model_path: str = ""
    local_style_timeout_seconds: int = 120
    use_local_model_for_responses: bool = False  # Legacy flag (no longer used). Hybrid is now the default: smart model (Grok) for decisions+plan, local for phrasing when LOCAL_STYLE_REWRITE_ENABLED + http mode configured.
    local_inference_mode: str = "subprocess"  # "subprocess" or "http" — http is required for VPS (container calls host inference server)
    local_inference_url: str = ""  # e.g. http://172.17.0.1:8765/generate . Required for http mode.
    # "standalone" = new advisor-format voice model: send raw context only, model writes the
    # reply (single cloned regular). "phrase" = legacy: smart model emits a plan, local model
    # renders it. Must match how the local model was trained (build_voice_training.py = standalone).
    voice_mode: str = "standalone"
    emoji_window: int = 5  # if any of the last (emoji_window - 1) bot msgs had an emoji, strip emojis from this one (0 disables)
    # Timing classifier (advisor's Part 2): a cheap, data-trained pre-gate that scores the
    # incoming message ("would a regular bother to reply?") and skips the expensive LLM
    # perception+decision calls below threshold. Enforces the realistic ~6% response rate
    # and cuts paid API calls. See conversation_engine/timing_classifier.py.
    timing_classifier_enabled: bool = True
    timing_classifier_model_path: str = "models/timing_classifier_v2.json"
    timing_classifier_threshold: float = 0.0  # 0 = use the model's chosen_threshold
    timing_classifier_shadow: bool = False  # score+log "would-fire" without acting (measure first)
    # Cloud "brain" = the OpenRouter perception + decision LLM calls ("what kind of
    # situation is this? / what does someone like me do here?"). When False, the engine
    # skips BOTH paid calls and falls back to local-only mode: the timing classifier
    # decides WHEN to respond and the voice model writes WHAT. Turn off to stop OpenRouter
    # usage entirely (e.g. during rate-limit / billing issues) and run fully local.
    cloud_brain_enabled: bool = True
    # Behavioral layer (Phase 2): the humanizer/governor/suspicion/output-planner/validator
    # seam wired into the send path. Master switch is DEFAULT OFF — when False the send
    # path is byte-for-byte the original single-send finalize. Phase 3 builds on this seam.
    behavioral_layer_enabled: bool = False
    behavioral_allow_media: bool = False  # opt-in sticker/media sends (live media is unverified)
    behavioral_burst_rate: float = 0.20  # self-burst split probability (output_planner)
    behavioral_mention_rate: float = (
        0.0  # @-mention injection probability (validators.inject_mention)
    )
    behavioral_donor_lowercase_rate: float = (
        0.964  # donor casing coin-flip (validators.apply_donor_casing)
    )
    behavioral_rng_seed: int | None = None  # fixed seed for determinism; None = per-process random
    # Phase 5: persist the 45-min delayed feedback observation to a DB-backed due-at table
    # (pending_observations) instead of an in-memory asyncio.Queue, so a process restart no
    # longer loses every pending observation. DEFAULT OFF — when False the in-memory queue
    # path is byte-for-byte unchanged. When True, schedule_observation writes a row and a
    # poller (run_due_observation_loop) claims overdue rows and scores them.
    feedback_due_at_enabled: bool = False
    # Observe-only mode: the conversation engine stays fully passive — no perception,
    # no decision, no sends. Ingestion + pipeline are separate services and keep
    # capturing and saving every message to the DB regardless. Turn on to "watch and
    # record" a group without the bot ever speaking.
    observe_only: bool = False


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    return value if isinstance(value, dict) else {}


def load_engine_config(path: str | Path = "config.toml") -> EngineConfig:
    load_dotenv()
    raw: dict[str, Any] = {}
    file_path = Path(path)
    if file_path.exists():
        raw = tomllib.loads(file_path.read_text())

    scheduler_raw = _section(raw, "scheduler")
    worker_pool_size = int(os.getenv("WORKER_POOL_SIZE", scheduler_raw.get("worker_pool_size", 5)))

    active_chat_ids = _split_ints(os.getenv("ACTIVE_CHAT_IDS"))
    if not active_chat_ids:
        active_chat_ids = _split_ints(os.getenv("MONITORED_CHAT_IDS"))

    return EngineConfig(
        active_chat_ids=active_chat_ids,
        xai_api_key=os.getenv("XAI_API_KEY", ""),
        xai_base_url=os.getenv("XAI_BASE_URL", "https://api.x.ai/v1"),
        conversation_tg_session_name=os.getenv("CONVERSATION_TG_SESSION_NAME", "conversation"),
        local_style_rewrite_enabled=os.getenv("LOCAL_STYLE_REWRITE_ENABLED", "false").lower()
        == "true",
        local_style_python=os.getenv("LOCAL_STYLE_PYTHON", "python3"),
        local_style_chat_script=os.getenv("LOCAL_STYLE_CHAT_SCRIPT", ""),
        local_style_model_path=os.getenv("LOCAL_STYLE_MODEL_PATH", ""),
        local_style_timeout_seconds=int(os.getenv("LOCAL_STYLE_TIMEOUT_SECONDS", "120")),
        use_local_model_for_responses=os.getenv("USE_LOCAL_MODEL_FOR_RESPONSES", "false").lower()
        == "true",
        local_inference_mode=os.getenv("LOCAL_INFERENCE_MODE", "subprocess").lower(),
        local_inference_url=os.getenv("LOCAL_INFERENCE_URL", ""),
        voice_mode=os.getenv("VOICE_MODE", "standalone").lower(),
        emoji_window=int(os.getenv("EMOJI_WINDOW", "5")),
        timing_classifier_enabled=os.getenv("TIMING_CLASSIFIER_ENABLED", "true").lower() == "true",
        timing_classifier_model_path=os.getenv(
            "TIMING_CLASSIFIER_MODEL_PATH", "models/timing_classifier_v2.json"
        ),
        timing_classifier_threshold=float(os.getenv("TIMING_CLASSIFIER_THRESHOLD", "0.0")),
        timing_classifier_shadow=os.getenv("TIMING_CLASSIFIER_SHADOW", "false").lower() == "true",
        cloud_brain_enabled=os.getenv("CLOUD_BRAIN_ENABLED", "true").lower() == "true",
        behavioral_layer_enabled=os.getenv("BEHAVIORAL_LAYER_ENABLED", "false").lower() == "true",
        behavioral_allow_media=os.getenv("BEHAVIORAL_ALLOW_MEDIA", "false").lower() == "true",
        behavioral_burst_rate=float(os.getenv("BEHAVIORAL_BURST_RATE", "0.20")),
        behavioral_mention_rate=float(os.getenv("BEHAVIORAL_MENTION_RATE", "0.0")),
        behavioral_donor_lowercase_rate=float(
            os.getenv("BEHAVIORAL_DONOR_LOWERCASE_RATE", "0.964")
        ),
        behavioral_rng_seed=(
            int(os.environ["BEHAVIORAL_RNG_SEED"]) if os.getenv("BEHAVIORAL_RNG_SEED") else None
        ),
        feedback_due_at_enabled=os.getenv("FEEDBACK_DUE_AT_ENABLED", "false").lower() == "true",
        observe_only=os.getenv("OBSERVE_ONLY", "false").lower() == "true",
        persona=PersonaConfig(**_section(raw, "persona")),
        ai=AiConfig(**_section(raw, "ai")),
        prompt=PromptConfig(**_section(raw, "prompt")),
        scheduler=SchedulerConfig(**{**scheduler_raw, "worker_pool_size": worker_pool_size}),
        circuit_breaker=CircuitBreakerConfig(**_section(raw, "circuit_breaker")),
        persona_engine=PersonaEngineConfig(**_section(raw, "persona_engine")),
        feedback_loop=FeedbackLoopConfig(**_section(raw, "feedback_loop")),
        engagement_gate=EngagementGateConfig(**_section(raw, "engagement_gate")),
    )
