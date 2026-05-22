from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _split_ints(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class PersonaConfig:
    identity: str = "A careful, low-frequency participant in a Telegram group."
    core_beliefs: list[str] = field(default_factory=list)
    speaking_style: str = "concise, specific, and calm"


@dataclass(frozen=True)
class AiConfig:
    perception_model: str = "claude-haiku-4-5-20251001"
    decision_model: str = "claude-sonnet-4-6"
    total_context_token_budget: int = 80_000
    min_confidence_to_send: float = 0.6
    prompt_version: str = "v1.0"
    persona_top_k: int = 5


@dataclass(frozen=True)
class PromptConfig:
    engagement_style: str = "lurker"
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


@dataclass(frozen=True)
class EngineConfig:
    active_chat_ids: list[int]
    anthropic_api_key: str
    conversation_tg_session_name: str
    persona: PersonaConfig
    ai: AiConfig
    prompt: PromptConfig
    scheduler: SchedulerConfig
    circuit_breaker: CircuitBreakerConfig
    persona_engine: PersonaEngineConfig
    feedback_loop: FeedbackLoopConfig
    engagement_gate: EngagementGateConfig


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    return value if isinstance(value, dict) else {}


def load_engine_config(path: str | Path = "config.toml") -> EngineConfig:
    raw: dict[str, Any] = {}
    file_path = Path(path)
    if file_path.exists():
        raw = tomllib.loads(file_path.read_text())

    scheduler_raw = _section(raw, "scheduler")
    worker_pool_size = int(os.getenv("WORKER_POOL_SIZE", scheduler_raw.get("worker_pool_size", 5)))

    return EngineConfig(
        active_chat_ids=_split_ints(os.getenv("ACTIVE_CHAT_IDS")),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        conversation_tg_session_name=os.getenv("CONVERSATION_TG_SESSION_NAME", "conversation"),
        persona=PersonaConfig(**_section(raw, "persona")),
        ai=AiConfig(**_section(raw, "ai")),
        prompt=PromptConfig(**_section(raw, "prompt")),
        scheduler=SchedulerConfig(**{**scheduler_raw, "worker_pool_size": worker_pool_size}),
        circuit_breaker=CircuitBreakerConfig(**_section(raw, "circuit_breaker")),
        persona_engine=PersonaEngineConfig(**_section(raw, "persona_engine")),
        feedback_loop=FeedbackLoopConfig(**_section(raw, "feedback_loop")),
        engagement_gate=EngagementGateConfig(**_section(raw, "engagement_gate")),
    )
