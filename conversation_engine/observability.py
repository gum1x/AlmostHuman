from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from core.logging import get_logger

log = get_logger(__name__)


@dataclass
class Metrics:
    gauges: dict[str, float] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def gauge(self, name: str, value: float, **labels: Any) -> None:
        key = _metric_key(name, labels)
        self.gauges[key] = value

    def increment(self, name: str, amount: int = 1, **labels: Any) -> None:
        key = _metric_key(name, labels)
        self.counters[key] += amount


def _metric_key(name: str, labels: dict[str, Any]) -> str:
    if not labels:
        return name
    rendered = ",".join(f"{key}={value}" for key, value in sorted(labels.items()))
    return f"{name}{{{rendered}}}"


metrics = Metrics()


def record_gate(gate_score: float, factors: dict[str, float]) -> None:
    metrics.gauge("conversation_engine.gate.score", gate_score)
    for factor, value in factors.items():
        if isinstance(value, int | float):
            metrics.gauge(f"conversation_engine.gate.factor.{factor}", float(value))


def record_feedback(outcome: str, score: float) -> None:
    metrics.increment("conversation_engine.feedback.outcome", outcome=outcome)
    metrics.gauge("conversation_engine.feedback.score", score)


def record_persona_drift(score: float) -> None:
    metrics.gauge("conversation_engine.persona.drift_score", score)


def record_reflection_triggered(trigger: str) -> None:
    metrics.increment("conversation_engine.reflection.triggered", trigger=trigger)


def record_vector_memory_retrieved(count: int) -> None:
    metrics.increment("conversation_engine.vector_memory.retrieved", amount=count)
