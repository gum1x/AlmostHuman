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


def _split_metric_key(key: str) -> tuple[str, str]:
    """Split a stored metric key into ``(prometheus_name, label_block)``.

    Stored keys look like ``name`` or ``name{a=1,b=2}``. Prometheus names may not
    contain ``.``, so dots become underscores; the ``{...}`` label block (if any)
    is passed through unchanged.
    """
    if key.endswith("}") and "{" in key:
        name, _, labels = key.partition("{")
        return name.replace(".", "_"), "{" + labels
    return key.replace(".", "_"), ""


def render_prometheus(m: Metrics | None = None) -> str:
    """Render recorded metrics in Prometheus text exposition format.

    Gauges and counters share the same numeric line shape; counters are emitted
    with a ``# TYPE ... counter`` hint. Names are sanitized (``.`` -> ``_``) and
    deterministically sorted so scrapes/tests are stable.
    """
    m = m if m is not None else metrics
    lines: list[str] = []
    seen_types: set[str] = set()
    for store, mtype in ((m.gauges, "gauge"), (m.counters, "counter")):
        for key in sorted(store):
            name, labels = _split_metric_key(key)
            if name not in seen_types:
                # TYPE may be declared only once per metric family.
                lines.append(f"# TYPE {name} {mtype}")
                seen_types.add(name)
            lines.append(f"{name}{labels} {store[key]}")
    return "\n".join(lines) + ("\n" if lines else "")
