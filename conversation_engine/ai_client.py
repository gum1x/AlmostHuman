from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from conversation_engine.config import EngineConfig

try:
    from anthropic import AsyncAnthropic
except ImportError:  # pragma: no cover
    AsyncAnthropic = None


class PerceptionDecision(BaseModel):
    should_respond: bool = False
    confidence: float = 0.0
    reasoning: str = ""
    entry_points: list[int] = Field(default_factory=list)
    target_message_id: int | None = None
    topic: str | None = None
    risks: str = ""
    annoying_reason: str = ""


class ResponseDecision(BaseModel):
    should_respond: bool = False
    confidence: float = 0.0
    response_text: str | None = None
    reply_to_message_id: int | None = None
    reply_to_user_id: int | None = None
    target_message_id: int | None = None
    reasoning: str = ""
    semantic_risk: str = ""
    annoying_reason: str = ""
    tone_calibration: str | None = None
    stances: dict[str, Any] = Field(default_factory=dict)
    feedback_informed: bool = False


class RelationshipUpdate(BaseModel):
    user_id: int
    notes: str


class ReflectionOutput(BaseModel):
    reflection_text: str
    updated_summary: str
    drift_score: float = 0.0
    drift_explanation: str = ""
    relationship_updates: list[RelationshipUpdate] = Field(default_factory=list)
    tone_adjustments: str = ""


class TonePreference(BaseModel):
    user_id: int
    preferred_tone: str


class TopicPerformance(BaseModel):
    topic: str
    verdict: str


class StanceRecommendation(BaseModel):
    topic: str
    recommended_approach: str


class MetaReflectionOutput(BaseModel):
    what_works: str = ""
    what_doesnt: str = ""
    tone_preferences_by_user: list[TonePreference] = Field(default_factory=list)
    topic_performance: list[TopicPerformance] = Field(default_factory=list)
    updated_stance_recommendations: list[StanceRecommendation] = Field(default_factory=list)


@dataclass(frozen=True)
class AiCallResult:
    text: str
    latency_ms: int
    tokens_used: int


class AnthropicAiClient:
    def __init__(self, config: EngineConfig):
        if AsyncAnthropic is None:
            raise RuntimeError("anthropic package is not installed")
        if not config.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required")
        self.config = config
        base_url = os.getenv("ANTHROPIC_BASE_URL") or None
        self._client = AsyncAnthropic(api_key=config.anthropic_api_key, base_url=base_url)

    async def call_perception_model(self, prompt: str, system: str | None = None) -> AiCallResult:
        return await self._call(self.config.ai.perception_model, prompt, system)

    async def call_decision_model(self, prompt: str, system: str | None = None) -> AiCallResult:
        return await self._call(self.config.ai.decision_model, prompt, system)

    async def _call(self, model: str, prompt: str, system: str | None) -> AiCallResult:
        started = time.perf_counter()
        response = await self._client.messages.create(
            model=model,
            max_tokens=2048,
            temperature=0.2,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        parts = []
        for block in response.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        usage = getattr(response, "usage", None)
        tokens = int(getattr(usage, "input_tokens", 0) or 0) + int(getattr(usage, "output_tokens", 0) or 0)
        return AiCallResult(text="\n".join(parts), latency_ms=latency_ms, tokens_used=tokens)


class FakeAiClient:
    async def call_perception_model(self, prompt: str, system: str | None = None) -> AiCallResult:
        if "Task: self_reflection" in prompt:
            return AiCallResult(
                text=json.dumps(
                    {
                        "reflection_text": "No production model is configured, so this is a placeholder reflection.",
                        "updated_summary": "The bot should remain concise, careful, and low-frequency.",
                        "drift_explanation": "No drift measured by fake client.",
                        "relationship_updates": [],
                        "tone_adjustments": "No changes.",
                    }
                ),
                latency_ms=0,
                tokens_used=0,
            )
        if "Task: meta_reflection" in prompt:
            return AiCallResult(
                text=json.dumps(
                    {
                        "what_works": "unknown",
                        "what_doesnt": "unknown",
                        "tone_preferences_by_user": [],
                        "topic_performance": [],
                        "updated_stance_recommendations": [],
                    }
                ),
                latency_ms=0,
                tokens_used=0,
            )
        if "Task: outcome_scoring" in prompt:
            return AiCallResult(
                text=json.dumps({"outcome": "neutral", "score": 0.0}),
                latency_ms=0,
                tokens_used=0,
            )
        return AiCallResult(
            text=json.dumps({"should_respond": False, "confidence": 0.0, "reasoning": "fake client"}),
            latency_ms=0,
            tokens_used=0,
        )

    async def call_decision_model(self, prompt: str, system: str | None = None) -> AiCallResult:
        return AiCallResult(
            text=json.dumps(
                {
                    "should_respond": False,
                    "confidence": 0.0,
                    "response_text": None,
                    "target_message_id": None,
                    "reasoning": "fake client",
                    "semantic_risk": "",
                    "annoying_reason": "",
                    "feedback_informed": False,
                }
            ),
            latency_ms=0,
            tokens_used=0,
        )


def extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("AI response did not contain a JSON object")
    return stripped[start : end + 1]


def parse_perception(text: str) -> PerceptionDecision:
    return PerceptionDecision.model_validate_json(extract_json_object(text))


def parse_response_decision(text: str) -> ResponseDecision:
    return ResponseDecision.model_validate_json(extract_json_object(text))


def parse_reflection(text: str) -> ReflectionOutput:
    return ReflectionOutput.model_validate_json(extract_json_object(text))


def parse_meta_reflection(text: str) -> MetaReflectionOutput:
    return MetaReflectionOutput.model_validate_json(extract_json_object(text))
