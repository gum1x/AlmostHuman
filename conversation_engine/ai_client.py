from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import BaseModel, Field

from conversation_engine.config import EngineConfig


class PerceptionDecision(BaseModel):
    should_respond: bool = False
    confidence: float = 0.0
    reasoning: str = ""
    entry_points: list[int] = Field(default_factory=list)
    target_message_id: int | None = None
    topic: str | None = None
    risks: str = ""
    annoying_reason: str = ""


class AnalyzedSignals(BaseModel):
    """Quantitative signals the AI model analyzes from context before deciding."""
    direct_address_score: float = 0.0  # 0-1: how directly is the bot being addressed
    social_debt: float = 0.0  # 0-1: obligation to respond (unanswered question, active thread, etc.)
    candidate_value_score: int = 0  # 0-100: how valuable is responding to this moment
    persona_relevance: float = 0.0  # 0-1: how relevant is this to the character's interests/expertise


class ResponseDecision(BaseModel):
    should_respond: bool = False
    confidence: float = 0.0
    signals: AnalyzedSignals = Field(default_factory=AnalyzedSignals)  # AI-analyzed quantitative signals
    response_text: str | None = None
    reply_to_message_id: int | None = None
    reply_to_user_id: int | None = None
    target_message_id: int | None = None
    topic: str | None = None
    reasoning: str = ""
    plan: str = ""  # High-level intent/strategy from smart model. Used to guide local phrasing model. "what we are actually doing"
    semantic_risk: str = ""
    annoying_reason: str = ""
    tone_calibration: str | None = None
    stances: dict[str, Any] = Field(default_factory=dict)
    feedback_informed: bool = False
    updated_engagement_posture: str | None = None  # Optional note from the character about shift in its own energy/mode (for persistent rhythm)


class ContextSummary(BaseModel):
    relevant_context: bool = False
    summary: str = ""
    target_message_id: int | None = None
    context_message_ids: list[int] = Field(default_factory=list)
    reasoning: str = ""


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


class GrokAiClient:
    def __init__(self, config: EngineConfig):
        key = config.xai_api_key or "sk-local"
        self.config = config
        self._client = httpx.AsyncClient(
            base_url=config.xai_base_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(120.0, connect=15.0),  # more generous for local models
        )

    async def call_perception_model(self, prompt: str, system: str | None = None) -> AiCallResult:
        return await self._call(
            self.config.ai.perception_model,
            prompt,
            system,
            cache_key="perception",
            temperature=0.2,
        )

    async def call_decision_model(self, prompt: str, system: str | None = None) -> AiCallResult:
        return await self._call(
            self.config.ai.decision_model,
            prompt,
            system,
            cache_key="decision",
            temperature=0.8,
        )

    async def _call(self, model: str, prompt: str, system: str | None, cache_key: str, temperature: float = 0.2) -> AiCallResult:
        started = time.perf_counter()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await self._client.post(
            "/chat/completions",
            headers={"x-grok-conv-id": f"{self.config.ai.prompt_version}:{cache_key}"},
            json={
                "model": model,
                "messages": messages,
                "max_tokens": self.config.ai.max_output_tokens,
                "temperature": temperature,
            },
        )
        response.raise_for_status()
        payload = response.json()
        latency_ms = int((time.perf_counter() - started) * 1000)
        choices = payload.get("choices") or []
        message = choices[0].get("message", {}) if choices else {}
        content = message.get("content") or ""
        usage = payload.get("usage") or {}
        tokens = int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0)
        return AiCallResult(text=content, latency_ms=latency_ms, tokens_used=tokens)

    async def close(self) -> None:
        await self._client.aclose()


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
        if "Summarize only context needed" in prompt:
            return AiCallResult(
                text=json.dumps(
                    {
                        "relevant_context": False,
                        "summary": "",
                        "target_message_id": None,
                        "context_message_ids": [],
                        "reasoning": "fake client",
                    }
                ),
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
                    "signals": {
                        "direct_address_score": 0.0,
                        "social_debt": 0.0,
                        "candidate_value_score": 0,
                        "persona_relevance": 0.0,
                    },
                    "should_respond": False,
                    "confidence": 0.0,
                    "response_text": None,
                    "target_message_id": None,
                    "reasoning": "fake client",
                    "plan": "",
                    "semantic_risk": "",
                    "annoying_reason": "",
                    "feedback_informed": False,
                    "updated_engagement_posture": None,
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


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0", ""}:
            return False
    return bool(value)


def parse_perception(text: str) -> PerceptionDecision:
    return PerceptionDecision.model_validate_json(extract_json_object(text))


def parse_response_decision(text: str) -> ResponseDecision:
    payload = json.loads(extract_json_object(text))
    confidence = payload.get("confidence")
    if isinstance(confidence, (int, float)) and confidence > 1:
        payload["confidence"] = min(float(confidence) / 100, 1.0)
    if payload.get("response_text") == "":
        payload["response_text"] = None
    if not isinstance(payload.get("semantic_risk"), str):
        payload["semantic_risk"] = str(payload.get("semantic_risk") or "")
    if not isinstance(payload.get("reasoning"), str):
        payload["reasoning"] = str(payload.get("reasoning") or "")
    if not isinstance(payload.get("plan"), str):
        payload["plan"] = str(payload.get("plan") or payload.get("reasoning") or "")
    if not isinstance(payload.get("annoying_reason"), str):
        payload["annoying_reason"] = str(payload.get("annoying_reason") or "")
    if not isinstance(payload.get("tone_calibration"), str | type(None)):
        payload["tone_calibration"] = str(payload.get("tone_calibration") or "")
    if not isinstance(payload.get("stances"), dict):
        payload["stances"] = {}
    if not isinstance(payload.get("updated_engagement_posture"), str | type(None)):
        payload["updated_engagement_posture"] = None
    # Coerce signals sub-object
    if not isinstance(payload.get("signals"), dict):
        payload["signals"] = {}
    signals = payload["signals"]
    for float_key in ("direct_address_score", "social_debt", "persona_relevance"):
        if float_key in signals:
            try:
                signals[float_key] = max(0.0, min(1.0, float(signals[float_key])))
            except (ValueError, TypeError):
                signals[float_key] = 0.0
    if "candidate_value_score" in signals:
        try:
            signals["candidate_value_score"] = max(0, min(100, int(signals["candidate_value_score"])))
        except (ValueError, TypeError):
            signals["candidate_value_score"] = 0
    return ResponseDecision.model_validate(payload)


def parse_context_summary(text: str) -> ContextSummary:
    payload = json.loads(extract_json_object(text))
    payload["relevant_context"] = _coerce_bool(payload.get("relevant_context"))
    if not isinstance(payload.get("summary"), str):
        payload["summary"] = str(payload.get("summary") or "")
    if not isinstance(payload.get("reasoning"), str):
        payload["reasoning"] = str(payload.get("reasoning") or "")
    if not isinstance(payload.get("context_message_ids"), list):
        payload["context_message_ids"] = []
    if not payload["relevant_context"]:
        payload["summary"] = ""
        payload["context_message_ids"] = []
    return ContextSummary.model_validate(payload)


def parse_reflection(text: str) -> ReflectionOutput:
    return ReflectionOutput.model_validate_json(extract_json_object(text))


def parse_meta_reflection(text: str) -> MetaReflectionOutput:
    return MetaReflectionOutput.model_validate_json(extract_json_object(text))
