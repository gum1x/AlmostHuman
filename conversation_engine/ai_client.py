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


class ResponseDecision(BaseModel):
    # NOTE: the prior quantitative decision model (STEP 2 inferred scores + AnalyzedSignals)
    # has been removed. The decision prompt now uses the qualitative three-question model
    # ("What kind of situation...", "What kind of person am I?", "What does a person like me do...?").
    # "signals" is no longer part of the output contract (defensively popped in parse).
    should_respond: bool = False
    confidence: float = 0.0
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
    # New richer output from the high/recent compressor (the actual context
    # passed to the 3Q reasoning AI). Legacy "summary" kept for compat.
    compressed_relevant_context: str = ""
    high_level_included: bool = False
    direct_mention_or_continuation: bool = False
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
                # All prompts demand a single JSON object. Asking the provider to
                # enforce JSON output cuts down on markdown fences / leaked prose
                # (notably from DeepSeek) that would otherwise need salvaging.
                "response_format": {"type": "json_object"},
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
                        "updated_summary": "The bot should be bold, eager to engage, and useful — high frequency is fine when making friends or enemies.",
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
        if "Summarize only context needed" in prompt or "HIGH-LEVEL CONTEXT" in prompt or "compressed_relevant_context" in prompt:
            # Return a payload that exercises the new high/recent compressor fields.
            # In real runs with a long chat the perception model will actually decide
            # relevance and may quote exact prior details + set direct_mention_or_continuation.
            return AiCallResult(
                text=json.dumps(
                    {
                        "relevant_context": True,
                        "summary": "recent context only (fake)",
                        "compressed_relevant_context": ("target + recent; HIGH-LEVEL RELEVANT: " + ("foo deal 1.2m (pulled from high)" if "foo deal" in prompt.lower() else "none")) if "HIGH-LEVEL CONTEXT" in prompt else "target + recent window (fake)",
                        "high_level_included": "foo deal" in prompt.lower() if "HIGH-LEVEL CONTEXT" in prompt else False,
                        "direct_mention_or_continuation": ("@" in prompt or "direct" in prompt.lower() or "reply_to" in prompt.lower()),
                        "target_message_id": None,
                        "context_message_ids": [],
                        "reasoning": "fake client - recent self-contained (or direct=false)",
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
    """Pull the first balanced JSON object out of a model response.

    Robust to markdown ```json fences, reasoning/prose emitted before or after
    the object, and braces appearing inside JSON string values. Some models
    (e.g. DeepSeek) frequently fence the JSON or leak chain-of-thought around
    it, which the old outermost find('{')/rfind('}') approach mishandled.
    """
    stripped = text.strip()
    # Strip a leading code fence (``` or ```json) if present.
    if stripped.startswith("```"):
        stripped = stripped.split("```", 2)
        stripped = stripped[1] if len(stripped) > 1 else text
        if stripped.lstrip().lower().startswith("json"):
            stripped = stripped.lstrip()[4:]

    # Scan for the first top-level balanced { ... }, ignoring braces inside strings.
    start = stripped.find("{")
    while start != -1:
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(stripped)):
            ch = stripped[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return stripped[start : i + 1]
        # Unbalanced from this start; try the next brace.
        start = stripped.find("{", start + 1)

    raise ValueError("AI response did not contain a JSON object")


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
    # Old outputs may still contain a "signals" object from the prior quantitative model.
    # The qualitative three-question model no longer emits it; pop defensively.
    payload.pop("signals", None)
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
    # New compressor fields (high/recent relevance for 3Q reasoning AI)
    if not isinstance(payload.get("compressed_relevant_context"), str):
        payload["compressed_relevant_context"] = str(payload.get("compressed_relevant_context") or "")
    payload["high_level_included"] = _coerce_bool(payload.get("high_level_included"))
    payload["direct_mention_or_continuation"] = _coerce_bool(payload.get("direct_mention_or_continuation"))
    # Compat: if the new compressed field is missing but legacy summary exists, use it
    if not payload.get("compressed_relevant_context") and payload.get("summary"):
        payload["compressed_relevant_context"] = payload["summary"]
    if not payload["relevant_context"]:
        payload["summary"] = ""
        payload["compressed_relevant_context"] = ""
        payload["context_message_ids"] = []
        payload["high_level_included"] = False
        payload["direct_mention_or_continuation"] = False
    return ContextSummary.model_validate(payload)


def parse_reflection(text: str) -> ReflectionOutput:
    return ReflectionOutput.model_validate_json(extract_json_object(text))


def parse_meta_reflection(text: str) -> MetaReflectionOutput:
    return MetaReflectionOutput.model_validate_json(extract_json_object(text))
