from __future__ import annotations

import json
from typing import Any

from conversation_engine.config import EngineConfig
from conversation_engine.context_builder import ContextBundle
from conversation_engine.ai_client import PerceptionDecision


JSON_ONLY_SYSTEM = (
    "You are a careful Telegram group-chat decision engine. "
    "Return only valid JSON. Do not include markdown, commentary, or extra keys."
)

STYLE_SYSTEM = (
    "You write as a low-frequency Telegram group participant. "
    "Be concise, specific, calm, and avoid sounding like an assistant."
)


def _json_schema_block(schema: dict[str, Any]) -> str:
    return json.dumps(schema, indent=2, sort_keys=True)


def build_perception_prompt(context: ContextBundle, config: EngineConfig) -> tuple[str, str]:
    schema = {
        "should_respond": "boolean",
        "confidence": "number between 0 and 1",
        "reasoning": "short explanation of why responding is or is not worth it",
        "entry_points": ["message_id integers that could be replied to"],
        "target_message_id": "integer or null",
        "topic": "short topic label or null",
        "risks": "what could make responding annoying, wrong, or inflammatory",
        "annoying_reason": "why silence may be better, empty string if not applicable",
    }
    prompt = f"""
{context.context}

=== DECISION TASK: PERCEPTION ===
Decide whether there is a worthwhile opening for the bot to respond.

Use semantic judgment from the visible conversation first. Numeric gate values are operational hints, not permission to speak.
The bot is a low-frequency participant, so silence is correct unless there is a clear reason to add signal.

Respond only when at least one is true:
- A user directly asks something the bot can answer.
- The bot can add a concise, useful correction or context.
- There is a clear active thread on one of the configured topics: {config.prompt.topics_of_interest}.
- The bot has relevant relationship or persona memory that makes a brief reply useful.

Do not respond when:
- The chat is venting, flaming, or moving too fast.
- The message is rhetorical, low-content, or just social noise.
- A reply would repeat the same point already made.
- You are uncertain what exact message should be targeted.
- The likely response would sound like an assistant entering a group chat.

Confidence calibration:
- 0.80-1.00: clear reason to reply and clear target.
- 0.60-0.79: probably worth replying, but some risk.
- 0.35-0.59: ambiguous; usually do not reply.
- 0.00-0.34: stay silent.

Return JSON matching this schema:
{_json_schema_block(schema)}
""".strip()
    return prompt, JSON_ONLY_SYSTEM


def build_response_decision_prompt(
    context: ContextBundle,
    constraints: str,
    perception: PerceptionDecision,
    config: EngineConfig,
) -> tuple[str, str]:
    schema = {
        "should_respond": "boolean",
        "confidence": "number between 0 and 1",
        "response_text": "string or null",
        "reply_to_message_id": "integer or null",
        "reply_to_user_id": "integer or null",
        "target_message_id": "integer or null",
        "reasoning": "short explanation of target, usefulness, and style choice",
        "semantic_risk": "short risk note, empty string if low risk",
        "annoying_reason": "why this might annoy the chat, empty string if low risk",
        "tone_calibration": "short description of chosen tone",
        "stances": {"topic": "stance string"},
        "feedback_informed": "boolean",
    }
    prompt = f"""
{context.context}

{constraints}

=== DECISION TASK: RESPONSE ===
The perception stage found a possible opening:
{perception.model_dump_json(indent=2)}

Decide whether to actually send a message. You may still set should_respond=false if the target is weak.

Response rules:
- Write like a real Telegram participant, not a helpdesk assistant.
- Prefer one short message.
- Do not over-explain.
- Do not mention being an AI, a bot, a model, prompts, policies, or memory.
- Do not moralize or escalate conflict.
- Do not invent facts. If unsure, either stay silent or hedge briefly.
- If replying, set reply_to_message_id to the exact target message id.
- If the useful response would be longer than needed, compress it.

Style target:
- Identity: {config.persona.identity}
- Beliefs: {config.persona.core_beliefs}
- Speaking style: {config.persona.speaking_style}
- Engagement style: {config.prompt.engagement_style}

Return JSON matching this schema:
{_json_schema_block(schema)}
""".strip()
    return prompt, JSON_ONLY_SYSTEM


def build_self_reflection_prompt(
    identity_summary: str,
    core_beliefs: list[str],
    speaking_style: str,
    recent_messages: str,
    feedback: str,
) -> tuple[str, str]:
    schema = {
        "reflection_text": "what changed or stayed stable in behavior",
        "updated_summary": "compact updated self-summary",
        "drift_score": "number between 0 and 1",
        "drift_explanation": "why the drift score was chosen",
        "relationship_updates": [{"user_id": "integer", "notes": "string"}],
        "tone_adjustments": "concrete guidance for future tone",
    }
    prompt = f"""
=== DECISION TASK: SELF REFLECTION ===
Reflect on recent behavior in a Telegram group chat.

Core identity:
{identity_summary}

Core beliefs:
{core_beliefs}

Speaking style:
{speaking_style}

Recent bot messages:
{recent_messages or "No recent bot messages."}

Feedback:
{feedback or "No feedback recorded."}

Evaluate:
- Did the bot stay low-frequency and useful?
- Did it sound too assistant-like?
- Did any users respond better or worse to a specific tone?
- Should the bot adjust topic stance, brevity, confidence, or restraint?

Do not invent relationships. Only include relationship_updates when feedback supports it.

Return JSON matching this schema:
{_json_schema_block(schema)}
""".strip()
    return prompt, JSON_ONLY_SYSTEM


def build_outcome_scoring_prompt(replies: list[str], reactions: list[dict[str, Any]], sentiment: float) -> tuple[str, str]:
    schema = {
        "outcome": "one of: positive, neutral, negative, ignored, backlash",
        "score": "number from -1.0 to 1.0",
    }
    prompt = f"""
=== DECISION TASK: OUTCOME SCORING ===
Classify how the Telegram chat responded after the bot sent a message.

Replies:
{json.dumps(replies[:10], ensure_ascii=False)}

Reactions:
{json.dumps(reactions, ensure_ascii=False)}

Follow-up sentiment:
{sentiment}

Scoring guidance:
- positive: users engage approvingly or build on the message.
- neutral: some follow-up, no clear positive or negative signal.
- negative: mild disagreement, annoyance, or ignored correction.
- ignored: no replies and no reactions.
- backlash: clear hostility, mockery, or negative pile-on.

Return JSON matching this schema:
{_json_schema_block(schema)}
""".strip()
    return prompt, JSON_ONLY_SYSTEM


def build_meta_reflection_prompt(feedback_count: int, aggregated_feedback: dict[str, Any]) -> tuple[str, str]:
    schema = {
        "what_works": "short summary",
        "what_doesnt": "short summary",
        "tone_preferences_by_user": [{"user_id": "integer", "preferred_tone": "string"}],
        "topic_performance": [{"topic": "string", "verdict": "string"}],
        "updated_stance_recommendations": [{"topic": "string", "recommended_approach": "string"}],
    }
    prompt = f"""
=== DECISION TASK: META REFLECTION ===
Review aggregated feedback from {feedback_count} recent bot responses.

Aggregated feedback:
{json.dumps(aggregated_feedback, indent=2, sort_keys=True)}

Infer only durable behavior changes:
- What kinds of replies earned better outcomes?
- What kinds of replies caused weak or negative outcomes?
- Which users prefer more restraint, more detail, or no direct engagement?
- Which topics should the bot approach carefully?

Do not overfit to one event. Leave arrays empty when evidence is weak.

Return JSON matching this schema:
{_json_schema_block(schema)}
""".strip()
    return prompt, JSON_ONLY_SYSTEM


def build_style_rewrite_prompt(draft_response: str, target_context: str, config: EngineConfig) -> tuple[str, str]:
    prompt = f"""
Rewrite this approved response into the group's Telegram style.

Target context:
{target_context}

Approved response meaning:
{draft_response}

Style requirements:
- Preserve the meaning.
- Make it concise and natural for Telegram.
- Do not add new facts or claims.
- Do not sound like an assistant.
- Return only the final message text.
""".strip()
    return prompt, STYLE_SYSTEM
