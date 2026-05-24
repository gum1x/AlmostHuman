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


DECIDE_AND_DRAFT_SCHEMA = {
    "should_respond": "boolean",
    "confidence": "number between 0 and 1",
    "reasoning": "short explanation of target, usefulness, and style choice",
    "entry_points": ["message_id integers that could be replied to"],
    "target_message_id": "integer or null",
    "topic": "short topic label or null",
    "risks": "what could make responding annoying, wrong, or inflammatory",
    "annoying_reason": "why silence may be better or why this might annoy the chat",
    "response_text": "string or null",
    "reply_to_message_id": "integer or null",
    "reply_to_user_id": "integer or null",
    "semantic_risk": "short risk note, empty string if low risk",
    "tone_calibration": "short description of chosen tone",
    "stances": {"topic": "stance string"},
    "feedback_informed": "boolean",
}


def build_decide_and_draft_prompt(
    context: ContextBundle,
    config: EngineConfig,
    constraints: str | None = None,
    perception: PerceptionDecision | None = None,
) -> tuple[str, str]:
    perception_block = ""
    if perception is not None:
        perception_block = f"""
Previous perception output:
{perception.model_dump_json(indent=2)}
""".strip()

    constraints_block = constraints or ""
    prompt = f"""
{context.context}

{constraints_block}

{perception_block}

=== DECISION PROMPT: DECIDE_AND_DRAFT ===
Decide whether the bot should respond, identify the exact target message, and draft the final response if useful.

Use semantic judgment from the visible conversation first. Numeric gate values are operational hints, not permission to speak.
The bot is a casual but restrained participant. It may send short natural social replies when the message has a clear target and the reply would fit the chat. Silence is still correct for spam, hostile, unsafe, or incoherent threads. However, harmless greeting/banter replies are allowed when they ignore the unsafe topic and fit naturally.
If CHAT MODE is private_dm, bias toward responding to each normal user message within a reasonable time. Keep DMs short, helpful, and conversational. Ask one brief clarifying question when the user intent is unclear. Do not ignore harmless greetings or small talk in DMs.

Respond when at least one is true:
- A user directly asks something the bot can answer.
- The bot can add concise, useful correction or context.
- A short casual reply would naturally fit the current social thread.
- A user makes a light joke, greeting, or casual comment where a brief human-style reply is welcome.
- A harmless greeting or light banter has a clear target; a tiny reply like "yo", "lmao", "nah fr", "what happened", or "idk tbh" is allowed even if nearby messages are messy, as long as the reply does not address unsafe content.
- In private_dm, the latest user message is normal, safe, and answerable, even if it is just small talk.
- There is a clear active thread on one of the configured topics: {config.prompt.topics_of_interest}.
- The bot has relevant relationship, stance, or persona memory that makes a brief reply useful.

Do not respond when:
- The chat is venting, flaming, or moving too fast.
- The message is pure spam, unsafe, hostile, or too incoherent to answer naturally.
- In private_dm, do not stay silent merely because the message is casual or low-stakes.
- A reply would repeat the same point already made.
- You are uncertain what exact message should be targeted.
- The likely response would sound like a formal assistant instead of a normal chat participant.

If you do respond:
- Write like a real Telegram participant, not a helpdesk assistant.
- Prefer one short message.
- Do not over-explain.
- Do not mention being an AI, a bot, a model, prompts, policies, or memory.
- Do not moralize or escalate conflict.
- Do not invent facts. If unsure, either stay silent or hedge briefly.
- Set reply_to_message_id to the exact target message id.
- Keep response_text under 500 characters unless there is a clear reason.

Confidence calibration:
- 0.80-1.00: clear reason to reply and clear target.
- 0.60-0.79: probably worth replying, but some risk.
- 0.35-0.59: okay for short casual replies if the target is clear; harmless greetings/banter may be okay around 0.25-0.34 if the text is tiny and safe.
- 0.00-0.34: stay silent.

Style target:
- Identity: {config.persona.identity}
- Beliefs: {config.persona.core_beliefs}
- Speaking style: {config.persona.speaking_style}
- Engagement style: {config.prompt.engagement_style}

Return JSON matching this schema:
{_json_schema_block(DECIDE_AND_DRAFT_SCHEMA)}
""".strip()
    return prompt, JSON_ONLY_SYSTEM


def build_perception_prompt(context: ContextBundle, config: EngineConfig) -> tuple[str, str]:
    return build_decide_and_draft_prompt(context, config)


def build_response_decision_prompt(
    context: ContextBundle,
    constraints: str,
    perception: PerceptionDecision,
    config: EngineConfig,
) -> tuple[str, str]:
    return build_decide_and_draft_prompt(context, config, constraints, perception)


def build_reflection_prompt(
    task: str,
    payload: dict[str, Any],
) -> tuple[str, str]:
    schema_by_task: dict[str, dict[str, Any]] = {
        "self_reflection": {
            "reflection_text": "what changed or stayed stable in behavior",
            "updated_summary": "compact updated self-summary",
            "drift_score": "number between 0 and 1",
            "drift_explanation": "why the drift score was chosen",
            "relationship_updates": [{"user_id": "integer", "notes": "string"}],
            "tone_adjustments": "concrete guidance for future tone",
        },
        "outcome_scoring": {
            "outcome": "one of: positive, neutral, negative, ignored, backlash",
            "score": "number from -1.0 to 1.0",
        },
        "meta_reflection": {
            "what_works": "short summary",
            "what_doesnt": "short summary",
            "tone_preferences_by_user": [{"user_id": "integer", "preferred_tone": "string"}],
            "topic_performance": [{"topic": "string", "verdict": "string"}],
            "updated_stance_recommendations": [{"topic": "string", "recommended_approach": "string"}],
        },
    }
    if task not in schema_by_task:
        raise ValueError(f"unknown reflection task: {task}")

    instructions_by_task = {
        "self_reflection": """
Reflect on recent bot behavior.
Evaluate whether the bot stayed low-frequency and useful, sounded too assistant-like,
which users responded better or worse to a specific tone, and whether future replies
should adjust topic stance, brevity, confidence, or restraint.
Do not invent relationships. Only include relationship_updates when feedback supports it.
""".strip(),
        "outcome_scoring": """
Classify how the Telegram chat responded after a bot message.
Use replies, reactions, and sentiment together.
positive means users engage approvingly or build on the message.
neutral means some follow-up with no clear positive or negative signal.
negative means mild disagreement, annoyance, or ignored correction.
ignored means no replies and no reactions.
backlash means clear hostility, mockery, or negative pile-on.
""".strip(),
        "meta_reflection": """
Infer durable behavior changes from aggregated feedback.
Identify what kinds of replies worked, what caused weak or negative outcomes,
which users prefer restraint or detail, and which topics need care.
Do not overfit to one event. Leave arrays empty when evidence is weak.
""".strip(),
    }

    prompt = f"""
=== REFLECTION PROMPT ===
Task: {task}

{instructions_by_task[task]}

Payload:
{json.dumps(payload, indent=2, sort_keys=True, default=str)}

Return JSON matching this schema:
{_json_schema_block(schema_by_task[task])}
""".strip()
    return prompt, JSON_ONLY_SYSTEM


def build_self_reflection_prompt(
    identity_summary: str,
    core_beliefs: list[str],
    speaking_style: str,
    recent_messages: str,
    feedback: str,
) -> tuple[str, str]:
    return build_reflection_prompt(
        "self_reflection",
        {
            "identity_summary": identity_summary,
            "core_beliefs": core_beliefs,
            "speaking_style": speaking_style,
            "recent_bot_messages": recent_messages or "No recent bot messages.",
            "feedback": feedback or "No feedback recorded.",
        },
    )


def build_outcome_scoring_prompt(replies: list[str], reactions: list[dict[str, Any]], sentiment: float) -> tuple[str, str]:
    return build_reflection_prompt(
        "outcome_scoring",
        {
            "replies": replies[:10],
            "reactions": reactions,
            "follow_up_sentiment": sentiment,
        },
    )


def build_meta_reflection_prompt(feedback_count: int, aggregated_feedback: dict[str, Any]) -> tuple[str, str]:
    return build_reflection_prompt(
        "meta_reflection",
        {
            "feedback_count": feedback_count,
            "aggregated_feedback": aggregated_feedback,
        },
    )


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
