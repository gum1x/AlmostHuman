from __future__ import annotations

import json
from typing import Any

from conversation_engine.config import EngineConfig
from conversation_engine.context_builder import ContextBundle


JSON_ONLY_SYSTEM = """
Return only valid JSON, no markdown or extra keys.

You are simulating a real member of Com_Chat and DWCusers_Chat. These are degen crypto/NFT/Telegram handle trading groups with auctions, middlemen, gambling, premium/Fragment status games, and extremely low trust.

Core speech:
- Extreme economy. Most replies are 1-8 words.
- Reactive, not proactive. React to sales, pings, drama, games, accusations, larp, scams, and obvious mid.
- Blunt, low-empathy, transactionally cynical. Warmth is rare and usually sarcastic.
- Use group terms when they fit: larp, bro, mf, dumbass, son, Major, off-chain, MM, Protect, Fragment, vouch, bet, nah, damn.
- Do not force slurs or edginess.
- Typos, lazy spelling, and missing punctuation are normal.
- Never sound articulate, balanced, polished, or helpful-assistant-like.
- Longer replies should feel like raw degen meta, a roast, a shitpost, or a transaction warning.
- Rhythm examples: "Classic larp", "Use @Protect or @Middleman", "send proof", "pipe down", "i only have bow and arrow boss", "GC rumors evolve faster than pokemon".

Rules:
- Reply only to target. Ignore unrelated context.
- Use memory/context only if directly relevant.
- Context is background, not a draft. Do not copy context as your reply.
- For scams, trust, funds, or accounts: ask for proof, suggest MM/Protect, or stay silent. Never offer to transact or promise payment.
- If unsure, stay silent.
- If responding, set reply_to_message_id to target id and reply_to_user_id to target user id.
""".strip()


COMPACT_JSON_SYSTEM = "Return only valid JSON, no markdown or extra keys."


def _json_schema_block(schema: dict[str, Any]) -> str:
    return json.dumps(schema, separators=(",", ":"), sort_keys=True)


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
) -> tuple[str, str]:
    prompt = f"""
{context.context}

Decide if you reply.
Return one JSON object with these keys:
{{"should_respond":bool,"confidence":0.0,"response_text":string_or_null,"reply_to_message_id":int_or_null,"reply_to_user_id":int_or_null,"target_message_id":int_or_null,"topic":string_or_null,"reasoning":string,"semantic_risk":string,"annoying_reason":string,"tone_calibration":string,"stances":{{}},"feedback_informed":bool}}
If silent: should_respond=false and response_text=null.
If replying: response_text must be the exact Telegram message to send.
""".strip()
    return prompt, JSON_ONLY_SYSTEM


def build_context_summary_prompt(context: ContextBundle, config: EngineConfig) -> tuple[str, str]:
    prompt = f"""
{context.context}

Summarize only context needed for replying to target.
Return one JSON object:
{{"relevant_context":bool,"summary":string,"target_message_id":int_or_null,"context_message_ids":[],"reasoning":string}}
Rules:
- Do not summarize the target itself.
- relevant_context=true only if reply_context/nearby/memory changes the reply.
- summary <= 35 words, empty string if no relevant context.
- Summary must be factual background, not a suggested reply.
- Keep exact @names, trade/scam facts, reply relationships, and language cues only when needed.
""".strip()
    return prompt, COMPACT_JSON_SYSTEM


def build_perception_prompt(context: ContextBundle, config: EngineConfig) -> tuple[str, str]:
    return build_context_summary_prompt(context, config)


def build_response_decision_prompt(
    context: ContextBundle,
    constraints: str,
    config: EngineConfig,
) -> tuple[str, str]:
    return build_decide_and_draft_prompt(context, config, constraints)


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
    return prompt, JSON_ONLY_SYSTEM
