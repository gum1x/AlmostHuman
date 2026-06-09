from __future__ import annotations

import asyncio
import os
import re

import httpx

from conversation_engine.config import EngineConfig
from core.logging import get_logger

log = get_logger(__name__)


class LocalStyleRewriter:
    def __init__(self, config: EngineConfig):
        self.config = config

    @property
    def enabled(self) -> bool:
        if not self.config.local_style_rewrite_enabled:
            return False

        mode = getattr(self.config, "local_inference_mode", "subprocess")
        if mode == "http":
            return bool(getattr(self.config, "local_inference_url", ""))

        # subprocess mode
        return (
            bool(self.config.local_style_python)
            and bool(self.config.local_style_chat_script)
            and bool(self.config.local_style_model_path)
        )

    async def rewrite(self, *, context: str, decision: str, draft: str) -> str:
        """Legacy post-draft rewrite path. Prefer .phrase() in the new hybrid (smart plan + local phrasing)."""
        if not self.enabled or not draft.strip():
            return draft

        prompt = self._build_prompt(context=context, decision=decision, draft=draft)
        return await self._run_local_model(prompt, log_event="local_style_rewrite")

    async def generate_response(self, *, context: str, incoming_message: str) -> str:
        """Directly ask the local fine-tuned model to respond to the message (legacy full-gen path)."""
        if not self.enabled or not incoming_message.strip():
            return ""

        prompt = self._build_generation_prompt(context=context, incoming_message=incoming_message)
        return await self._run_local_model(prompt, log_event="local_direct_response")

    # System message MUST match scripts/build_voice_training.py::SYSTEM_MSG and
    # scripts/voice_generate_shim.py::VOICE_SYSTEM exactly, so the served prompt
    # matches what the model was fine-tuned on (train==serve).
    VOICE_SYSTEM = (
        "You are an active member of a crypto/Telegram handle-trading group chat. "
        "Reply naturally: short, casual, lowercase, blunt. Typos are fine. "
        "Most replies are 1-6 words. Keep the energy of the chat. "
        "Output only your message, nothing else."
    )

    @staticmethod
    def build_voice_context(enriched_messages, max_lines: int = 2) -> str:
        """Build the voice model's context EXACTLY as scripts/build_voice_training.py did:
        raw "u<sender_id>: <text>" lines joined by newlines, most-recent last. This is the
        single source of truth for train==serve on the voice path — pass the enriched
        message list (not the smart-model context.context, which has persona/signal blocks
        the voice model never saw and which wreck its output).
        """
        lines = []
        for m in enriched_messages or []:
            uid = getattr(m, "sender_id", None)
            txt = (getattr(m, "cleaned_text", None) or getattr(m, "text", None) or "").strip()
            if uid is None or not txt:
                continue
            lines.append(f"u{uid}: {txt}")
        return "\n".join(lines[-max_lines:])

    async def generate_voice(self, *, context: str) -> str:
        """Standalone-generator path (advisor's format): the fine-tuned voice model
        receives ONLY the raw recent context (formatted "uXXXX: text" lines) and
        produces the reply itself. No plan/intent signal — the model trained on a
        single regular's real (context -> reply) pairs handles voice end to end.

        This matches scripts/build_voice_training.py: system=VOICE_SYSTEM,
        user=<context block>, assistant=<reply>. The smart decision model still
        decides WHETHER to speak; this only decides the WORDS.

        `context` should already be clean "uXXXX: text" lines (use build_voice_context()
        to produce them from enriched messages). _build_voice_prompt() defensively strips
        any non-chat scaffolding that slips through.
        """
        if not self.enabled or not context.strip():
            return ""
        prompt = self._build_voice_prompt(context=context)
        return await self._run_local_model(prompt, log_event="local_voice_generate")

    async def phrase(self, *, context: str, plan: str, target_message: str = "", tone: str = "") -> str:
        """Core hybrid path: local fine-tuned model phrases the actual reply text.

        Smart model (Grok) provides the high-level control: the 'plan' (what we are actually doing,
        intent, angle, meaning). The local model (LoRA trained on group history) renders it
        in authentic voice, brevity, and rhythm.

        This replaces post-hoc "rewrite my draft" with smart-cognition + local-phrasing.
        """
        if not self.enabled or not plan.strip():
            return ""

        prompt = self._build_phrasing_prompt(
            context=context,
            plan=plan,
            target_message=target_message,
            tone=tone,
        )
        return await self._run_local_model(prompt, log_event="local_style_phrase")

    async def _run_local_model(self, prompt: str, log_event: str) -> str:
        mode = getattr(self.config, "local_inference_mode", "subprocess")

        if mode == "http":
            url = getattr(self.config, "local_inference_url", "")
            if not url:
                await log.awarning(f"{log_event}_failed", error="LOCAL_INFERENCE_URL not set for http mode")
                return ""
            try:
                async with httpx.AsyncClient(timeout=self.config.local_style_timeout_seconds) as client:
                    resp = await client.post(
                        url,
                        json={
                            "prompt": prompt,
                            "max_tokens": 80,
                            "temperature": 0.5,
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    text = data.get("text", "")
                    if text:
                        await log.ainfo(log_event + "_applied")
                    return text
            except Exception as exc:
                await log.awarning(f"{log_event}_failed", error=str(exc))
                return ""

        # subprocess mode — only used for local host development
        # If we're here and the python path is bad, fail fast instead of spamming errors
        python_path = self.config.local_style_python or ""
        if not python_path or not os.path.exists(python_path):
            await log.awarning(
                f"{log_event}_failed",
                error=f"Subprocess python not available: {python_path}"
            )
            return ""

        try:
            proc = await asyncio.create_subprocess_exec(
                python_path,
                self.config.local_style_chat_script,
                "--model",
                self.config.local_style_model_path,
                "--max-tokens",
                "80",
                "--temperature",
                "0.5",
                "--prompt",
                prompt,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.config.local_style_timeout_seconds,
            )
        except Exception as exc:
            await log.awarning(f"{log_event}_failed", error=str(exc))
            return ""

        if proc.returncode != 0:
            await log.awarning(
                f"{log_event}_nonzero",
                returncode=proc.returncode,
                stderr=stderr.decode("utf-8", errors="replace")[-1000:],
            )
            return ""

        text = self._extract_model_text(stdout.decode("utf-8", errors="replace"))
        if text:
            await log.ainfo(log_event + "_applied")
        return text or ""

    # Training context lines are strictly "u<digits>: <text>" (build_voice_training.py).
    # Anything else in the engine context (persona prose, "=== HEADERS ===", PRE-COMPUTED
    # SIGNALS, posture lines) was NEVER seen at train time and pushes the model
    # off-distribution into incoherent output, so we strip it here.
    _VOICE_LINE_RE = re.compile(r"^u\d+:\s")

    def _build_voice_prompt(self, *, context: str) -> str:
        """Return ONLY the raw "uXXXX: text" chat lines, matching the training `user`
        field in scripts/build_voice_training.py (nothing appended, no persona/signal
        blocks). The serve shim applies the chat template with VOICE_SYSTEM, reproducing
        the exact training shape. Keep the tail (latest messages matter most).
        """
        ctx = (context or "").strip()
        # Keep only genuine chat lines; drop engine scaffolding the voice model never saw.
        chat_lines = [
            ln.strip() for ln in ctx.splitlines()
            if self._VOICE_LINE_RE.match(ln.strip())
        ]
        if chat_lines:
            return "\n".join(chat_lines[-2:])
        # Fallback: no uXXX-prefixed lines found (older context format) — use the tail
        # of whatever we got rather than nothing.
        lines = [ln for ln in ctx.splitlines() if ln.strip()]
        return "\n".join(lines[-2:]) if lines else ctx

    def _build_generation_prompt(self, *, context: str, incoming_message: str) -> str:
        context_excerpt = context[-2500:] if context else "No recent context."
        return f"""
Recent Telegram group context:
{context_excerpt}

{incoming_message}
""".strip()

    def _build_phrasing_prompt(
        self, *, context: str, plan: str, target_message: str = "", tone: str = ""
    ) -> str:
        """Build prompt for local model: it receives the smart model's plan as the 'what to do'
        and must produce only the styled utterance. Matches the minimal style that performed
        well in direct evals, augmented with explicit control signal from the capable model.
        """
        context_excerpt = context[-2200:] if context else "No recent context."
        target = (target_message or "").strip()
        plan_clean = (plan or "").strip()
        tone_part = f"\nTone direction: {tone}" if tone.strip() else ""

        return f"""
Recent Telegram group context:
{context_excerpt}

Target message:
{target}

Smart model intent (it only roughly decided the meaning/angle — you turn this into the actual words and rhythm):
{plan_clean}{tone_part}

Phrase a natural, short reply in the group's exact Telegram voice and rhythm that carries out the above intent.
You handle all the low-level phrasing, slang, brevity, and feel. Economy of words. Reactive. Return only the reply text.
""".strip()

    def _build_prompt(self, *, context: str, decision: str, draft: str) -> str:
        context_excerpt = context[-3000:] if context else "No context available."
        return f"""
Recent Telegram context:
{context_excerpt}

Decision from main AI:
{decision}

Draft to preserve:
{draft}

Task: Rewrite the draft in the target Telegram style. Keep the same meaning. Return only the final reply.
""".strip()

    def _extract_model_text(self, output: str) -> str:
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        for line in lines:
            if line.startswith("MODEL:"):
                return line.removeprefix("MODEL:").strip()
        for idx, line in enumerate(lines):
            if line == "Model loaded" and idx + 1 < len(lines):
                next_line = lines[idx + 1].strip()
                if not next_line.startswith("PROMPT:"):
                    return next_line
        return lines[-1].strip() if lines else ""
