from __future__ import annotations

import asyncio

from conversation_engine.config import EngineConfig
from core.logging import get_logger

log = get_logger(__name__)


class LocalStyleRewriter:
    def __init__(self, config: EngineConfig):
        self.config = config

    @property
    def enabled(self) -> bool:
        return (
            self.config.local_style_rewrite_enabled
            and bool(self.config.local_style_python)
            and bool(self.config.local_style_chat_script)
            and bool(self.config.local_style_model_path)
        )

    async def rewrite(self, *, context: str, decision: str, draft: str) -> str:
        if not self.enabled or not draft.strip():
            return draft

        prompt = self._build_prompt(context=context, decision=decision, draft=draft)
        try:
            proc = await asyncio.create_subprocess_exec(
                self.config.local_style_python,
                self.config.local_style_chat_script,
                "--model",
                self.config.local_style_model_path,
                "--max-tokens",
                "80",
                "--temperature",
                "0.45",
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
            await log.awarning("local_style_rewrite_failed", error=str(exc))
            return draft

        if proc.returncode != 0:
            await log.awarning(
                "local_style_rewrite_nonzero",
                returncode=proc.returncode,
                stderr=stderr.decode("utf-8", errors="replace")[-1000:],
            )
            return draft

        rewritten = self._extract_model_text(stdout.decode("utf-8", errors="replace"))
        if not rewritten:
            return draft
        await log.ainfo("local_style_rewrite_applied")
        return rewritten

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
