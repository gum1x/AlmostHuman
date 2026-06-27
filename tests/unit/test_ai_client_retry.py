"""Offline, deterministic tests for GrokAiClient bounded retry + backoff.

The private _post_with_retry wraps the LLM HTTP POST + raise_for_status with a
bounded exponential-backoff retry that fires ONLY on transient failures
(timeout / transport drop / HTTP 429 / >=500). Non-transient 4xx must not
retry. No network, no real sleeping: asyncio.sleep is monkeypatched to a no-op
and the httpx client is a fake whose post() is fully scripted.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from conversation_engine import ai_client
from conversation_engine.ai_client import GrokAiClient


def _make_client(monkeypatch: pytest.MonkeyPatch, max_retries: int = 3) -> GrokAiClient:
    """Build a GrokAiClient without touching the network or config.toml.

    Bypasses __init__ (which would construct a real httpx.AsyncClient) and wires
    a minimal duck-typed config covering what _call / call_raw / _post_with_retry
    read. asyncio.sleep is patched to a no-op so backoff is instant.
    """
    client = GrokAiClient.__new__(GrokAiClient)
    client.config = SimpleNamespace(
        cloud_brain_enabled=True,
        ai=SimpleNamespace(
            decision_model="test-model",
            perception_model="test-model",
            max_output_tokens=64,
            prompt_version="test",
            max_retries=max_retries,
        ),
    )

    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(ai_client.asyncio, "sleep", _no_sleep)
    # Make jitter deterministic (and zero) so nothing depends on randomness.
    monkeypatch.setattr(ai_client.random, "uniform", lambda _a, _b: 0.0)
    return client


class _FakeResponse:
    def __init__(self, status_code: int = 200, payload: dict[str, Any] | None = None):
        self.status_code = status_code
        self._payload = payload or {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"status {self.status_code}",
                request=httpx.Request("POST", "http://test/chat/completions"),
                response=httpx.Response(self.status_code),
            )

    def json(self) -> dict[str, Any]:
        return self._payload


class _ScriptedClient:
    """Fake httpx.AsyncClient.post that replays a list of outcomes in order.

    Each item is either an Exception instance (raised) or a _FakeResponse
    (returned). Records how many times post() was invoked.
    """

    def __init__(self, outcomes: list[Any]):
        self._outcomes = list(outcomes)
        self.calls = 0

    async def post(self, _url: str, **_kwargs: Any) -> _FakeResponse:
        self.calls += 1
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _timeout() -> httpx.TimeoutException:
    return httpx.ReadTimeout("boom", request=httpx.Request("POST", "http://test"))


def _status(code: int) -> httpx.HTTPStatusError:
    return httpx.HTTPStatusError(
        f"status {code}",
        request=httpx.Request("POST", "http://test"),
        response=httpx.Response(code),
    )


async def test_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch):
    """Two transient blips (timeout, then 503) then a success -> 3 calls, returns content."""
    client = _make_client(monkeypatch, max_retries=3)
    client._client = _ScriptedClient([_timeout(), _status(503), _FakeResponse()])

    result = await client.call_decision_model("hi", system=None)

    assert result.text == "ok"
    assert client._client.calls == 3


async def test_gives_up_after_cap(monkeypatch: pytest.MonkeyPatch):
    """A transient error on every attempt -> exactly max_retries calls, then re-raises."""
    client = _make_client(monkeypatch, max_retries=3)
    client._client = _ScriptedClient([_timeout(), _timeout(), _timeout()])

    with pytest.raises(httpx.TimeoutException):
        await client.call_decision_model("hi", system=None)

    assert client._client.calls == 3


async def test_400_does_not_retry(monkeypatch: pytest.MonkeyPatch):
    """A non-transient 4xx (400) must NOT retry: exactly one call, then re-raises."""
    client = _make_client(monkeypatch, max_retries=3)
    client._client = _ScriptedClient([_status(400), _FakeResponse()])

    with pytest.raises(httpx.HTTPStatusError):
        await client.call_decision_model("hi", system=None)

    assert client._client.calls == 1


async def test_429_is_retried(monkeypatch: pytest.MonkeyPatch):
    """429 (rate limit) is transient: retries once then succeeds -> 2 calls."""
    client = _make_client(monkeypatch, max_retries=3)
    client._client = _ScriptedClient([_status(429), _FakeResponse()])

    result = await client.call_decision_model("hi", system=None)

    assert result.text == "ok"
    assert client._client.calls == 2


async def test_call_raw_also_retries(monkeypatch: pytest.MonkeyPatch):
    """The other POST chokepoint (call_raw) shares the retry path."""
    client = _make_client(monkeypatch, max_retries=3)
    client._client = _ScriptedClient([_status(500), _FakeResponse()])

    result = await client.call_raw([{"role": "user", "content": "hi"}])

    assert result.text == "ok"
    assert client._client.calls == 2
