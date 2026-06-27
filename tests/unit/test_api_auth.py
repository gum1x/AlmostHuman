"""API auth + binding + metrics (P1 security).

The FastAPI app serves ingested chat content, including private-DM PII. These tests
pin the security contract:

* data endpoints require a valid bearer token (401 without, 200 with),
* an unset ``API_AUTH_TOKEN`` fails closed (503) rather than serving PII,
* ``GET /health`` stays unauthenticated (the compose healthcheck depends on it),
* ``GET /metrics`` exposes recorded engine telemetry (no token, no PII).

Everything is offline/deterministic: the DB/repo dependency is stubbed via
FastAPI ``dependency_overrides`` and the health probe's engine/redis globals are
monkeypatched, so no real Postgres/Redis is touched.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Pydantic-settings reads these at import time; pin before importing the app.
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://u:p@localhost:5432/x")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")

import api.app as app_module  # noqa: E402
from api.app import app  # noqa: E402
from api.dependencies import get_message_repo  # noqa: E402
from conversation_engine import observability  # noqa: E402

TOKEN = "s3cret-test-token"


class _FakeRepo:
    """Stand-in for MessageRepository — no DB, deterministic empty results."""

    async def get_messages(self, chat_id, limit=50, offset=0, include_deleted=False):
        return [], 0

    async def get_thread(self, chat_id, message_id):
        return []


class _FakeConn:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, *a, **k):
        return None


class _FakeEngine:
    """Health probe calls ``engine.connect()`` directly; make it succeed offline."""

    def connect(self):
        return _FakeConn()


@pytest.fixture(autouse=True)
def _offline_health(monkeypatch):
    # Keep /health from touching a real DB/Redis. _redis_pool is None unless the
    # lifespan runs (we don't enter the TestClient context manager), so the redis
    # branch is skipped; stub the engine so the postgres branch succeeds offline.
    monkeypatch.setattr(app_module, "engine", _FakeEngine())
    monkeypatch.setattr(app_module, "_redis_pool", None)


@pytest.fixture
def client():
    app.dependency_overrides[get_message_repo] = lambda: _FakeRepo()
    try:
        # No context manager => lifespan (redis connect / engine dispose) does not run.
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def _set_token(monkeypatch, value: str | None):
    from core.config import settings

    monkeypatch.setattr(settings, "api_auth_token", value or "")


# ---- auth on data endpoints ----


def test_messages_requires_token(client, monkeypatch):
    _set_token(monkeypatch, TOKEN)
    resp = client.get("/messages/123")
    assert resp.status_code == 401


def test_messages_rejects_wrong_token(client, monkeypatch):
    _set_token(monkeypatch, TOKEN)
    resp = client.get("/messages/123", headers={"Authorization": "Bearer nope"})
    assert resp.status_code == 401


def test_messages_with_token_ok(client, monkeypatch):
    _set_token(monkeypatch, TOKEN)
    resp = client.get("/messages/123", headers={"Authorization": f"Bearer {TOKEN}"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["items"] == []
    assert body["total"] == 0


def test_thread_requires_token(client, monkeypatch):
    _set_token(monkeypatch, TOKEN)
    assert client.get("/messages/123/thread/9").status_code == 401
    ok = client.get("/messages/123/thread/9", headers={"Authorization": f"Bearer {TOKEN}"})
    assert ok.status_code == 200
    assert ok.json() == []


def test_fails_closed_when_token_unset(client, monkeypatch):
    # No API_AUTH_TOKEN configured: must reject (503), never serve PII — even if a
    # caller presents some bearer token.
    _set_token(monkeypatch, "")
    assert client.get("/messages/123").status_code == 503
    assert (
        client.get("/messages/123", headers={"Authorization": "Bearer anything"}).status_code
        == 503
    )


# ---- unauthenticated endpoints ----


def test_health_no_token(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "status" in resp.json()


def test_metrics_exposes_recorded_metrics(client):
    # Record one gauge + one counter, then assert the scrape renders both.
    observability.metrics.gauge("conversation_engine.gate.score", 0.73)
    observability.metrics.increment("conversation_engine.feedback.outcome", outcome="reply")

    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.text
    # Dots sanitized to underscores; values present; TYPE hints emitted.
    assert "conversation_engine_gate_score 0.73" in body
    assert 'conversation_engine_feedback_outcome{outcome=reply} 1' in body
    assert "# TYPE conversation_engine_gate_score gauge" in body
    assert "# TYPE conversation_engine_feedback_outcome counter" in body


def test_metrics_no_token_required(client):
    # /metrics carries no PII and must be scrapeable without a token.
    assert client.get("/metrics").status_code == 200
