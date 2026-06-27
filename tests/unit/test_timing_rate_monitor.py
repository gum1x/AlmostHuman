from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import timing_rate_monitor as mon  # noqa: E402


def _row(p, would_pass, is_direct, sent):
    return {"timing_p": p, "timing_would_pass": would_pass,
            "timing_is_direct": is_direct, "should_respond": sent}


def test_summarize_splits_addressed_and_unprompted():
    rows = [_row(0.9, True, False, True), _row(0.1, False, False, False),
            _row(0.0, False, True, True), _row(0.0, False, True, True)]
    s = mon.summarize(rows)
    assert s["n"] == 4
    assert s["addressed_frac"] == 0.5            # 2 of 4 addressed
    assert s["unprompted_pass_rate"] == 0.5      # 1 of 2 unprompted passed
    assert s["send_rate"] == 0.75                # 3 of 4 sent


def test_check_band():
    ok, _ = mon.check_band({"unprompted_pass_rate": 0.08}, 0.05, 0.15)
    assert ok is True
    bad, msg = mon.check_band({"unprompted_pass_rate": 0.40}, 0.05, 0.15)
    assert bad is False and "0.40" in msg


class _FakeSession:
    """Async-context-manager stand-in for an AsyncSession from async_session_factory()."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeMemory:
    """Records the positional arg it was constructed with so we can assert the
    entrypoint passes a *session*, not an EngineConfig (the bug)."""

    constructed_with = None

    def __init__(self, arg):
        type(self).constructed_with = arg
        self._session = arg

    async def recent_timing_decisions(self, hours):
        self.hours = hours
        return [_row(0.9, True, False, True), _row(0.1, False, False, False)]


def test_main_entrypoint_wires_session_into_memory_manager(monkeypatch, capsys):
    """The entrypoint must open async_session_factory() and hand the AsyncSession to
    ConversationMemoryManager. Before the fix it passed an EngineConfig, so the real
    constructor (def __init__(self, session)) ran but the manager held a non-session;
    here we prove main() runs end-to-end and the manager receives the session object."""
    session = _FakeSession()
    monkeypatch.setattr("storage.database.async_session_factory", lambda: session)
    monkeypatch.setattr(
        "conversation_engine.memory_manager.ConversationMemoryManager", _FakeMemory
    )

    # 2 rows, both unprompted, 1 passes -> 50% -> out of [5%, 15%] band -> exit 1, no crash.
    rc = mon.main(["--hours", "12", "--lo", "0.05", "--hi", "0.15"])

    assert rc == 1  # ran to completion (band check), no TypeError
    assert _FakeMemory.constructed_with is session  # session wired in, not a config
    out = capsys.readouterr().out
    assert "n=2" in out and "unprompted_pass=50.00%" in out


def test_load_rows_passes_session_not_config(monkeypatch):
    """Tighter unit on the wiring: _load_rows opens a session and constructs the
    manager with it, then awaits the query against that same session."""
    import asyncio

    session = _FakeSession()
    seen = {}

    class _RecordingMemory(_FakeMemory):
        def __init__(self, arg):
            super().__init__(arg)
            seen["arg"] = arg

    monkeypatch.setattr("storage.database.async_session_factory", lambda: session)
    monkeypatch.setattr(
        "conversation_engine.memory_manager.ConversationMemoryManager", _RecordingMemory
    )

    rows = asyncio.run(mon._load_rows(24.0))

    assert seen["arg"] is session
    assert len(rows) == 2  # the canned rows came back through the manager
