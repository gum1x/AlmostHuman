from __future__ import annotations

from conversation_engine.config import load_engine_config
from conversation_engine.timing_classifier import timing_should_skip


def test_shadow_flag_parses_from_env(monkeypatch):
    monkeypatch.setenv("TIMING_CLASSIFIER_SHADOW", "true")
    cfg = load_engine_config()
    assert cfg.timing_classifier_shadow is True


def test_shadow_flag_defaults_false(monkeypatch):
    monkeypatch.delenv("TIMING_CLASSIFIER_SHADOW", raising=False)
    cfg = load_engine_config()
    assert cfg.timing_classifier_shadow is False


def test_enforcing_skips_on_reject():
    assert timing_should_skip(passes=False, enforcing=True) is True

def test_shadow_never_skips():
    assert timing_should_skip(passes=False, enforcing=False) is False

def test_pass_never_skips():
    assert timing_should_skip(passes=True, enforcing=True) is False
    assert timing_should_skip(passes=True, enforcing=False) is False
