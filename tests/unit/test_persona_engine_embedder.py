from __future__ import annotations

import pytest

import conversation_engine.persona_engine as persona_engine
from conversation_engine.persona_engine import ZeroEmbedder, embed_text, load_embedder


@pytest.fixture(autouse=True)
def reset_embedder(monkeypatch):
    monkeypatch.setattr(persona_engine, "_embedder", None)
    monkeypatch.delenv("ALLOW_FAKE_EMBEDDER", raising=False)


def test_missing_sentence_transformers_raises(monkeypatch):
    monkeypatch.setattr(persona_engine, "SentenceTransformer", None)
    with pytest.raises(RuntimeError, match="sentence-transformers"):
        load_embedder("all-MiniLM-L6-v2")


def test_embed_text_without_loaded_embedder_raises():
    with pytest.raises(RuntimeError, match="not loaded"):
        embed_text("hello")


def test_explicit_fake_embedder_opt_in(monkeypatch):
    # Fake only kicks in when sentence-transformers is missing AND the flag is set.
    monkeypatch.setattr(persona_engine, "SentenceTransformer", None)
    monkeypatch.setenv("ALLOW_FAKE_EMBEDDER", "true")
    embedder = load_embedder("all-MiniLM-L6-v2")
    assert isinstance(embedder, ZeroEmbedder)
    assert embed_text("hello") == [0.0] * 384


def test_fake_flag_does_not_override_real_embedder(monkeypatch):
    class FakeSentenceTransformer:
        def __init__(self, model_name):
            self.model_name = model_name

        def encode(self, text):
            return [1.0] * 384

    monkeypatch.setattr(persona_engine, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setenv("ALLOW_FAKE_EMBEDDER", "true")
    embedder = load_embedder("all-MiniLM-L6-v2")
    assert isinstance(embedder, FakeSentenceTransformer)
    assert embedder.model_name == "all-MiniLM-L6-v2"


def test_injected_embedder_is_used(monkeypatch):
    class FakeEmbedder:
        def encode(self, text):
            return [1.0, 2.0]

    monkeypatch.setattr(persona_engine, "_embedder", FakeEmbedder())
    assert embed_text("hello") == [1.0, 2.0]
