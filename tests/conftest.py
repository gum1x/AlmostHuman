"""Global test setup: keep the suite fully offline.

Several unit tests load the persona embedder at import time. Production refuses
the zero-vector ``ZeroEmbedder`` stub whenever ``sentence-transformers`` is
installed (so prod never silently corrupts pgvector ranking). In CI the package
IS installed, so those tests would try to fetch a real model from HuggingFace
and fail with no network. Force the test process to look as if
``sentence-transformers`` is absent and opt into the stub, so embedder loading
stays offline regardless of what's installed. Per-test monkeypatching (e.g. in
``test_persona_engine_embedder.py``) still overrides this locally.
"""

import os

import conversation_engine.persona_engine as persona_engine

os.environ.setdefault("ALLOW_FAKE_EMBEDDER", "true")
persona_engine.SentenceTransformer = None
