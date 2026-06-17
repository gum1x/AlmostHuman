# GroupGhost

Self-hosted autonomous group-chat participant. It ingests a live Telegram group over MTProto, decides *whether* to speak with a machine-learning classifier trained on which messages real members actually reply to, before spending a single token, reasons about *what* to say with an LLM, rewrites the result in a fine-tuned local voice, sends it back, and grades how each message landed 45 minutes later so the next decision gets smarter. It tracks mood, memory, and relationships across cycles, so it reads as a real low-ego member of the room instead of a bot that replies to everything.

No reply-to-everything loop. No obvious AI voice. No third-party chat API in the hot path. Your machine, your models, the group's own message stream.

## Why

Most chat bots answer every message and sound like an assistant, which is exactly how a room clocks them as fake. GroupGhost splits the problem the way a person actually works: a cheap control plane decides if anything is even worth saying, and only then does an expensive model decide what. A delayed feedback loop scores every message it sends on what really happened (replies, reactions, sentiment shift) and feeds that back into the gate, so the agent adapts to *this* room rather than a global heuristic. A local fine-tuned voice model rewrites the plan into authentic phrasing. All of it runs on infrastructure you host, pointed at any model you choose.

## By the numbers

- **~162k messages** of real group history (≈3 weeks) train and calibrate the when-to-respond model.
- **11-feature logistic-regression** classifier, fit on a **time-ordered 60/20/20 split** (no future leakage) with **isotonic-calibrated** probabilities.
- **One tunable threshold** sets how chatty it is. At its shipped **~6% reply cadence**, the large majority of the unprompted firehose is dropped before a single token is spent.
- Perception compresses up to **200 messages** per decision into a **6k-token** context budget.
- Every message it sends is **scored 45 minutes later**, and that grade feeds back into the gate.
- Local ranking is free: **384-dim** MiniLM embeddings + VADER sentiment, zero API calls.

## 60-second quickstart

```bash
cp .env.example .env
#   TG_API_ID / TG_API_HASH / TG_PHONE   Telegram MTProto creds
#   XAI_API_KEY / XAI_BASE_URL           any OpenAI-compatible endpoint

docker compose up -d postgres redis      # Postgres + pgvector, Redis
docker compose run --rm migrate          # alembic schema
docker compose up -d                      # ingestion, worker, api, conversation-engine, autoheal
```

No API key handy? Set `ALLOW_FAKE_AI=true` and the engine runs end to end against a stub so you can watch the plumbing work.

## Pipeline (real-time, two-tier)

```
Telegram  ->  Telethon ingest  ->  Redis Stream  ->  Pipeline workers  ->  Postgres + pgvector
                                                                                 |
                                                             Conversation Engine (polling loop)
                                                                                 |
                                               gate -> context -> perceive -> decide -> (style rewrite)
                                                                                 |
                                                          Telethon send  ->  feedback scored 45 min later
                                                                                 ^                   |
                                                                                 +-------------------+
```

| Tier | Cost | What it does |
|------|------|--------------|
| Control plane | Pure Python, no tokens | A trained timing classifier (logreg) + the engagement gate, plus posture/mood, fatigue, relationship and thread tracking. Decides *whether* to act: messages addressed to the bot always engage, the rest of the firehose is filtered to a calibrated rate. |
| LLM reasoning | One model call per survivor | Perception compresses the room, then the character decides *what* to say at temp 0.8. |
| Local voice | Local GPU/CPU, no tokens | Optional local fine-tune rewrites the plan into authentic, in-room phrasing. |

## Commands

```bash
docker compose up -d postgres redis                 # infra: Postgres + pgvector, Redis
docker compose run --rm migrate                     # run alembic migrations
docker compose up -d                                # full stack (all services)

python -m ingestion.main                            # Telethon event capture -> Redis Stream
python -m pipeline.queue_consumer                   # Redis -> Postgres workers
python -m uvicorn api.app:app --port 8000           # FastAPI REST endpoints
python -m conversation_engine                       # the decision loop (the brain)

bash test-ui/run.sh                                 # tuning dashboard -> http://localhost:7777
PYTHON=/path/to/python bash test-ui/run.sh          # override interpreter (needs 3.11+)

python -m alembic revision --autogenerate -m "msg"  # new migration from model changes
python -m alembic upgrade head                       # apply migrations

pytest                                              # full suite (offline)
pytest tests/unit                                   # unit only
pytest tests/unit/test_foo.py::test_bar             # single test
```

## The decision cycle and anti-bot filtering

Every poll runs a gauntlet (`conversation_engine/scheduler.py:_run_cycle`). Most messages die early and cheap, which is the design. Only survivors reach the part that costs money.

1. **Circuit breaker** stops everything if the system is misbehaving.
2. **Threshold** skips unless there are 3+ new group messages (1 in DMs) or an active thread it is already in.
3. **Enrichment** runs VADER sentiment (with per-group overrides) and topic extraction.
4. **Brief** reads the room: tension, active threads, topic drift.
5. **When-to-respond filter** is two cheap layers before any token is spent. A trained logistic-regression **timing classifier** (`timing_classifier.py`, learned from which messages real regulars actually reply to) scores the incoming message: messages that @mention or reply to the bot always pass, the rest of the firehose must clear a calibrated threshold (the dial that sets how chatty it is). Survivors then hit the **engagement gate**: a weighted score over velocity, fatigue, relationship, repeat, and past feedback vs `min_gate_score_to_send`.
6. **Context build** assembles the target plus nearby messages, persona, recent self-activity, posture, and signals.
7. **Perception** compresses ~200 high-level messages down to what matters for this one decision.
8. **Decision** has the character answer three honest questions and emit JSON (`should_respond`, plan, posture, stances).
9. **Style rewrite** (optional) re-says the plan in the right voice via the local model.
10. **Send and record** validates, sends, logs what it said and why, and schedules feedback 45 minutes out.

The **feedback loop** is what turns a responder into something that adapts. A line that flops makes the agent slightly less likely to jump in next time; a line that lands buys it confidence. Persistent state across cycles (`posture` mood, `BotMemory` self-history, an evolving `BotPersonaCore`, and `ResponseFeedback` grades) is what makes it read as one continuous someone instead of a fresh prompt each time.

## Voice model

A large model is smart but writes like a large model. So the labor is split: the big model decides *what* to say, and a small local fine-tune re-says it in the target voice (`LOCAL_STYLE_REWRITE_ENABLED`, served over HTTP; see `conversation_engine/style_rewriter.py`). The rewrite runs on your own hardware, so authentic phrasing costs no tokens and never leaves your machine. Turn the flag off and the engine sends the LLM's plan verbatim.

The voice model is trained and validated offline against real message history before it ships; only the inference-time rewrite lives in this repo.

## How it works

```
ingestion/main.py                 Telethon MTProto capture -> Redis Stream (telethon_client, event_handlers)
pipeline/queue_consumer.py        Redis Stream -> idempotent Postgres upserts (workers.py)
storage/                          async SQLAlchemy models + pgvector 384-dim MiniLM (postgres_models, repositories, database)
conversation_engine/scheduler.py  the decision cycle (_run_cycle), per-cycle orchestration
conversation_engine/engagement_gate.py   cheap pre-LLM filter: the bouncer
conversation_engine/timing_classifier.py  learned logreg: would a regular bother replying here?
conversation_engine/context_builder.py   assembles target + nearby + persona + posture + signals
conversation_engine/ai_client.py  perception + decision LLM calls (OpenAI-compatible)
conversation_engine/prompts.py    SMART_PARTICIPANT_SYSTEM: the character, three questions, JSON out
conversation_engine/style_rewriter.py     local fine-tuned voice rewrite over HTTP
conversation_engine/feedback_loop.py      delayed 45-min scoring -> ResponseFeedback
conversation_engine/persona_engine.py     self-reflection -> BotPersonaCore mutation
conversation_engine/memory_manager.py     the shared DB bus (all access goes through here)
api/app.py                        FastAPI REST endpoints
test-ui/server.py                 drives the real engine via FakeConversationMemoryManager (runner.py)
config.toml / .env                all tunables / all secrets
```

All DB access goes through `ConversationMemoryManager`. Tables are append-only with snapshots (brief, stances, raw context in `AiDecision`) and a tracked `prompt_version`, so every decision stays explainable after the fact.

## LLM brain

Local ranking is free: VADER sentiment and `sentence-transformers` embeddings (all-MiniLM-L6-v2, 384-dim) handle enrichment and context selection without a model call. The expensive perception and decision steps run on any OpenAI-compatible endpoint (OpenRouter, xAI, or a local Ollama / llama.cpp server). Model names live in `[ai]` in `config.toml` (currently `moonshotai/kimi-k2-0905` for perception and decision), so swapping models never touches `.env` or a line of code. Leave `XAI_API_KEY` empty for a local server and the client sends a dummy `Bearer sk-local`.

## Configuration

Everything you tune lives in `config.toml`; every secret lives in `.env`. If you touch one thing, make it `[engagement_gate]`, the dial that decides how chatty it is.

| Section | Controls |
|---------|----------|
| `[persona]` | Character identity, beliefs, speaking style |
| `[ai]` | Model selection, token budgets |
| `[prompt]` | Prompt assembly and versioning |
| `[scheduler]` | Message thresholds, backoff intervals |
| `[circuit_breaker]` | Failure cutoffs |
| `[engagement_gate]` | Gate weights and thresholds, the primary tuning lever |
| `[feedback_loop]` | Delayed feedback timing and scoring |
| `[persona_engine]` | Self-reflection and persona mutation |

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

Conventions: Python 3.11+, async throughout, SQLAlchemy async, pydantic schemas in `core/`. `asyncio_mode = "auto"` in `pyproject.toml`, so no `@pytest.mark.asyncio` decorators. The suite runs fully offline, so CI never downloads embedding models. The test UI exercises the real engine logic, so use it to verify gate, context, and prompt changes before deploying.

## Further reading

- [`docs/telegram_group_speech_style_guide.md`](docs/telegram_group_speech_style_guide.md) the voice the model is trained toward
- `CLAUDE.md` repo map and conventions

## License

No license file is included. This is a private research project, not licensed for redistribution.
