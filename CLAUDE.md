# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## VPS Access

```bash
ssh vps
```

## What This Is

Autonomous Telegram group chat participant — an AI that acts as a real, low-ego member of chaotic crypto/degen Telegram groups. Hybrid architecture: cheap structured control plane (gate, posture, signals) + rich LLM character reasoning (Grok via OpenRouter/xAI) + optional local fine-tuned voice model for authentic phrasing.

## Commands

```bash
# Infrastructure (Postgres+pgvector, Redis)
docker compose up -d postgres redis
docker compose run --rm migrate          # run alembic migrations

# Full stack (all services)
docker compose up -d

# Individual services (local dev, outside Docker)
python -m ingestion.main                 # Telethon event capture
python -m pipeline.queue_consumer        # Redis → Postgres workers
python -m uvicorn api.app:app --port 8000
python -m conversation_engine            # main decision loop

# Test UI (primary tuning tool)
bash test-ui/run.sh                      # http://localhost:7777
# Needs python3.11+; override: PYTHON=/path/to/python bash test-ui/run.sh

# Tests
pytest                                   # all tests
pytest tests/unit                        # unit only
pytest tests/unit/test_foo.py::test_bar  # single test
# asyncio_mode = "auto" in pyproject.toml — no @pytest.mark.asyncio needed

# Migrations
python -m alembic revision --autogenerate -m "description"
python -m alembic upgrade head
```

## Architecture

**Data flow**: Telegram → Telethon ingestion → Redis Stream → pipeline workers → Postgres → conversation engine polling loop → LLM decisions → Telethon send → delayed feedback observation

### Key layers and files

| Layer | Entry point | Core files |
|-------|------------|------------|
| **Ingestion** | `ingestion/main.py` | `telethon_client.py`, `event_handlers.py` — captures messages/edits/deletes via MTProto, produces to Redis Stream |
| **Pipeline** | `pipeline/queue_consumer.py` | `workers.py` — idempotent upserts (Message, Sender, Chat, ChatMember), edit history, deletion tracking |
| **Storage** | `storage/` | `postgres_models.py` (Message, BotMemory, AiDecision, ResponseFeedback, BotVectorMemory, BotPersonaCore, BotSelfReflection, UserRelationshipProfile, StanceTracker), `repositories.py`, `database.py` |
| **Conversation Engine** | `conversation_engine/__main__.py` | The brain. See below. |
| **API** | `api/app.py` | FastAPI REST endpoints |
| **Test UI** | `test-ui/server.py` | `runner.py` drives real engine logic with `FakeConversationMemoryManager`; primary tool for tuning gate/context/prompts |

### Conversation Engine decision cycle (`scheduler.py:_run_cycle`)

1. Circuit breaker check
2. New message threshold (3 group / 1 DM) + active bot thread detection
3. Enrichment (`enrichment.py`) — VADER sentiment with group-specific overrides, topic extraction
4. Brief — tension, active threads, topic drift
5. **Engagement gate** (`engagement_gate.py`) — cheap pre-LLM filter: velocity, fatigue, relationship, thread repeat, feedback signal → weighted score vs `min_gate_score_to_send`
6. Context building (`context_builder.py`) — target message + nearby + persona + recent bot activity ("I said...") + posture + slim activity signals
7. **Perception** (`ai_client.py`) — compresses ~200 high-level msgs into relevant context for decision
8. **Decision** (`prompts.py:SMART_PARTICIPANT_SYSTEM`) — character at temp 0.8, three qualitative questions, outputs JSON with should_respond/plan/posture/stances
9. Optional **hybrid style rewriter** (`style_rewriter.py`) — local LoRA fine-tune phrases the plan in authentic voice
10. Validate + send + record BotMemory (with round-tripped posture) + schedule feedback (45min)

### Critical state that persists across cycles

- **Posture**: `current_posture` in BotMemory, round-tripped via `_infer_social_posture` — the bot's durable energy/mood state
- **BotMemory**: "what I said and why" — reconstructed as "MY RECENT ACTIVITY AS ME" in context
- **BotPersonaCore**: identity/beliefs/style, mutated by self-reflections
- **ResponseFeedback**: delayed scoring of how responses landed (replies, reactions, sentiment)

## Config

- `config.toml` — all tunables: persona, AI models, gate weights, scheduler intervals, feedback timing
- `.env` — secrets and infrastructure URLs (Telegram creds, API keys, DB/Redis URLs)
- Gate weights in `[engagement_gate]` section are the primary tuning lever
- `XAI_API_KEY` / `XAI_BASE_URL` — AI backend (xAI Grok, OpenRouter, or local)
- `LOCAL_STYLE_REWRITE_ENABLED` — toggles hybrid voice mode (needs local model served via HTTP)

## Tech Stack

Python 3.11+, Telethon (MTProto), Redis Streams, PostgreSQL + pgvector, SQLAlchemy async, FastAPI, VADER sentiment, sentence-transformers (all-MiniLM-L6-v2 for 384-dim embeddings), httpx for LLM calls.

## Conventions

- All DB access goes through `ConversationMemoryManager` (conversation_engine/memory_manager.py) — it's the shared bus
- Append-only tables with snapshots (brief, stances, raw context in AiDecision) for explainability
- `prompt_version` tracked in decisions and bot memory for drift analysis
- The test UI (`test-ui/`) exercises the real engine logic — use it to verify gate/context/prompt changes before deploying
- VPS deployment: `rsync` + `docker compose` (see `deploy/vps/README.md` and memory notes)
