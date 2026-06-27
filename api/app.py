from contextlib import asynccontextmanager

import redis.asyncio as redis
from fastapi import Depends, FastAPI, Query, Security
from fastapi.responses import PlainTextResponse
from sqlalchemy import text

from api.dependencies import get_message_repo, require_auth
from conversation_engine.observability import render_prometheus
from core.config import settings
from core.logging import get_logger, setup_logging
from core.schemas import HealthResponse, MessageResponse, PaginatedResponse
from storage.database import dispose_engine, engine
from storage.repositories import MessageRepository

log = get_logger(__name__)

_redis_pool: redis.Redis | None = None


@asynccontextmanager
async def lifespan(application: FastAPI):
    global _redis_pool
    setup_logging(settings.log_level, settings.log_json)
    _redis_pool = redis.from_url(settings.redis_url, decode_responses=True)
    yield
    if _redis_pool:
        await _redis_pool.aclose()
    await dispose_engine()


app = FastAPI(title="Telegram CI API", version="0.1.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health():
    pg_status = "ok"
    redis_status = "ok"
    stream_lag = None

    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as exc:
        await log.awarning("health_postgres_check_failed", error=str(exc))
        pg_status = "error"

    try:
        if _redis_pool:
            await _redis_pool.ping()
            try:
                groups = await _redis_pool.xinfo_groups(settings.redis_stream_key)
                if groups:
                    stream_lag = groups[0].get("lag", 0)
            except Exception as exc:
                await log.awarning("health_stream_lag_check_failed", error=str(exc))
    except Exception as exc:
        await log.awarning("health_redis_check_failed", error=str(exc))
        redis_status = "error"

    status = "healthy" if pg_status == "ok" and redis_status == "ok" else "degraded"

    return HealthResponse(
        status=status,
        postgres=pg_status,
        redis=redis_status,
        stream_lag=stream_lag,
    )


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """Prometheus-format engine telemetry (counters/gauges from observability).

    Left unauthenticated: it exposes only aggregate engine metrics, never chat
    content or PII. The compose/Prometheus scraper can reach it without a token.
    """
    return render_prometheus()


@app.get(
    "/messages/{chat_id}",
    response_model=PaginatedResponse,
    dependencies=[Security(require_auth)],
)
async def get_messages(
    chat_id: int,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    include_deleted: bool = Query(default=False),
    repo: MessageRepository = Depends(get_message_repo),
):
    messages, total = await repo.get_messages(
        chat_id=chat_id,
        limit=limit,
        offset=offset,
        include_deleted=include_deleted,
    )

    items = [
        MessageResponse(
            id=m.id,
            message_id=m.message_id,
            chat_id=m.chat_id,
            sender_id=m.sender_id,
            timestamp=m.timestamp,
            message_type=m.message_type,
            text_raw=m.text_raw,
            text_cleaned=m.text_cleaned,
            reply_to_message_id=m.reply_to_message_id,
            is_deleted=m.is_deleted,
            edit_history=m.edit_history or [],
            mention_list=m.mention_list or [],
            created_at=m.created_at,
            updated_at=m.updated_at,
        )
        for m in messages
    ]

    return PaginatedResponse(items=items, total=total, limit=limit, offset=offset)


@app.get(
    "/messages/{chat_id}/thread/{message_id}",
    response_model=list[MessageResponse],
    dependencies=[Security(require_auth)],
)
async def get_thread(
    chat_id: int,
    message_id: int,
    repo: MessageRepository = Depends(get_message_repo),
):
    messages = await repo.get_thread(chat_id, message_id)

    return [
        MessageResponse(
            id=m.id,
            message_id=m.message_id,
            chat_id=m.chat_id,
            sender_id=m.sender_id,
            timestamp=m.timestamp,
            message_type=m.message_type,
            text_raw=m.text_raw,
            text_cleaned=m.text_cleaned,
            reply_to_message_id=m.reply_to_message_id,
            is_deleted=m.is_deleted,
            edit_history=m.edit_history or [],
            mention_list=m.mention_list or [],
            created_at=m.created_at,
            updated_at=m.updated_at,
        )
        for m in messages
    ]
