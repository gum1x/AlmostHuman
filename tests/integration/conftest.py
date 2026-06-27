"""Integration-test harness: real Postgres (pgvector) + Redis via testcontainers.

These tests exercise the genuine SQL / pgvector / concurrency behaviour that the
unit suite necessarily fakes. They need a Docker daemon, so the whole suite is
gated by the ``pytest_collection_modifyitems`` hook below: when testcontainers or
Docker is unavailable, every ``integration``-marked test is marked skipped at
collection time, so a plain ``pytest`` run on a machine without Docker stays green
(collected-then-skipped, exit 0) rather than erroring in the container fixtures.
CI (GitHub Actions ubuntu-latest) has Docker, so the ``integration`` job runs them
for real. The hook deliberately avoids importing this conftest by dotted path
(``tests`` is not an installed package, so ``from tests.integration.conftest
import ...`` fails under the bare ``pytest`` console script CI uses).

This conftest must import cleanly even when testcontainers is absent, so every
testcontainers/SQLAlchemy-async import is done lazily inside the fixtures rather
than at module top level (a conftest cannot be "skipped" the way a test module
can, so it must never raise on import).

Design notes
------------
* The session-scoped fixtures are ``async`` and pinned to a session-scoped event
  loop (``loop_scope="session"``, matched by the module-level ``pytestmark`` in
  each test file). asyncpg connections are bound to the loop that created them,
  so a single shared loop keeps the engine usable across every test.
* Alembic migrations are driven through ``command.upgrade(..., "head")``. The
  project's ``alembic/env.py`` reads ``core.config.settings.database_url`` and
  calls ``asyncio.run`` itself, so we (a) point ``settings.database_url`` at the
  container and (b) run the upgrade in a worker thread (``asyncio.to_thread``)
  where a fresh event loop is free to spin up.
* The live code reaches the DB through ``storage.database.async_session_factory``
  (and ``pipeline.workers`` re-imports that name). We rebind both to a factory
  backed by the container engine so the real worker/upsert path is under test.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

PG_IMAGE = "pgvector/pgvector:pg16"
REDIS_IMAGE = "redis:7-alpine"

# Escape hatch: point the Postgres fixtures at an already-running pgvector
# instance (must match pgvector/pgvector:pg16) instead of starting a container.
# Useful where Docker is unavailable but a DB is provisioned. The URL must use
# the asyncpg driver, e.g. postgresql+asyncpg://user:pass@host:5432/db.
_EXTERNAL_DB_URL = os.environ.get("INTEGRATION_DATABASE_URL")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ALEMBIC_DIR = _PROJECT_ROOT / "alembic"


@lru_cache(maxsize=1)
def docker_is_available() -> bool:
    """True iff testcontainers imports and a Docker daemon answers a ping.

    Cached so the (slightly expensive) daemon ping happens once per session.
    Any failure -- missing testcontainers, no daemon, permission error -- is
    treated as "unavailable" so the integration modules skip cleanly rather than
    erroring during collection. An ``INTEGRATION_DATABASE_URL`` override bypasses
    the Docker requirement entirely (the suite uses that DB directly).
    """
    if _EXTERNAL_DB_URL:
        return True
    try:
        from testcontainers.core.docker_client import DockerClient

        DockerClient().client.ping()
        return True
    except Exception:
        return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip every ``integration``-marked test when Docker is unavailable.

    Runs as a conftest hook -- pytest auto-discovers it, so nothing has to import
    this module by dotted path (which fails under the bare ``pytest`` console
    script because ``tests`` is not an installed package). With no Docker daemon
    (and no ``INTEGRATION_DATABASE_URL`` escape hatch) the container fixtures can't
    start, so we mark the whole integration suite skipped here instead of letting
    those fixtures error. The tests are still *collected*, so pytest exits 0, not 5.
    """
    if docker_is_available():
        return
    skip = pytest.mark.skip(reason="Docker daemon / testcontainers unavailable")
    for item in items:
        if item.get_closest_marker("integration") is not None:
            item.add_marker(skip)


def _run_migrations(database_url: str) -> None:
    """Run ``alembic upgrade head`` against ``database_url`` in this thread.

    Invoked via ``asyncio.to_thread`` so ``env.py``'s ``asyncio.run`` gets its
    own event loop. ``env.py`` reads ``core.config.settings.database_url``, so we
    override it for the duration of the upgrade and also set it on the Alembic
    config (``set_main_option`` stores a literal value, sidestepping the
    ``%(DATABASE_URL)s`` interpolation in alembic.ini).
    """
    from alembic.config import Config

    from alembic import command
    from core.config import settings

    original_url = settings.database_url
    settings.database_url = database_url
    try:
        cfg = Config()
        cfg.set_main_option("script_location", str(_ALEMBIC_DIR))
        cfg.set_main_option("sqlalchemy.url", database_url)
        command.upgrade(cfg, "head")
    finally:
        settings.database_url = original_url


@pytest.fixture(scope="session")
def monkeypatch_session() -> Iterator[Any]:
    """Session-scoped MonkeyPatch (the built-in ``monkeypatch`` is function-scoped)."""
    from _pytest.monkeypatch import MonkeyPatch

    mpatch = MonkeyPatch()
    try:
        yield mpatch
    finally:
        mpatch.undo()


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def redis_container() -> AsyncIterator[Any]:
    """A live Redis 7 container; yields the started container for URL/client use."""
    from testcontainers.redis import RedisContainer

    container = RedisContainer(REDIS_IMAGE)
    await asyncio.to_thread(container.start)
    try:
        yield container
    finally:
        await asyncio.to_thread(container.stop)


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def redis_url(redis_container: Any) -> str:
    host = redis_container.get_container_host_ip()
    port = redis_container.get_exposed_port(redis_container.port)
    return f"redis://{host}:{port}/0"


async def _build_engine_and_patch(database_url: str, monkeypatch_session: Any) -> tuple[Any, Any]:
    """Migrate ``database_url``, build an engine/factory, and rebind the live
    code paths (``storage.database`` + the copy ``pipeline.workers`` imported) at
    that factory so code under test that opens its own session hits this DB."""
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    # Migrations run in their own thread so env.py's asyncio.run is happy.
    await asyncio.to_thread(_run_migrations, database_url)

    engine = create_async_engine(database_url, pool_pre_ping=True)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    import pipeline.workers as workers
    import storage.database as database

    monkeypatch_session.setattr(database, "engine", engine, raising=False)
    monkeypatch_session.setattr(database, "async_session_factory", factory, raising=False)
    monkeypatch_session.setattr(workers, "async_session_factory", factory, raising=False)
    return engine, factory


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def db_engine_and_factory(monkeypatch_session: Any) -> AsyncIterator[tuple[Any, Any]]:
    """Spin up Postgres (pgvector), migrate it, and yield ``(engine, factory)``.

    Honours ``INTEGRATION_DATABASE_URL`` (use an existing DB, no container);
    otherwise starts a ``pgvector/pgvector:pg16`` container via testcontainers.
    """
    if _EXTERNAL_DB_URL:
        engine, factory = await _build_engine_and_patch(_EXTERNAL_DB_URL, monkeypatch_session)
        try:
            yield engine, factory
        finally:
            await engine.dispose()
        return

    from testcontainers.postgres import PostgresContainer

    container = PostgresContainer(
        PG_IMAGE,
        username="ci_user",
        password="ci_pass",
        dbname="telegram_ci",
        driver="asyncpg",
    )
    await asyncio.to_thread(container.start)
    try:
        database_url = container.get_connection_url()  # postgresql+asyncpg://...
        engine, factory = await _build_engine_and_patch(database_url, monkeypatch_session)
        try:
            yield engine, factory
        finally:
            await engine.dispose()
    finally:
        await asyncio.to_thread(container.stop)


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def session_factory(db_engine_and_factory: tuple[Any, Any]) -> Any:
    """The container-backed async session factory (what production code uses)."""
    _engine, factory = db_engine_and_factory
    return factory


def _all_table_names() -> list[str]:
    """Every mapped table name, ordered so ``TRUNCATE ... CASCADE`` is harmless.

    Pulled from the metadata so a newly added model never silently escapes the
    between-tests cleanup.
    """
    from storage.postgres_models import Base

    return [table.name for table in Base.metadata.sorted_tables]


@pytest_asyncio.fixture(loop_scope="session")
async def db_session(session_factory: Any) -> AsyncIterator[Any]:
    """A clean :class:`AsyncSession` per test.

    Every mapped table is truncated (RESTART IDENTITY, CASCADE) before the test
    runs, so each test starts from an empty schema regardless of order. The
    session is rolled back and closed afterwards.
    """
    from sqlalchemy import text

    tables = _all_table_names()
    async with session_factory() as cleanup:
        await cleanup.execute(text(f"TRUNCATE {', '.join(tables)} RESTART IDENTITY CASCADE"))
        await cleanup.commit()

    session = session_factory()
    try:
        yield session
    finally:
        await session.rollback()
        await session.close()
