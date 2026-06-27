import asyncio
import os
import platform

import orjson
import redis.asyncio as redis

from core.config import settings
from core.logging import get_logger, setup_logging
from core.schemas import RawTelegramEvent
from pipeline.workers import MessageWorker

log = get_logger(__name__)


class QueueConsumer:
    def __init__(self):
        self._redis: redis.Redis | None = None
        self._stream_key = settings.redis_stream_key
        self._group = settings.redis_consumer_group
        self._consumer_name = f"{platform.node()}-{os.getpid()}"
        self._batch_size = settings.redis_batch_size
        self._block_ms = settings.redis_block_ms
        self._autoclaim_interval = settings.redis_autoclaim_interval_s
        self._autoclaim_min_idle = settings.redis_autoclaim_min_idle_ms
        self._dlq_stream_key = settings.redis_dlq_stream_key
        self._dlq_max_delivery = settings.redis_dlq_max_delivery
        self._shutdown = asyncio.Event()
        self._worker: MessageWorker | None = None

    async def connect(self):
        self._redis = redis.from_url(settings.redis_url, decode_responses=False)

        try:
            await self._redis.xgroup_create(
                self._stream_key, self._group, id="0", mkstream=True
            )
            await log.ainfo("consumer_group_created", group=self._group)
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

        self._worker = MessageWorker()
        await self._worker.connect()
        await log.ainfo("consumer_connected", name=self._consumer_name, group=self._group)

    async def run(self):
        await self.connect()

        pending_task = asyncio.create_task(self._process_pending())
        autoclaim_task = asyncio.create_task(self._autoclaim_loop())

        try:
            await pending_task
            await self._read_loop()
        finally:
            autoclaim_task.cancel()
            try:
                await autoclaim_task
            except asyncio.CancelledError:
                pass
            await self.close()

    async def _process_pending(self):
        while not self._shutdown.is_set():
            try:
                entries = await self._redis.xreadgroup(
                    self._group, self._consumer_name,
                    {self._stream_key: "0"},
                    count=self._batch_size,
                )
            except (redis.ConnectionError, redis.TimeoutError):
                await log.aerror("redis_connection_lost")
                if not self._shutdown.is_set():
                    await asyncio.sleep(2)
                continue

            has_entries = False
            for stream, messages in entries:
                if not messages:
                    continue
                has_entries = True
                await self._process_batch(messages)

            if not has_entries:
                return

    async def _read_loop(self):
        while not self._shutdown.is_set():
            try:
                entries = await self._redis.xreadgroup(
                    self._group, self._consumer_name,
                    {self._stream_key: ">"},
                    count=self._batch_size,
                    block=self._block_ms,
                )

                for stream, messages in entries:
                    if messages:
                        await self._process_batch(messages)

            except (redis.ConnectionError, redis.TimeoutError):
                await log.aerror("redis_connection_lost")
                if not self._shutdown.is_set():
                    await asyncio.sleep(2)

    async def _process_batch(self, messages: list):
        for entry_id, data in messages:
            try:
                payload = orjson.loads(data[b"payload"])
                event = RawTelegramEvent(**payload)
                await self._worker.process(event)
                await self._redis.xack(self._stream_key, self._group, entry_id)
            except Exception as exc:
                await log.aexception("event_processing_failed", entry_id=entry_id)
                await self._maybe_dead_letter(entry_id, data, exc)

    async def _maybe_dead_letter(self, entry_id, data: dict, exc: Exception):
        """Dead-letter an entry once it has failed too many times.

        The entry is still pending (we never acked it), so its authoritative
        delivery count lives in Redis. When that count reaches the configured
        cap we move the fields to the dead-letter stream and ack the original,
        so a poison event stops re-delivering forever via the autoclaim loop.
        Below the cap the entry is left pending for the next redelivery.
        """
        attempts = await self._delivery_count(entry_id)
        if attempts < self._dlq_max_delivery:
            return

        dlq_fields = dict(data)
        dlq_fields[b"original_id"] = (
            entry_id if isinstance(entry_id, bytes) else str(entry_id).encode()
        )
        dlq_fields[b"attempts"] = str(attempts).encode()
        dlq_fields[b"error"] = repr(exc).encode()

        await self._redis.xadd(self._dlq_stream_key, dlq_fields)
        await self._redis.xack(self._stream_key, self._group, entry_id)
        await log.aerror(
            "event_dead_lettered",
            entry_id=entry_id,
            attempts=attempts,
            dlq_stream=self._dlq_stream_key,
            error=repr(exc),
        )

    async def _delivery_count(self, entry_id) -> int:
        """Return how many times Redis has delivered this pending entry."""
        try:
            pending = await self._redis.xpending_range(
                self._stream_key, self._group,
                min=entry_id, max=entry_id, count=1,
            )
        except Exception:
            await log.aexception("delivery_count_lookup_failed", entry_id=entry_id)
            return 0
        if not pending:
            return 0
        return int(pending[0]["times_delivered"])

    async def _autoclaim_loop(self):
        while not self._shutdown.is_set():
            try:
                await asyncio.sleep(self._autoclaim_interval)
                result = await self._redis.xautoclaim(
                    self._stream_key, self._group, self._consumer_name,
                    min_idle_time=self._autoclaim_min_idle,
                    start_id="0-0",
                    count=self._batch_size,
                )
                claimed = result[1] if len(result) > 1 else []
                if claimed:
                    await log.ainfo("autoclaimed_entries", count=len(claimed))
                    await self._process_batch(claimed)
            except asyncio.CancelledError:
                raise
            except Exception:
                await log.aexception("autoclaim_error")

    def shutdown(self):
        self._shutdown.set()

    async def close(self):
        if self._worker:
            await self._worker.close()
        if self._redis:
            await self._redis.aclose()


async def main():
    setup_logging(settings.log_level, settings.log_json)
    consumer = QueueConsumer()

    loop = asyncio.get_running_loop()
    for sig_name in ("SIGINT", "SIGTERM"):
        import signal
        loop.add_signal_handler(
            getattr(signal, sig_name),
            consumer.shutdown,
        )

    await consumer.run()


if __name__ == "__main__":
    asyncio.run(main())
