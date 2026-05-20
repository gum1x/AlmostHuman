import asyncio
import platform
import os

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
            entries = await self._redis.xreadgroup(
                self._group, self._consumer_name,
                {self._stream_key: "0"},
                count=self._batch_size,
            )

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

            except redis.ConnectionError:
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
            except Exception:
                await log.aexception("event_processing_failed", entry_id=entry_id)

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
