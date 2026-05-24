import asyncio
import signal

from core.config import settings
from core.logging import get_logger, setup_logging
from ingestion.event_handlers import register_handlers
from ingestion.telethon_client import TelethonClientManager
from pipeline.queue_producer import QueueProducer

log = get_logger(__name__)


async def main():
    setup_logging(settings.log_level, settings.log_json)
    await log.ainfo(
        "ingestion_starting",
        chat_ids=settings.chat_ids,
        monitor_private_dms=settings.monitor_private_dms,
    )

    shutdown_event = asyncio.Event()
    manager = TelethonClientManager()
    producer = QueueProducer()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown_event.set)

    try:
        await producer.connect()
        client = await manager.connect()
        register_handlers(client, producer, settings.chat_ids, settings.monitor_private_dms)

        await log.ainfo(
            "ingestion_running",
            monitored_chats=len(settings.chat_ids),
            monitor_private_dms=settings.monitor_private_dms,
        )

        await shutdown_event.wait()
        await log.ainfo("shutdown_signal_received")

    except Exception:
        await log.aexception("ingestion_fatal_error")
    finally:
        await manager.disconnect()
        await producer.close()
        await log.ainfo("ingestion_stopped")


if __name__ == "__main__":
    asyncio.run(main())
