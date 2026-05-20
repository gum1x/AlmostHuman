import orjson
import redis.asyncio as redis

from core.config import settings
from core.logging import get_logger
from core.schemas import RawTelegramEvent

log = get_logger(__name__)


class QueueProducer:
    def __init__(self, redis_client: redis.Redis | None = None):
        self._redis = redis_client
        self._stream_key = settings.redis_stream_key
        self._maxlen = settings.redis_stream_maxlen

    async def connect(self):
        if self._redis is None:
            self._redis = redis.from_url(settings.redis_url, decode_responses=False)
        await log.ainfo("queue_producer_connected", stream=self._stream_key)

    async def produce(self, event: RawTelegramEvent):
        payload = orjson.dumps(event.model_dump(mode="json"))
        await self._redis.xadd(
            self._stream_key,
            {"payload": payload},
            maxlen=self._maxlen,
            approximate=True,
        )
        await log.adebug(
            "event_produced",
            event_type=event.event_type,
            chat_id=event.chat_id,
            message_id=event.message_id,
        )

    async def close(self):
        if self._redis:
            await self._redis.aclose()
