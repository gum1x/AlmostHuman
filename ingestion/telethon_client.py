import asyncio

from telethon import TelegramClient
from telethon.errors import FloodWaitError, AuthKeyUnregisteredError

from core.config import settings
from core.logging import get_logger

log = get_logger(__name__)

MAX_RETRIES = 5
RETRY_BACKOFF_BASE = 2


class TelethonClientManager:
    def __init__(self):
        self._client: TelegramClient | None = None
        self._connected = False

    @property
    def client(self) -> TelegramClient:
        if self._client is None:
            raise RuntimeError("client not initialized")
        return self._client

    async def connect(self) -> TelegramClient:
        session_path = f"sessions/{settings.tg_session_name}"
        self._client = TelegramClient(
            session_path,
            settings.tg_api_id,
            settings.tg_api_hash,
        )

        for attempt in range(MAX_RETRIES):
            try:
                await self._client.connect()

                if not await self._client.is_user_authorized():
                    await log.ainfo("requesting_auth_code", phone=settings.tg_phone)
                    await self._client.send_code_request(settings.tg_phone)
                    raise RuntimeError(
                        "Telethon session not authorized. "
                        "Run the client interactively first to complete login."
                    )

                self._connected = True
                me = await self._client.get_me()
                await log.ainfo(
                    "telethon_connected",
                    user_id=me.id,
                    username=me.username,
                )
                return self._client

            except FloodWaitError as e:
                wait = e.seconds
                await log.awarning("flood_wait", seconds=wait, attempt=attempt + 1)
                await asyncio.sleep(wait)

            except AuthKeyUnregisteredError:
                await log.aerror("auth_key_unregistered")
                raise

            except (ConnectionError, OSError) as e:
                backoff = RETRY_BACKOFF_BASE ** attempt
                await log.awarning(
                    "connection_failed",
                    error=str(e),
                    retry_in=backoff,
                    attempt=attempt + 1,
                )
                await asyncio.sleep(backoff)

        raise RuntimeError(f"failed to connect after {MAX_RETRIES} attempts")

    async def disconnect(self):
        if self._client and self._connected:
            await self._client.disconnect()
            self._connected = False
            await log.ainfo("telethon_disconnected")
