from __future__ import annotations

import asyncio

from telethon import TelegramClient
from telethon.errors import FloodWaitError

from conversation_engine.config import EngineConfig
from core.config import settings
from core.logging import get_logger

log = get_logger(__name__)


class TelegramSender:
    def __init__(self, config: EngineConfig):
        self.config = config
        self._client: TelegramClient | None = None

    @property
    def client(self) -> TelegramClient:
        if self._client is None:
            raise RuntimeError("sender client not connected")
        return self._client

    async def connect(self) -> None:
        session_path = f"sessions/{self.config.conversation_tg_session_name}"
        self._client = TelegramClient(session_path, settings.tg_api_id, settings.tg_api_hash)
        await self._client.connect()
        if not await self._client.is_user_authorized():
            await self._client.send_code_request(settings.tg_phone)
            raise RuntimeError(
                "Conversation Telethon session is not authorized. "
                "Authorize CONVERSATION_TG_SESSION_NAME interactively before starting the engine."
            )
        me = await self._client.get_me()
        await log.ainfo("conversation_sender_connected", user_id=me.id, username=me.username)

    async def get_bot_user_id(self) -> int:
        me = await self.client.get_me()
        return int(me.id)

    async def send_message(self, chat_id: int, text: str, reply_to_message_id: int | None = None) -> int:
        while True:
            try:
                sent = await self.client.send_message(chat_id, text, reply_to=reply_to_message_id)
                return int(sent.id)
            except FloodWaitError as exc:
                await log.awarning("conversation_sender_flood_wait", seconds=exc.seconds)
                await asyncio.sleep(exc.seconds)

    async def close(self) -> None:
        if self._client:
            await self._client.disconnect()
            await log.ainfo("conversation_sender_disconnected")
