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

    async def send_reaction(self, chat_id: int, message_id: int, emoji: str) -> bool:
        from telethon.tl.functions.messages import SendReactionRequest
        from telethon.tl.types import ReactionEmoji

        while True:
            try:
                peer = await self.client.get_input_entity(chat_id)
                await self.client(
                    SendReactionRequest(
                        peer=peer,
                        msg_id=message_id,
                        reaction=[ReactionEmoji(emoticon=emoji)],
                    )
                )
                return True
            except FloodWaitError as exc:
                await log.awarning("conversation_sender_flood_wait", seconds=exc.seconds)
                await asyncio.sleep(exc.seconds)
            except Exception as exc:  # noqa: BLE001 - reactions are best-effort
                await log.awarning("conversation_sender_reaction_failed", chat_id=chat_id, message_id=message_id, error=str(exc))
                return False

    async def send_typing(self, chat_id: int) -> None:
        try:
            async with self.client.action(chat_id, "typing"):
                pass
        except Exception as exc:  # noqa: BLE001 - typing is best-effort
            await log.awarning("conversation_sender_typing_failed", chat_id=chat_id, error=str(exc))

    async def send_sticker(
        self,
        chat_id: int,
        *,
        source_message_id: int | None = None,
        file=None,
        reply_to_message_id: int | None = None,
    ) -> int | None:
        """Resend a sticker/media into ``chat_id``.

        If ``source_message_id`` is given, the source message is fetched and its
        ``.media`` is re-sent via ``send_file``. Otherwise ``file`` is sent directly.

        NOTE: the source-message resend path is UNVERIFIED against live Telegram
        (resending media the bot did not author may require a fresh file reference)
        and is strictly opt-in. Callers must pass an explicit source/file.
        """
        while True:
            try:
                media = file
                if source_message_id is not None:
                    msg = await self.client.get_messages(chat_id, ids=source_message_id)
                    if msg is None or msg.media is None:
                        return None
                    media = msg.media
                if media is None:
                    return None
                sent = await self.client.send_file(chat_id, media, reply_to=reply_to_message_id)
                return int(sent.id)
            except FloodWaitError as exc:
                await log.awarning("conversation_sender_flood_wait", seconds=exc.seconds)
                await asyncio.sleep(exc.seconds)
            except Exception as exc:  # noqa: BLE001 - media resend is best-effort
                await log.awarning("conversation_sender_sticker_failed", chat_id=chat_id, error=str(exc))
                return None

    async def close(self) -> None:
        if self._client:
            await self._client.disconnect()
            await log.ainfo("conversation_sender_disconnected")
