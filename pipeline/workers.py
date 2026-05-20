from datetime import datetime, timezone

from core.constants import EventType, MessageType
from core.logging import get_logger
from core.schemas import CanonicalMessage, RawTelegramEvent
from core.utils import clean_text, extract_mentions
from storage.database import async_session_factory
from storage.repositories import (
    ChatMemberRepository,
    ChatRepository,
    MessageRepository,
    SenderRepository,
)

log = get_logger(__name__)

_MEDIA_TYPE_MAP = {
    "photo": MessageType.PHOTO,
    "document": MessageType.DOCUMENT,
    "video": MessageType.VIDEO,
    "sticker": MessageType.STICKER,
    "voice": MessageType.VOICE,
    "video_note": MessageType.VIDEO_NOTE,
    "animation": MessageType.ANIMATION,
    "contact": MessageType.CONTACT,
    "location": MessageType.LOCATION,
    "poll": MessageType.POLL,
}


class MessageWorker:
    async def connect(self):
        await log.ainfo("worker_ready")

    async def process(self, event: RawTelegramEvent):
        if event.event_type == EventType.DELETE:
            await self._handle_delete(event)
        elif event.event_type == EventType.EDIT:
            await self._handle_edit(event)
        elif event.event_type == EventType.NEW_MESSAGE:
            await self._handle_new_message(event)
        elif event.event_type == EventType.CHAT_ACTION:
            await self._handle_chat_action(event)

    async def _handle_new_message(self, event: RawTelegramEvent):
        msg = self._transform(event)

        async with async_session_factory() as session:
            async with session.begin():
                msg_repo = MessageRepository(session)
                await msg_repo.upsert_message(msg)

                if event.sender_info:
                    sender_repo = SenderRepository(session)
                    await sender_repo.upsert(
                        sender_id=event.sender_info.sender_id,
                        username=event.sender_info.username,
                        first_name=event.sender_info.first_name,
                        last_name=event.sender_info.last_name,
                        is_bot=event.sender_info.is_bot,
                        is_premium=event.sender_info.is_premium,
                    )

                chat_repo = ChatRepository(session)
                await chat_repo.upsert(chat_id=event.chat_id)

                if event.sender_id:
                    member_repo = ChatMemberRepository(session)
                    await member_repo.upsert(chat_id=event.chat_id, sender_id=event.sender_id)

        await log.adebug(
            "message_persisted",
            chat_id=event.chat_id,
            message_id=event.message_id,
        )

    async def _handle_edit(self, event: RawTelegramEvent):
        cleaned = clean_text(event.text)

        async with async_session_factory() as session:
            async with session.begin():
                msg_repo = MessageRepository(session)
                await msg_repo.apply_edit(
                    chat_id=event.chat_id,
                    message_id=event.message_id,
                    new_text_raw=event.text or "",
                    new_text_cleaned=cleaned or "",
                    edited_at=event.timestamp,
                )

        await log.adebug("edit_applied", chat_id=event.chat_id, message_id=event.message_id)

    async def _handle_delete(self, event: RawTelegramEvent):
        ids = event.deleted_message_ids or [event.message_id]
        now = datetime.now(timezone.utc)

        async with async_session_factory() as session:
            async with session.begin():
                msg_repo = MessageRepository(session)
                await msg_repo.apply_deletions(
                    chat_id=event.chat_id,
                    message_ids=ids,
                    observed_at=now,
                )

        await log.adebug("deletions_applied", chat_id=event.chat_id, count=len(ids))

    async def _handle_chat_action(self, event: RawTelegramEvent):
        await log.adebug("chat_action_received", chat_id=event.chat_id, raw=event.raw)

    def _transform(self, event: RawTelegramEvent) -> CanonicalMessage:
        cleaned = clean_text(event.text)
        mentions = extract_mentions(event.text)

        msg_type = MessageType.TEXT
        if event.media and event.media.media_type:
            msg_type = _MEDIA_TYPE_MAP.get(event.media.media_type, MessageType.OTHER)

        forward_from_id = None
        forward_from_chat_id = None
        forward_from_message_id = None
        forward_from_name = None
        forward_date = None
        if event.forward:
            forward_from_id = event.forward.from_id
            forward_from_chat_id = event.forward.from_chat_id
            forward_from_message_id = event.forward.from_message_id
            forward_from_name = event.forward.from_name
            forward_date = event.forward.date

        media_type = None
        media_file_id = None
        media_file_size = None
        media_mime_type = None
        media_duration = None
        media_width = None
        media_height = None
        if event.media:
            media_type = event.media.media_type
            media_file_id = event.media.file_id
            media_file_size = event.media.file_size
            media_mime_type = event.media.mime_type
            media_duration = event.media.duration
            media_width = event.media.width
            media_height = event.media.height

        return CanonicalMessage(
            message_id=event.message_id,
            chat_id=event.chat_id,
            sender_id=event.sender_id,
            timestamp=event.timestamp,
            message_type=msg_type,
            text_raw=event.text,
            text_cleaned=cleaned,
            reply_to_message_id=event.reply_to_message_id,
            forward_from_id=forward_from_id,
            forward_from_chat_id=forward_from_chat_id,
            forward_from_message_id=forward_from_message_id,
            forward_from_name=forward_from_name,
            forward_date=forward_date,
            media_type=media_type,
            media_file_id=media_file_id,
            media_file_size=media_file_size,
            media_mime_type=media_mime_type,
            media_duration=media_duration,
            media_width=media_width,
            media_height=media_height,
            mention_list=mentions,
            entity_list=event.entities,
            grouped_id=event.grouped_id,
            raw_event=event.raw,
        )

    async def close(self):
        await log.ainfo("worker_shutdown")
