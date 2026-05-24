from datetime import timezone

from telethon import events, types

from core.constants import EventType
from core.logging import get_logger
from core.schemas import ForwardMetadata, MediaMetadata, RawTelegramEvent, SenderInfo
from pipeline.queue_producer import QueueProducer

log = get_logger(__name__)


def _extract_media(media) -> MediaMetadata | None:
    if media is None:
        return None

    if isinstance(media, types.MessageMediaPhoto):
        photo = media.photo
        if photo and hasattr(photo, "sizes") and photo.sizes:
            largest = photo.sizes[-1]
            w = getattr(largest, "w", None)
            h = getattr(largest, "h", None)
        else:
            w, h = None, None
        return MediaMetadata(
            media_type="photo",
            file_id=str(photo.id) if photo else None,
            file_size=getattr(photo, "size", None) if photo else None,
            width=w,
            height=h,
        )

    if isinstance(media, types.MessageMediaDocument):
        doc = media.document
        if doc is None:
            return MediaMetadata(media_type="document")

        mime = getattr(doc, "mime_type", None)
        size = getattr(doc, "size", None)
        duration = None
        w, h = None, None
        media_type = "document"

        for attr in getattr(doc, "attributes", []):
            if isinstance(attr, types.DocumentAttributeVideo):
                media_type = "video"
                duration = attr.duration
                w, h = attr.w, attr.h
            elif isinstance(attr, types.DocumentAttributeAudio):
                media_type = "voice" if attr.voice else "audio"
                duration = attr.duration
            elif isinstance(attr, types.DocumentAttributeSticker):
                media_type = "sticker"
            elif isinstance(attr, types.DocumentAttributeAnimated):
                media_type = "animation"
            elif isinstance(attr, types.DocumentAttributeImageSize):
                w, h = attr.w, attr.h

        return MediaMetadata(
            media_type=media_type,
            file_id=str(doc.id),
            file_size=size,
            mime_type=mime,
            duration=duration,
            width=w,
            height=h,
        )

    if isinstance(media, types.MessageMediaContact):
        return MediaMetadata(media_type="contact")

    if isinstance(media, types.MessageMediaGeo):
        return MediaMetadata(media_type="location")

    if isinstance(media, types.MessageMediaPoll):
        return MediaMetadata(media_type="poll")

    return MediaMetadata(media_type="other")


def _extract_forward(fwd) -> ForwardMetadata | None:
    if fwd is None:
        return None

    from_id = None
    from_chat_id = None
    if fwd.from_id:
        if isinstance(fwd.from_id, types.PeerUser):
            from_id = fwd.from_id.user_id
        elif isinstance(fwd.from_id, types.PeerChannel):
            from_chat_id = fwd.from_id.channel_id
        elif isinstance(fwd.from_id, types.PeerChat):
            from_chat_id = fwd.from_id.chat_id

    return ForwardMetadata(
        from_id=from_id,
        from_name=fwd.from_name,
        from_chat_id=from_chat_id,
        from_message_id=fwd.channel_post,
        date=fwd.date.replace(tzinfo=timezone.utc) if fwd.date else None,
    )


def _extract_entities(msg) -> list[dict]:
    if not msg.entities:
        return []

    result = []
    for ent in msg.entities:
        entry = {
            "type": type(ent).__name__,
            "offset": ent.offset,
            "length": ent.length,
        }
        if hasattr(ent, "url") and ent.url:
            entry["url"] = ent.url
        if hasattr(ent, "user_id") and ent.user_id:
            entry["user_id"] = ent.user_id
        result.append(entry)
    return result


async def _get_sender_info(event) -> SenderInfo | None:
    try:
        sender = await event.get_sender()
        if sender is None:
            return None
        return SenderInfo(
            sender_id=sender.id,
            username=getattr(sender, "username", None),
            first_name=getattr(sender, "first_name", None),
            last_name=getattr(sender, "last_name", None),
            is_bot=getattr(sender, "bot", False) or False,
            is_premium=getattr(sender, "premium", False) or False,
        )
    except Exception:
        return None


def _msg_to_raw_dict(msg, chat_type: str | None = None) -> dict:
    try:
        return {
            "id": msg.id,
            "chat_id": msg.chat_id,
            "sender_id": msg.sender_id,
            "text": msg.text,
            "date": msg.date.isoformat() if msg.date else None,
            "chat_type": chat_type,
            "is_private": chat_type == "private",
        }
    except Exception:
        return {}


async def _produce_new_message(event, producer: QueueProducer, chat_type: str) -> None:
    msg = event.message
    sender_info = await _get_sender_info(event)

    if sender_info and sender_info.is_bot:
        return

    ts = msg.date
    if ts and ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    raw_event = RawTelegramEvent(
        event_type=EventType.NEW_MESSAGE,
        message_id=msg.id,
        chat_id=msg.chat_id,
        sender_id=msg.sender_id,
        timestamp=ts,
        text=msg.text,
        reply_to_message_id=msg.reply_to.reply_to_msg_id if msg.reply_to else None,
        forward=_extract_forward(msg.fwd_from),
        media=_extract_media(msg.media),
        entities=_extract_entities(msg),
        grouped_id=msg.grouped_id,
        sender_info=sender_info,
        raw=_msg_to_raw_dict(msg, chat_type),
    )

    try:
        await producer.produce(raw_event)
    except Exception:
        await log.aexception("produce_failed", message_id=msg.id, chat_id=msg.chat_id)


def register_handlers(client, producer: QueueProducer, chat_ids: list[int], monitor_private_dms: bool = True):
    chats = chat_ids if chat_ids else None

    @client.on(events.NewMessage(chats=chats))
    async def on_new_message(event):
        if event.is_private:
            return
        await _produce_new_message(event, producer, "group")

    if monitor_private_dms:
        @client.on(events.NewMessage(incoming=True))
        async def on_private_message(event):
            if not event.is_private:
                return
            await _produce_new_message(event, producer, "private")

    @client.on(events.MessageEdited(chats=chats))
    async def on_message_edited(event):
        msg = event.message

        ts = msg.edit_date or msg.date
        if ts and ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        raw_event = RawTelegramEvent(
            event_type=EventType.EDIT,
            message_id=msg.id,
            chat_id=msg.chat_id,
            sender_id=msg.sender_id,
            timestamp=ts,
            text=msg.text,
            reply_to_message_id=msg.reply_to.reply_to_msg_id if msg.reply_to else None,
            media=_extract_media(msg.media),
            entities=_extract_entities(msg),
            raw=_msg_to_raw_dict(msg, "private" if event.is_private else "group"),
        )

        try:
            await producer.produce(raw_event)
        except Exception:
            await log.aexception("produce_edit_failed", message_id=msg.id)

    @client.on(events.MessageDeleted())
    async def on_message_deleted(event):
        chat_id = getattr(event, "chat_id", None) or 0
        deleted_ids = event.deleted_ids or []

        if not deleted_ids:
            return

        from datetime import datetime
        raw_event = RawTelegramEvent(
            event_type=EventType.DELETE,
            message_id=deleted_ids[0],
            chat_id=chat_id,
            timestamp=datetime.now(timezone.utc),
            deleted_message_ids=deleted_ids,
            raw={"deleted_ids": deleted_ids, "chat_id": chat_id},
        )

        try:
            await producer.produce(raw_event)
        except Exception:
            await log.aexception("produce_delete_failed", chat_id=chat_id)

    @client.on(events.ChatAction(chats=chats))
    async def on_chat_action(event):
        from datetime import datetime
        raw_event = RawTelegramEvent(
            event_type=EventType.CHAT_ACTION,
            message_id=0,
            chat_id=event.chat_id,
            sender_id=event.user_id if hasattr(event, "user_id") else None,
            timestamp=datetime.now(timezone.utc),
            raw={
                "action_type": type(event.action_message.action).__name__ if event.action_message and event.action_message.action else "unknown",
                "chat_id": event.chat_id,
            },
        )

        try:
            await producer.produce(raw_event)
        except Exception:
            await log.aexception("produce_action_failed", chat_id=event.chat_id)

    return client
