from datetime import datetime

from sqlalchemy import select, func, update, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from core.schemas import CanonicalMessage
from storage.postgres_models import Chat, ChatMember, DeletionEvent, Message, Sender


class MessageRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def upsert_message(self, msg: CanonicalMessage) -> Message:
        stmt = insert(Message).values(
            message_id=msg.message_id,
            chat_id=msg.chat_id,
            sender_id=msg.sender_id,
            timestamp=msg.timestamp,
            message_type=msg.message_type.value,
            text_raw=msg.text_raw,
            text_cleaned=msg.text_cleaned,
            reply_to_message_id=msg.reply_to_message_id,
            forward_from_id=msg.forward_from_id,
            forward_from_chat_id=msg.forward_from_chat_id,
            forward_from_message_id=msg.forward_from_message_id,
            forward_from_name=msg.forward_from_name,
            forward_date=msg.forward_date,
            media_type=msg.media_type,
            media_file_id=msg.media_file_id,
            media_file_size=msg.media_file_size,
            media_mime_type=msg.media_mime_type,
            media_duration=msg.media_duration,
            media_width=msg.media_width,
            media_height=msg.media_height,
            mention_list=msg.mention_list,
            entity_list=msg.entity_list,
            grouped_id=msg.grouped_id,
            raw_event=msg.raw_event,
        )

        stmt = stmt.on_conflict_do_update(
            constraint="uq_chat_message",
            set_={
                "text_raw": stmt.excluded.text_raw,
                "text_cleaned": stmt.excluded.text_cleaned,
                "sender_id": stmt.excluded.sender_id,
                "mention_list": stmt.excluded.mention_list,
                "entity_list": stmt.excluded.entity_list,
                "raw_event": stmt.excluded.raw_event,
            },
        )

        await self.session.execute(stmt)
        await self.session.flush()

        result = await self.session.execute(
            select(Message).where(
                Message.chat_id == msg.chat_id, Message.message_id == msg.message_id
            )
        )
        return result.scalar_one()

    async def apply_edit(self, chat_id: int, message_id: int, new_text_raw: str, new_text_cleaned: str, edited_at: datetime):
        result = await self.session.execute(
            select(Message).where(Message.chat_id == chat_id, Message.message_id == message_id)
        )
        row = result.scalar_one_or_none()
        if row is None:
            return

        history = list(row.edit_history) if row.edit_history else []
        history.append({"text": row.text_raw, "edited_at": edited_at.isoformat()})

        await self.session.execute(
            update(Message)
            .where(Message.chat_id == chat_id, Message.message_id == message_id)
            .values(text_raw=new_text_raw, text_cleaned=new_text_cleaned, edit_history=history)
        )
        await self.session.flush()

    async def apply_reactions(self, chat_id: int, message_id: int, reactions: list[dict], reaction_count: int):
        await self.session.execute(
            update(Message)
            .where(Message.chat_id == chat_id, Message.message_id == message_id)
            .values(reactions=reactions, reaction_count=reaction_count)
        )
        await self.session.flush()

    async def apply_deletions(self, chat_id: int, message_ids: list[int], observed_at: datetime):
        if not message_ids:
            return

        await self.session.execute(
            update(Message)
            .where(Message.chat_id == chat_id, Message.message_id.in_(message_ids))
            .values(is_deleted=True, deleted_at=observed_at)
        )

        self.session.add(DeletionEvent(
            chat_id=chat_id,
            message_ids=message_ids,
            observed_at=observed_at,
        ))
        await self.session.flush()

    async def get_messages(
        self,
        chat_id: int,
        limit: int = 50,
        offset: int = 0,
        include_deleted: bool = False,
    ) -> tuple[list[Message], int]:
        base = select(Message).where(Message.chat_id == chat_id)
        if not include_deleted:
            base = base.where(Message.is_deleted.is_(False))

        count_result = await self.session.execute(
            select(func.count()).select_from(base.subquery())
        )
        total = count_result.scalar_one()

        result = await self.session.execute(
            base.order_by(Message.timestamp.desc()).limit(limit).offset(offset)
        )
        return list(result.scalars().all()), total

    async def get_thread(self, chat_id: int, message_id: int) -> list[Message]:
        cte_sql = text("""
            WITH RECURSIVE find_root AS (
                SELECT id, message_id, reply_to_message_id
                FROM messages
                WHERE chat_id = :chat_id AND message_id = :msg_id
                UNION ALL
                SELECT m.id, m.message_id, m.reply_to_message_id
                FROM messages m
                JOIN find_root fr ON m.message_id = fr.reply_to_message_id AND m.chat_id = :chat_id
            ),
            root AS (
                SELECT message_id FROM find_root WHERE reply_to_message_id IS NULL LIMIT 1
            ),
            thread AS (
                SELECT m.id, m.message_id, m.reply_to_message_id
                FROM messages m, root r
                WHERE m.chat_id = :chat_id AND m.message_id = r.message_id
                UNION ALL
                SELECT m.id, m.message_id, m.reply_to_message_id
                FROM messages m
                JOIN thread t ON m.reply_to_message_id = t.message_id AND m.chat_id = :chat_id
            )
            SELECT thread.id FROM thread
        """)

        result = await self.session.execute(cte_sql, {"chat_id": chat_id, "msg_id": message_id})
        ids = [row[0] for row in result.fetchall()]

        if not ids:
            return []

        result = await self.session.execute(
            select(Message).where(Message.id.in_(ids)).order_by(Message.timestamp.asc())
        )
        return list(result.scalars().all())


class SenderRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def upsert(self, sender_id: int, username: str | None = None,
                     first_name: str | None = None, last_name: str | None = None,
                     is_bot: bool = False, is_premium: bool = False):
        stmt = insert(Sender).values(
            sender_id=sender_id,
            username=username,
            first_name=first_name,
            last_name=last_name,
            is_bot=is_bot,
            is_premium=is_premium,
            last_seen_at=func.now(),
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["sender_id"],
            set_={
                "username": stmt.excluded.username,
                "first_name": stmt.excluded.first_name,
                "last_name": stmt.excluded.last_name,
                "is_bot": stmt.excluded.is_bot,
                "is_premium": stmt.excluded.is_premium,
                "last_seen_at": func.now(),
            },
        )
        await self.session.execute(stmt)


class ChatRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def upsert(self, chat_id: int, title: str | None = None, chat_type: str | None = None):
        stmt = insert(Chat).values(
            chat_id=chat_id,
            title=title,
            chat_type=chat_type,
            last_message_at=func.now(),
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["chat_id"],
            set_={
                "title": stmt.excluded.title,
                "chat_type": stmt.excluded.chat_type,
                "last_message_at": func.now(),
            },
        )
        await self.session.execute(stmt)


class ChatMemberRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def upsert(self, chat_id: int, sender_id: int):
        stmt = insert(ChatMember).values(
            chat_id=chat_id,
            sender_id=sender_id,
            last_seen_at=func.now(),
            message_count=1,
        )
        stmt = stmt.on_conflict_do_update(
            constraint="uq_chat_sender",
            set_={
                "last_seen_at": func.now(),
                "message_count": ChatMember.message_count + 1,
            },
        )
        await self.session.execute(stmt)
