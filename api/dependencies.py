from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from storage.database import async_session_factory
from storage.repositories import MessageRepository


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        yield session


async def get_message_repo(session: AsyncSession) -> MessageRepository:
    return MessageRepository(session)
