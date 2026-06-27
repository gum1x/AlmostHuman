from collections.abc import AsyncGenerator

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings
from storage.database import async_session_factory
from storage.repositories import MessageRepository


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        yield session


async def get_message_repo(
    session: AsyncSession = Depends(get_session),
) -> MessageRepository:
    return MessageRepository(session)


# auto_error=False so a missing/garbage Authorization header reaches our handler,
# letting us fail closed with a consistent 401/503 instead of FastAPI's default 403.
_bearer = HTTPBearer(auto_error=False)


async def require_auth(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    """Guard data endpoints that return chat content / PII.

    Fails closed: if ``API_AUTH_TOKEN`` is unset the server cannot authenticate
    anyone, so we reject with 503 rather than leak PII to unauthenticated callers.
    """
    expected = settings.api_auth_token
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API auth is not configured (API_AUTH_TOKEN unset).",
        )
    if credentials is None or credentials.credentials != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
