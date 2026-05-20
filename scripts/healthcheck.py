import asyncio
import sys

import asyncpg
import redis.asyncio as aioredis

from core.config import settings


async def check():
    pg_ok = False
    redis_ok = False

    try:
        url = settings.database_url.replace("+asyncpg", "")
        conn = await asyncpg.connect(url)
        await conn.fetchval("SELECT 1")
        await conn.close()
        pg_ok = True
    except Exception:
        pass

    try:
        r = aioredis.from_url(settings.redis_url)
        await r.ping()
        await r.aclose()
        redis_ok = True
    except Exception:
        pass

    if pg_ok and redis_ok:
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    asyncio.run(check())
