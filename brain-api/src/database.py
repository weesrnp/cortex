import logging
from psycopg_pool import AsyncConnectionPool
from .config import settings

logger = logging.getLogger("cortex.database")

class Database:
    def __init__(self):
        self.pool = AsyncConnectionPool(
            conninfo=settings.database_url,
            min_size=2,
            max_size=10,
            timeout=10.0,
            open=False
        )

    async def open(self):
        await self.pool.open()
        logger.info("Pool de conexões PostgreSQL iniciado.")

    async def close(self):
        await self.pool.close()
        logger.info("Pool de conexões PostgreSQL encerrado.")

db = Database()
