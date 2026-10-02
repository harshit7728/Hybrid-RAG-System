from sqlalchemy import text

from app.db.database import engine, Base

from app.db.models import Document, DocumentChunk


async def init_db():

    async with engine.begin() as connection:

        await connection.execute(
            text("CREATE EXTENSION IF NOT EXISTS vector")
        )

        await connection.run_sync(
            Base.metadata.create_all
        )

    print("Database initialized successfully")