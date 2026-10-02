from sqlalchemy import select

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import DocumentChunk

from app.schemas.state import RetrievedDocument


async def vector_search(
    db: AsyncSession,
    query_embedding: list[float],
    top_k: int = 10
) -> list[RetrievedDocument]:

    distance = DocumentChunk.embedding.cosine_distance(
        query_embedding
    )

    statement = (
        select(
            DocumentChunk,
            distance.label("distance")
        )
        .where(
            DocumentChunk.embedding.is_not(None)
        )
        .order_by(distance)
        .limit(top_k)
    )

    result = await db.execute(statement)

    rows = result.all()

    documents = []

    for chunk, distance_score in rows:

        documents.append(
            RetrievedDocument(
                id=str(chunk.id),
                content=chunk.content,
                metadata=chunk.metadata_json,
                score=float(1 - distance_score),
                source="vector"
            )
        )

    return documents