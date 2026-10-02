from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import DocumentChunk
from app.schemas.state import RetrievedDocument


async def bm25_search(
    db: AsyncSession,
    query: str,
    top_k: int = 10
) -> list[RetrievedDocument]:

    # Convert user query into PostgreSQL search query
    search_query = func.websearch_to_tsquery(
        "english",
        query
    )

    # Calculate keyword relevance
    rank = func.ts_rank_cd(
        DocumentChunk.search_vector,
        search_query
    )

    # Build SQL query
    statement = (
        select(
            DocumentChunk,
            rank.label("rank")
        )
        .where(
            DocumentChunk.search_vector.op("@@")(
                search_query
            )
        )
        .order_by(desc(rank))
        .limit(top_k)
    )

    result = await db.execute(statement)

    rows = result.all()

    documents = []

    for chunk, rank_score in rows:

        documents.append(
            RetrievedDocument(
                id=str(chunk.id),
                content=chunk.content,
                metadata=chunk.metadata_json,
                score=float(rank_score),
                source="bm25"
            )
        )

    return documents