from google import genai
from google.genai import types

from app.core.config import settings


client = genai.Client(
    api_key=settings.GEMINI_API_KEY
)


async def generate_embeddings(
    texts: list[str],
    task_type: str = "RETRIEVAL_DOCUMENT"
):

    response = await client.aio.models.embed_content(
        model=settings.EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=settings.EMBEDDING_DIMENSION
        )
    )

    return [
        embedding.values
        for embedding in response.embeddings
    ]


async def generate_query_embedding(
    query: str
) -> list[float]:

    embeddings = await generate_embeddings(
        texts=[query],
        task_type="RETRIEVAL_QUERY"
    )

    return embeddings[0]