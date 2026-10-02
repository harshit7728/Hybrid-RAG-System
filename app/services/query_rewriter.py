from google import genai
from google.genai import types

from app.core.config import settings


client = genai.Client(
    api_key=settings.GEMINI_API_KEY
)


REWRITE_SYSTEM_PROMPT = """
You are a search query rewriting system.

Your job is to rewrite the user's question into a clear,
self-contained query optimized for document retrieval.

Rules:

1. Preserve the user's original meaning.
2. Do not answer the question.
3. Do not add facts that are not present in the query.
4. Remove unnecessary conversational wording.
5. Return ONLY the rewritten query.
"""


async def rewrite_query(query: str) -> str:

    response = await client.aio.models.generate_content(
        model=settings.GEMINI_MODEL,
        contents=query,
        config=types.GenerateContentConfig(
            system_instruction=REWRITE_SYSTEM_PROMPT,
            temperature=0.0,
            max_output_tokens=200,
        ),
    )

    rewritten = response.text.strip()

    return rewritten or query