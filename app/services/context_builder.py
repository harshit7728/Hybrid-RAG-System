from app.schemas.state import RetrievedDocument


def build_context(
    documents: list[RetrievedDocument],
    max_chunks: int = 5,
    max_chars_per_chunk: int = 4000
) -> str:

    if not documents:
        return ""

    selected_documents = documents[:max_chunks]

    context_parts = []

    for index, document in enumerate(
        selected_documents,
        start=1
    ):

        metadata = document.metadata or {}

        filename = metadata.get(
            "filename",
            "Unknown"
        )

        page_number = metadata.get(
            "page_number",
            "Unknown"
        )

        content = document.content[:max_chars_per_chunk]

        context_parts.append(
            f"""
[Source {index}]
Document: {filename}
Page: {page_number}

Content:
{content}
""".strip()
        )

    return "\n\n---\n\n".join(context_parts)


def build_sources(
    documents: list[RetrievedDocument]
) -> list[dict]:

    sources = []

    for document in documents:

        metadata = document.metadata or {}

        sources.append({
            "document_id": document.id,
            "filename": metadata.get(
                "filename",
                "Unknown"
            ),
            "page_number": metadata.get(
                "page_number",
                "Unknown"
            ),
            "score": document.score
        })

    return sources