from typing import Any

from pydantic import BaseModel, Field


class RetrievedDocument(BaseModel):
    id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float | None = None
    source: str | None = None


class RAGState(BaseModel):
    original_query: str

    rewritten_query: str | None = None

    route: str = "hybrid"

    vector_results: list[RetrievedDocument] = Field(
        default_factory=list
    )

    bm25_results: list[RetrievedDocument] = Field(
        default_factory=list
    )

    fused_results: list[RetrievedDocument] = Field(
        default_factory=list
    )

    reranked_results: list[RetrievedDocument] = Field(
        default_factory=list
    )

    context: str = ""

    answer: str = ""