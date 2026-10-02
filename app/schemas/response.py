from pydantic import BaseModel


class Source(BaseModel):

    document_id: str

    filename: str

    page_number: int | str

    score: float | None = None


class RAGResponse(BaseModel):

    answer: str

    sources: list[Source]