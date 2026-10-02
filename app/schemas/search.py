from pydantic import BaseModel, Field


class VectorSearchRequest(BaseModel):

    question: str = Field(
        min_length=1,
        max_length=5000
    )

    top_k: int = Field(
        default=10,
        ge=1,
        le=50
    )