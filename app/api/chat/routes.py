from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.schemas.request import ChatRequest
from app.services.context_builder import build_sources
from app.pipeline.rag_pipeline import run_rag_pipeline


router = APIRouter(
    prefix="/api",
    tags=["RAG"]
)


@router.post("/chat")
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db)
):

    state = await run_rag_pipeline(
        question=request.question,
        db=db
    )

    sources = build_sources(
        state.reranked_results
    )

    return {
        "question": request.question,
        "answer": state.answer,
        "sources": sources
    }