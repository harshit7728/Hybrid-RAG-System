from fastapi import FastAPI

from app.schemas.request import ChatRequest
from app.schemas.state import RAGState
from contextlib import asynccontextmanager



from app.db.init_db import init_db
from app.db.database import engine


from app.api.routes import router
from app.api.chat.routes import router as chat_route

@asynccontextmanager
async def lifespan(app: FastAPI):

    await init_db()

    yield

    await engine.dispose()




app = FastAPI(
    title="Hybrid RAG API",
    description="FastAPI Hybrid RAG with RRF and Reranking",
    version="1.0.0"
)

app.include_router(router)
app.include_router(chat_route)


@app.get("/")
async def health_check():
    return {
        "status": "healthy",
        "service": "Hybrid RAG API"
    }


@app.post("/api/chat")
async def chat(request: ChatRequest):

    state = RAGState(
        original_query=request.question
    )

    return {
        "message": "RAG pipeline initialized",
        "state": state.model_dump()
    }