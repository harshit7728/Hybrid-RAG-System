from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.ingestion.document_ingestion import ingest_pdf
from app.services.bm25_search import bm25_search
from app.schemas.search import VectorSearchRequest
from pydantic import BaseModel

from app.services.gemini_service import generate_answer

from app.services.embedding_service import (
    generate_query_embedding
)

from app.services.vector_search import vector_search
from app.services.reranker import reranker
from app.services.rrf import reciprocal_rank_fusion
from app.services.context_builder import (
    build_context,
    build_sources
)

router = APIRouter(
    prefix="/api/documents",
    tags=["Documents"]
)


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    pdf_bytes = await file.read()

    if not pdf_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty"
        )

    if len(pdf_bytes) > 20 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Maximum file size is 20 MB"
        )

    try:

        result = await ingest_pdf(
            db=db,
            filename=file.filename,
            pdf_bytes=pdf_bytes
        )

        return result

    except ValueError as error:

        raise HTTPException(
            status_code=422,
            detail=str(error)
        )

    except Exception:

        raise HTTPException(
            status_code=500,
            detail="Document ingestion failed"
        )
    


@router.post("/search/vector")
async def search_vector(
    request: VectorSearchRequest,
    db: AsyncSession = Depends(get_db)
):

    # 1. Generate query embedding
    query_embedding = await generate_query_embedding(
        request.question
    )

    # 2. Retrieve relevant chunks
    results = await vector_search(
        db=db,
        query_embedding=query_embedding,
        top_k=request.top_k
    )

    # 3. Return results
    return {
        "query": request.question,
        "total_results": len(results),
        "results": [
            result.model_dump()
            for result in results
        ]
    }



@router.post("/search/bm25")
async def search_bm25(
    request: VectorSearchRequest,
    db: AsyncSession = Depends(get_db)
):

    results = await bm25_search(
        db=db,
        query=request.question,
        top_k=request.top_k
    )

    return {
        "query": request.question,
        "total_results": len(results),
        "results": [
            result.model_dump()
            for result in results
        ]
    }



@router.post("/search/hybrid")
async def hybrid_search(
    request: VectorSearchRequest,
    db: AsyncSession = Depends(get_db)
):

    query = request.question

    # -------------------------
    # Vector retrieval
    # -------------------------

    query_embedding = await generate_query_embedding(
        query
    )

    vector_results = await vector_search(
        db=db,
        query_embedding=query_embedding,
        top_k=10
    )

    # -------------------------
    # BM25 retrieval
    # -------------------------

    bm25_results = await bm25_search(
        db=db,
        query=query,
        top_k=10
    )

    # -------------------------
    # RRF
    # -------------------------

    fused_results = reciprocal_rank_fusion(
        result_lists=[
            vector_results,
            bm25_results
        ],
        k=60,
        top_k=20
    )

    # -------------------------
    # Reranking
    # -------------------------

    reranked_results = reranker.rerank(
        query=query,
        documents=fused_results,
        top_k=5
    )

    return {
        "query": query,

        "vector_results": [
            result.model_dump()
            for result in vector_results
        ],

        "bm25_results": [
            result.model_dump()
            for result in bm25_results
        ],

        "fused_results": [
            result.model_dump()
            for result in fused_results
        ],

        "reranked_results": [
            result.model_dump()
            for result in reranked_results
        ]
    }


@router.post("/search/context")
async def build_search_context(
    request: VectorSearchRequest,
    db: AsyncSession = Depends(get_db)
):

    query = request.question

    # -------------------------
    # Vector search
    # -------------------------

    query_embedding = await generate_query_embedding(
        query
    )

    vector_results = await vector_search(
        db=db,
        query_embedding=query_embedding,
        top_k=10
    )

    # -------------------------
    # BM25
    # -------------------------

    bm25_results = await bm25_search(
        db=db,
        query=query,
        top_k=10
    )

    # -------------------------
    # RRF
    # -------------------------

    fused_results = reciprocal_rank_fusion(
        result_lists=[
            vector_results,
            bm25_results
        ],
        k=60,
        top_k=20
    )

    # -------------------------
    # Reranker
    # -------------------------

    reranked_results = reranker.rerank(
        query=query,
        documents=fused_results,
        top_k=5
    )

    # -------------------------
    # Context
    # -------------------------

    context = build_context(
        documents=reranked_results
    )

    sources = build_sources(
        documents=reranked_results
    )

    return {
        "query": query,
        "context": context,
        "sources": sources
    }






class GenerationTestRequest(BaseModel):
    question: str
    context: str


@router.post("/test/generate")
async def test_generation(request: GenerationTestRequest):

    answer = await generate_answer(
        query=request.question,
        context=request.context
    )

    return {
        "question": request.question,
        "answer": answer
    }



