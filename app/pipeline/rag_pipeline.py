from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.state import RAGState

from app.services.embedding_service import (
    generate_query_embedding
)

from app.services.vector_search import vector_search
from app.services.bm25_search import bm25_search
from app.services.rrf import reciprocal_rank_fusion
from app.services.reranker import reranker
from app.services.context_builder import build_context

from app.services.gemini_service import generate_answer
from app.services.query_router import route_query
from app.services.query_rewriter import rewrite_query

async def run_vector_pipeline(
    state: RAGState,
    db: AsyncSession
) -> RAGState:

    # Generate embedding
    query_embedding = await generate_query_embedding(
        state.original_query
    )

    # Retrieve documents
    results = await vector_search(
        db=db,
        query_embedding=query_embedding,
        top_k=10
    )

    # Update shared state
    state.vector_results = results

    return state


async def run_bm25_pipeline(
    state: RAGState,
    db: AsyncSession
) -> RAGState:

    query = (
        state.rewritten_query
        or state.original_query
    )

    results = await bm25_search(
        db=db,
        query=query,
        top_k=10
    )

    state.bm25_results = results

    return state

async def run_rrf(
    state: RAGState
) -> RAGState:

    state.fused_results = reciprocal_rank_fusion(
        result_lists=[
            state.vector_results,
            state.bm25_results
        ],
        k=60,
        top_k=20
    )

    return state


async def run_reranker(
    state: RAGState
) -> RAGState:

    query = (
        state.rewritten_query
        or state.original_query
    )

    state.reranked_results = reranker.rerank(
        query=query,
        documents=state.fused_results,
        top_k=5
    )

    return state


async def run_context_builder(
    state: RAGState
) -> RAGState:

    state.context = build_context(
        documents=state.reranked_results,
        max_chunks=5,
        max_chars_per_chunk=4000
    )

    return state


async def run_generation(state: RAGState) -> RAGState:

    query = state.rewritten_query or state.original_query

    state.answer = await generate_answer(
        query=query,
        context=state.context
    )

    return state





# async def run_rag_pipeline(
#     question: str,
#     db: AsyncSession
# ) -> RAGState:

#     state = RAGState(
#         original_query=question
#     )

#     # -------------------------
#     # 1. Query Router
#     # -------------------------

#     state.route = route_query(
#         question
#     )

#     # -------------------------
#     # 2. Query Rewriter
#     # -------------------------

#     state.rewritten_query = await rewrite_query(
#         question
#     )

#     retrieval_query = state.rewritten_query

#     # -------------------------
#     # 3. Vector Search
#     # -------------------------

#     query_embedding = await generate_query_embedding(
#         retrieval_query
#     )

#     state.vector_results = await vector_search(
#         db=db,
#         query_embedding=query_embedding,
#         top_k=10
#     )

#     # -------------------------
#     # 4. BM25 Search
#     # -------------------------

#     state.bm25_results = await bm25_search(
#         db=db,
#         query=retrieval_query,
#         top_k=10
#     )

#     # -------------------------
#     # 5. RRF
#     # -------------------------

#     state.fused_results = reciprocal_rank_fusion(
#         result_lists=[
#             state.vector_results,
#             state.bm25_results
#         ],
#         k=60,
#         top_k=20
#     )

#     # -------------------------
#     # 6. Reranking
#     # -------------------------

#     state.reranked_results = reranker.rerank(
#         query=retrieval_query,
#         documents=state.fused_results,
#         top_k=5
#     )

#     # -------------------------
#     # 7. Context Builder
#     # -------------------------

#     state.context = build_context(
#         documents=state.reranked_results,
#         max_chunks=5,
#         max_chars_per_chunk=4000
#     )

#     # -------------------------
#     # 8. Gemini
#     # -------------------------

#     state.answer = await generate_answer(
#         query=question,
#         context=state.context
#     )

#     return state







async def run_rag_pipeline(
    question: str,
    db: AsyncSession
) -> RAGState:

    state = RAGState(
        original_query=question
    )

    # -------------------------
    # 1. Query Router
    # -------------------------

    state.route = route_query(
        question
    )

    # -------------------------
    # 2. Query Rewriter
    # -------------------------

    state.rewritten_query = await rewrite_query(
        question
    )

    retrieval_query = state.rewritten_query

     

    # -------------------------
    # Retrieval
    # -------------------------

    if state.route == "vector":

        query_embedding = await generate_query_embedding(
            retrieval_query
        )

        state.vector_results = await vector_search(
            db=db,
            query_embedding=query_embedding,
            top_k=10
        )

        state.fused_results = state.vector_results

    elif state.route == "bm25":

        state.bm25_results = await bm25_search(
            db=db,
            query=retrieval_query,
            top_k=10
        )

        state.fused_results = state.bm25_results

    else:

        query_embedding = await generate_query_embedding(
            retrieval_query
        )

        state.vector_results = await vector_search(
            db=db,
            query_embedding=query_embedding,
            top_k=10
        )

        state.bm25_results = await bm25_search(
            db=db,
            query=retrieval_query,
            top_k=10
        )

        state.fused_results = reciprocal_rank_fusion(
            result_lists=[
                state.vector_results,
                state.bm25_results
            ],
            k=60,
            top_k=20
        )

    # -------------------------
    # 6. Reranking
    # -------------------------

    state.reranked_results = reranker.rerank(
        query=retrieval_query,
        documents=state.fused_results,
        top_k=5
    )

    # -------------------------
    # 7. Context Builder
    # -------------------------

    state.context = build_context(
        documents=state.reranked_results,
        max_chunks=5,
        max_chars_per_chunk=4000
    )

    # -------------------------
    # 8. Gemini
    # -------------------------

    state.answer = await generate_answer(
        query=question,
        context=state.context
    )

    return state

