from app.schemas.state import RetrievedDocument


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievedDocument]],
    k: int = 60,
    top_k: int = 20
) -> list[RetrievedDocument]:

    scores = {}

    documents = {}

    for results in result_lists:

        for rank, document in enumerate(
            results,
            start=1
        ):

            document_id = document.id

            rrf_score = 1 / (k + rank)

            scores[document_id] = (
                scores.get(document_id, 0)
                + rrf_score
            )

            documents[document_id] = document

    ranked_documents = sorted(
        documents.values(),
        key=lambda document: scores[document.id],
        reverse=True
    )

    fused_results = []

    for document in ranked_documents[:top_k]:

        fused_results.append(
            document.model_copy(
                update={
                    "score": scores[document.id],
                    "source": "rrf"
                }
            )
        )

    return fused_results