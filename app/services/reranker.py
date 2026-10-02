from sentence_transformers import CrossEncoder

from app.schemas.state import RetrievedDocument


class Reranker:

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
        top_k: int = 5
    ) -> list[RetrievedDocument]:

        if not documents:
            return []

        pairs = [
            [query, document.content]
            for document in documents
        ]

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(documents, scores),
            key=lambda item: float(item[1]),
            reverse=True
        )

        results = []

        for document, score in ranked[:top_k]:

            results.append(
                document.model_copy(
                    update={
                        "score": float(score),
                        "source": "reranker"
                    }
                )
            )

        return results


reranker = Reranker()