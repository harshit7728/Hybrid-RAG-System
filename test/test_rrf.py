from app.schemas.state import RetrievedDocument
from app.services.rrf import reciprocal_rank_fusion


def test_rrf():

    vector_results = [
        RetrievedDocument(
            id="A",
            content="Document A",
            score=0.9,
            source="vector"
        ),
        RetrievedDocument(
            id="B",
            content="Document B",
            score=0.8,
            source="vector"
        ),
        RetrievedDocument(
            id="C",
            content="Document C",
            score=0.7,
            source="vector"
        ),
    ]

    bm25_results = [
        RetrievedDocument(
            id="B",
            content="Document B",
            score=10,
            source="bm25"
        ),
        RetrievedDocument(
            id="D",
            content="Document D",
            score=9,
            source="bm25"
        ),
        RetrievedDocument(
            id="A",
            content="Document A",
            score=8,
            source="bm25"
        ),
    ]

    results = reciprocal_rank_fusion(
        result_lists=[
            vector_results,
            bm25_results
        ]
    )

    assert len(results) == 4

    ids = [
        document.id
        for document in results
    ]

    assert set(ids) == {
        "A",
        "B",
        "C",
        "D"
    }