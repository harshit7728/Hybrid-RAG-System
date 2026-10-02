import re


def route_query(query: str) -> str:

    query_lower = query.lower().strip()

    # Strong keyword/entity signals
    keyword_patterns = [
        r"\b[A-Z]{2,}-\d+\b",
        r"\b\d{4,}\b",
        r"\bexact\b",
        r"\bcode\b",
        r"\bid\b",
        r"\btransaction\b",
        r"\border\b",
    ]

    for pattern in keyword_patterns:
        if re.search(pattern, query):
            return "bm25"

    # Semantic/explanation signals
    semantic_patterns = [
        "explain",
        "why",
        "how does",
        "difference between",
        "describe",
        "architecture",
        "concept",
        "meaning",
    ]

    for pattern in semantic_patterns:
        if pattern in query_lower:
            return "vector"

    return "hybrid"