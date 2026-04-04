from __future__ import annotations

import re
from typing import Any, Dict, List

from retrieve import BM25Retriever, load_jsonl
from reranker import CrossEncoderReranker


FRENCH_MONTHS = (
    "janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|"
    "septembre|octobre|novembre|décembre|decembre"
)


def is_structured_query(query: str) -> bool:
    q = query.lower()

    patterns = [
        r"\barticle\s+(1er|\d+)\b",
        r"\bart\.?\s*(1er|\d+)\b",
        r"\b(premier|première|deuxième|second|seconde|troisième|quatrième|cinquième)\s+article\b",
        rf"\b\d{{1,2}}\s+(?:{FRENCH_MONTHS})\s+\d{{4}}\b",
        r"\bnor\b",
        r"\bminarm\b",
        r"\bherses\b",
    ]

    return any(re.search(pattern, q) for pattern in patterns)


class GatedRetriever:
    def __init__(
        self,
        bm25: BM25Retriever,
        reranker: CrossEncoderReranker,
    ) -> None:
        self.bm25 = bm25
        self.reranker = reranker

    def search(
        self,
        query: str,
        k: int = 5,
        bm25_k: int = 10,
        per_doc_limit: int = 10,
    ) -> List[Dict[str, Any]]:
        bm25_results = self.bm25.search(
            query,
            k=bm25_k,
            per_doc_limit=per_doc_limit,
        )

        if is_structured_query(query):
            for item in bm25_results:
                item["retrieval_mode"] = "bm25_only"
            return bm25_results[:k]

        reranked = self.reranker.rerank(query, bm25_results, top_k=bm25_k)
        for item in reranked:
            item["retrieval_mode"] = "bm25_reranked"

        return reranked[:k]


def main() -> None:
    chunks = load_jsonl("data/processed/article_chunks.jsonl")
    bm25 = BM25Retriever(chunks)
    reranker = CrossEncoderReranker(
        model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    )
    retriever = GatedRetriever(bm25, reranker)

    test_queries = [
        "Que dit l’article 2 de l’arrêté du 16 janvier 2023 ?",
        "Dans quelles conditions les militaires peuvent-ils immobiliser les véhicules routiers automobiles ?",
        "Qui fixe les conditions et les limites de l’emploi de chaque type de matériel ?",
        "Quel article dit que le présent arrêté sera publié au Journal officiel de la République française ?",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print(f"Structured: {is_structured_query(query)}\n")

        results = retriever.search(query, k=5)

        for i, result in enumerate(results, start=1):
            print(
                f"Result {i} | chunk_id={result['chunk_id']} | "
                f"doc_id={result['doc_id']} | article={result.get('article')} | "
                f"mode={result.get('retrieval_mode')} | "
                f"bm25_score={result.get('score', 0):.4f} | "
                f"rerank_score={result.get('rerank_score', 0):.4f}"
            )
            print(result["text"][:700])
            print("-" * 80)


if __name__ == "__main__":
    main()