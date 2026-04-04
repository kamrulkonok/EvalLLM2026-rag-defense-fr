from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from dense_retrieve import DenseRetriever
from retrieve import BM25Retriever, load_jsonl


def reciprocal_rank_fusion(
    result_lists: List[List[Dict[str, Any]]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    """
    Combine ranked lists using Reciprocal Rank Fusion (RRF).

    score(doc) = sum(1 / (k + rank_i))
    """
    fused_scores = defaultdict(float)
    chunk_map: Dict[str, Dict[str, Any]] = {}

    for results in result_lists:
        for rank, item in enumerate(results, start=1):
            chunk_id = item["chunk_id"]
            fused_scores[chunk_id] += 1.0 / (k + rank)
            chunk_map[chunk_id] = item

    ranked = sorted(
        fused_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    fused_results = []
    for chunk_id, score in ranked:
        chunk = dict(chunk_map[chunk_id])
        chunk["score"] = float(score)
        fused_results.append(chunk)

    return fused_results


class HybridRetriever:
    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
    ) -> None:
        self.bm25 = bm25_retriever
        self.dense = dense_retriever

    def search(
        self,
        query: str,
        k: int = 5,
        bm25_k: int = 10,
        dense_k: int = 10,
        per_doc_limit: int = 2,
    ) -> List[Dict[str, Any]]:
        bm25_results = self.bm25.search(
            query,
            k=bm25_k,
            per_doc_limit=per_doc_limit,
        )
        dense_results = self.dense.search(
            query,
            k=dense_k,
            per_doc_limit=per_doc_limit,
        )

        fused = reciprocal_rank_fusion([bm25_results, dense_results])

        final_results = []
        doc_counts: Dict[str, int] = {}

        for chunk in fused:
            doc_id = chunk["doc_id"]

            if doc_counts.get(doc_id, 0) >= per_doc_limit:
                continue

            final_results.append(chunk)
            doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1

            if len(final_results) >= k:
                break

        return final_results


def main() -> None:
    chunks = load_jsonl("data/processed/article_chunks.jsonl")

    bm25 = BM25Retriever(chunks)
    dense = DenseRetriever.load("data/processed/dense_index")
    retriever = HybridRetriever(bm25, dense)

    test_queries = [
        "Que dit l’article 2 de l’arrêté du 16 janvier 2023 ?",
        "Dans quelles conditions les militaires peuvent-ils immobiliser les véhicules routiers automobiles ?",
        "Qui fixe les conditions et les limites de l’emploi de chaque type de matériel ?",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}\n")
        results = retriever.search(query, k=5)

        for i, result in enumerate(results, start=1):
            print(
                f"Result {i} | chunk_id={result['chunk_id']} | "
                f"doc_id={result['doc_id']} | article={result.get('article')} | "
                f"score={result['score']:.4f}"
            )
            print(result["text"][:700])
            print("-" * 80)


if __name__ == "__main__":
    main()