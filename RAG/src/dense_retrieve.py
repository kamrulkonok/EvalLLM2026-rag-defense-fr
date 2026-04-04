from __future__ import annotations

import json
import os
import pickle
from typing import Any, Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer


def load_jsonl(input_path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def cosine_similarity_matrix(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    """
    query_vec: shape (d,)
    doc_matrix: shape (n, d)
    returns: shape (n,)
    """
    query_norm = np.linalg.norm(query_vec)
    doc_norms = np.linalg.norm(doc_matrix, axis=1)

    # avoid divide-by-zero
    if query_norm == 0:
        return np.zeros(len(doc_matrix), dtype=float)

    denom = doc_norms * query_norm
    denom = np.where(denom == 0, 1e-12, denom)

    sims = np.dot(doc_matrix, query_vec) / denom
    return sims


class DenseRetriever:
    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        model_name: str = "intfloat/multilingual-e5-base",
    ) -> None:
        self.chunks = chunks
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        self.texts = [self._build_embedding_text(chunk) for chunk in chunks]
        self.embeddings: np.ndarray | None = None

    def _build_embedding_text(self, chunk: Dict[str, Any]) -> str:
        """
        Use index_text if available, otherwise raw text.
        For E5 models, prefix passages with 'passage: '.
        """
        text = chunk.get("index_text", chunk.get("text", ""))
        return f"passage: {text}"

    def build_index(self, batch_size: int = 32) -> None:
        self.embeddings = self.model.encode(
            self.texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

    def search(self, query: str, k: int = 5, per_doc_limit: int = 2) -> List[Dict[str, Any]]:
        if self.embeddings is None:
            raise ValueError("Dense index not built. Call build_index() first.")

        query_text = f"query: {query}"
        query_vec = self.model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=False,
        )[0]

        scores = cosine_similarity_matrix(query_vec, self.embeddings)
        ranked_indices = np.argsort(scores)[::-1]

        results = []
        doc_counts: Dict[str, int] = {}

        for idx in ranked_indices:
            chunk = dict(self.chunks[int(idx)])
            doc_id = chunk["doc_id"]

            if doc_counts.get(doc_id, 0) >= per_doc_limit:
                continue

            chunk["score"] = float(scores[int(idx)])
            results.append(chunk)
            doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1

            if len(results) >= k:
                break

        return results

    def save(self, output_dir: str) -> None:
        if self.embeddings is None:
            raise ValueError("No embeddings to save. Call build_index() first.")

        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False)

        with open(os.path.join(output_dir, "model_name.txt"), "w", encoding="utf-8") as f:
            f.write(self.model_name)

        with open(os.path.join(output_dir, "embeddings.pkl"), "wb") as f:
            pickle.dump(self.embeddings, f)

    @classmethod
    def load(cls, output_dir: str) -> "DenseRetriever":
        with open(os.path.join(output_dir, "chunks.json"), "r", encoding="utf-8") as f:
            chunks = json.load(f)

        with open(os.path.join(output_dir, "model_name.txt"), "r", encoding="utf-8") as f:
            model_name = f.read().strip()

        retriever = cls(chunks, model_name=model_name)

        with open(os.path.join(output_dir, "embeddings.pkl"), "rb") as f:
            retriever.embeddings = pickle.load(f)

        return retriever


if __name__ == "__main__":
    chunks = load_jsonl("data/processed/article_chunks.jsonl")

    retriever = DenseRetriever(
        chunks,
        model_name="intfloat/multilingual-e5-base",
    )
    retriever.build_index()
    retriever.save("data/processed/dense_index")

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