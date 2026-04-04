from __future__ import annotations

from typing import Any, Dict, List

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(
        self,
        # model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    ) -> None:
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int | None = None,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        pairs = []
        for item in candidates:
            text = item.get("index_text", item.get("text", ""))
            pairs.append((query, text))

        scores = self.model.predict(pairs)

        reranked = []
        for item, score in zip(candidates, scores):
            chunk = dict(item)
            chunk["rerank_score"] = float(score)
            reranked.append(chunk)

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

        if top_k is not None:
            reranked = reranked[:top_k]

        return reranked