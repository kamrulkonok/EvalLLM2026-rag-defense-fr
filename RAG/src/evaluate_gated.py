from __future__ import annotations

import json
from typing import Any, Dict, List

from gated_retriever import GatedRetriever
from reranker import CrossEncoderReranker
from retrieve import BM25Retriever, load_jsonl


def load_eval_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def normalize_article(article: Any) -> str:
    if article is None:
        return ""
    article = str(article).strip().lower()
    if article == "1":
        return "1er"
    return article


def is_correct(result: Dict[str, Any] | None, gold_doc_id: str, gold_article: str) -> bool:
    if result is None:
        return False

    pred_doc_id = result.get("doc_id", "")
    pred_article = normalize_article(result.get("article"))
    gold_article = normalize_article(gold_article)

    return pred_doc_id == gold_doc_id and pred_article == gold_article


def main() -> None:
    chunks = load_jsonl("data/processed/article_chunks.jsonl")
    eval_rows = load_eval_jsonl("data/processed/eval_questions.jsonl")

    bm25 = BM25Retriever(chunks)
    reranker = CrossEncoderReranker(
        model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    )
    retriever = GatedRetriever(bm25, reranker)

    correct_at_1 = 0
    total = len(eval_rows)

    for row in eval_rows:
        query = row["query"]
        gold_doc_id = row["gold_doc_id"]
        gold_article = row["gold_article"]

        results = retriever.search(query, k=1)
        top1 = results[0] if results else None
        ok = is_correct(top1, gold_doc_id, gold_article)

        if ok:
            correct_at_1 += 1

        print(f"QID: {row['qid']}")
        print(f"Query: {query}")
        print(f"Gold : {gold_doc_id} | article={gold_article}")

        if top1:
            print(
                f"Top1 : {top1['doc_id']} | article={top1.get('article')} | "
                f"mode={top1.get('retrieval_mode')} | "
                f"bm25_score={top1.get('score', 0):.4f} | "
                f"rerank_score={top1.get('rerank_score', 0):.4f}"
            )
        else:
            print("Top1 : None")

        print(f"Correct@1: {ok}")
        print("-" * 100)

    recall_at_1 = correct_at_1 / total if total else 0.0

    print(f"Total questions: {total}")
    print(f"Correct@1: {correct_at_1}")
    print(f"Recall@1: {recall_at_1:.4f}")


if __name__ == "__main__":
    main()