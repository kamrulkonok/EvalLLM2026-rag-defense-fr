from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

# from chunking import chunk_pages, save_jsonl as save_chunks_jsonl
from article_chunking import build_article_chunks, save_jsonl as save_article_chunks_jsonl
from generate import build_prompt
from ingest import load_pages, save_jsonl as save_pages_jsonl
from retrieve import BM25Retriever


# def ensure_data_ready() -> BM25Retriever:
#     pages_path = "data/processed/pages.jsonl"
#     # chunks_path = "data/processed/chunks.jsonl"
#     chunks_path = "data/processed/article_chunks.jsonl"

#     if not os.path.exists(pages_path):
#         pages = load_pages()
#         save_pages_jsonl(pages, pages_path)
#     else:
#         pages = []
#         with open(pages_path, "r", encoding="utf-8") as f:
#             for line in f:
#                 pages.append(json.loads(line))

#     if not os.path.exists(chunks_path):
#         chunks = chunk_pages(pages, chunk_size=1200, overlap=150)
#         save_chunks_jsonl(chunks, chunks_path)
#     else:
#         chunks = []
#         with open(chunks_path, "r", encoding="utf-8") as f:
#             for line in f:
#                 chunks.append(json.loads(line))

#     return BM25Retriever(chunks)

def ensure_data_ready() -> BM25Retriever:
    pages_path = "data/processed/pages.jsonl"
    article_chunks_path = "data/processed/article_chunks.jsonl"

    if not os.path.exists(pages_path):
        pages = load_pages()
        save_pages_jsonl(pages, pages_path)
    else:
        pages = []
        with open(pages_path, "r", encoding="utf-8") as f:
            for line in f:
                pages.append(json.loads(line))

    if not os.path.exists(article_chunks_path):
        article_chunks = build_article_chunks(pages)
        save_article_chunks_jsonl(article_chunks, article_chunks_path)
    else:
        article_chunks = []
        with open(article_chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                article_chunks.append(json.loads(line))

    return BM25Retriever(article_chunks)

def answer_question(question: str, retriever: BM25Retriever, top_k: int = 5) -> Dict[str, Any]:
    retrieved = retriever.search(question, k=top_k)
    prompt = build_prompt(question, retrieved, max_chunks=top_k)

    return {
        "question": question,
        "retrieved_chunks": retrieved,
        "prompt": prompt,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    retriever = ensure_data_ready()
    result = answer_question(args.question, retriever, top_k=args.top_k)

    print("\n=== QUESTION ===")
    print(result["question"])

    print("\n=== RETRIEVED CHUNKS ===")
    for i, chunk in enumerate(result["retrieved_chunks"], start=1):
        print(
            f"\n[{i}] chunk_id={chunk['chunk_id']} | doc_id={chunk['doc_id']} | page={chunk.get('page')} | score={chunk['score']:.4f}"
        )
        print(chunk["text"][:900])

    print("\n=== PROMPT TO SEND TO YOUR GENERATOR ===")
    print(result["prompt"][:5000])

    os.makedirs("outputs", exist_ok=True)
    safe_name = "_".join(result["question"].lower().split())[:80]
    output_path = f"outputs/{safe_name}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()