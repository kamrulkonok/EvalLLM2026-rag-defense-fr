from __future__ import annotations

import json
import os
from typing import Any, Dict, List


def load_jsonl(input_path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def save_jsonl(rows: List[Dict[str, Any]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def chunk_text_by_chars(text: str, chunk_size: int = 1200, overlap: int = 150) -> List[Dict[str, Any]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                {
                    "text": chunk_text,
                    "start_char": start,
                    "end_char": end,
                }
            )
        if end == n:
            break
        start += chunk_size - overlap

    return chunks


def chunk_pages(
    pages: List[Dict[str, Any]],
    chunk_size: int = 1200,
    overlap: int = 150,
) -> List[Dict[str, Any]]:
    all_chunks: List[Dict[str, Any]] = []

    for page in pages:
        text = page["text"]

        # short pages stay as single chunk
        if len(text) <= chunk_size:
            page_chunks = [
                {
                    "text": text,
                    "start_char": 0,
                    "end_char": len(text),
                }
            ]
        else:
            page_chunks = chunk_text_by_chars(text, chunk_size=chunk_size, overlap=overlap)

        for idx, chunk in enumerate(page_chunks):
            all_chunks.append(
                {
                    "chunk_id": f"{page['page_id']}_chunk_{idx}",
                    "doc_id": page["doc_id"],
                    "doc_name": page["doc_name"],
                    "title": page["title"],
                    "page": page["page"],
                    "page_id": page["page_id"],
                    "source": page["source"],
                    "split": page["split"],
                    "text": chunk["text"],
                    "start_char": chunk["start_char"],
                    "end_char": chunk["end_char"],
                }
            )

    return all_chunks


if __name__ == "__main__":
    pages = load_jsonl("data/processed/pages.jsonl")
    chunks = chunk_pages(pages, chunk_size=1200, overlap=150)

    print(f"Created {len(chunks)} chunks.")
    print(json.dumps(chunks[0], ensure_ascii=False, indent=2)[:1500])

    save_jsonl(chunks, "data/processed/chunks.jsonl")
    print("Saved to data/processed/chunks.jsonl")