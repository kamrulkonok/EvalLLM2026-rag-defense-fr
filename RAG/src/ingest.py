from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from datasets import load_dataset


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.strip() for line in text.splitlines())
    text = "\n".join(line for line in text.splitlines() if line.strip())
    return text.strip()


def load_pages(
    dataset_name: str = "kamrulkonok/evalllm2026-french-defense",
    split: str = "train",
) -> List[Dict[str, Any]]:
    ds = load_dataset(dataset_name)

    if split not in ds:
        raise ValueError(f"Split '{split}' not found. Available splits: {list(ds.keys())}")

    records = ds[split]
    pages: List[Dict[str, Any]] = []

    for i, item in enumerate(records):
        doc_name = item["doc_name"]
        page = int(item["page"])
        text = normalize_text(item["text"])

        if not text:
            continue

        pages.append(
            {
                "doc_id": doc_name,
                "page_id": f"{doc_name}__page_{page}",
                "title": doc_name,
                "doc_name": doc_name,
                "page": page,
                "text": text,
                "source": dataset_name,
                "split": split,
            }
        )

    pages = sorted(pages, key=lambda x: (x["doc_name"], x["page"]))
    return pages


def save_jsonl(rows: List[Dict[str, Any]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    pages = load_pages()
    print(f"Loaded {len(pages)} pages.")
    print("Sample page:")
    print(json.dumps(pages[0], ensure_ascii=False, indent=2)[:2000])

    save_jsonl(pages, "data/processed/pages.jsonl")
    print("Saved to data/processed/pages.jsonl")