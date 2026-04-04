from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


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


# Stricter legal article header detection.
# Matches things like:
#   Art. 1er. –
#   Art. 2. -
#   Article 3
#   ARTICLE 4 -
#
# Avoids matching prose mentions like:
#   "selon l'article 8"
#   "article 8, que ..."
#   "article L. 2338-3"
ARTICLE_HEADER_RE = re.compile(
    r"(?im)^\s*(?:Art\.?\s*(1er|\d+)\s*[.\-–]|Article\s+(1er|\d+)\b(?:\s*[.\-–]|$))"
)

TITLE_RE = re.compile(
    r"(?im)^(Arrêté du .+|Décret du .+|Ordonnance du .+|Loi n° .+)$"
)

NOR_RE = re.compile(r"\bNOR\s*:?\s*([A-Z0-9]+)\b")
MINISTRY_RE = re.compile(r"(?m)^MINISTÈRE.+$")

DATE_RE = re.compile(
    r"\b(\d{1,2}\s+"
    r"(janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|"
    r"septembre|octobre|novembre|décembre|decembre)"
    r"\s+\d{4})\b",
    flags=re.IGNORECASE,
)


def normalize_whitespace(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()


def group_pages_by_doc(pages: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    docs: Dict[str, List[Dict[str, Any]]] = {}
    for page in pages:
        docs.setdefault(page["doc_id"], []).append(page)

    for doc_id in docs:
        docs[doc_id].sort(key=lambda x: x["page"])
    return docs


def extract_title(text: str, fallback: str) -> str:
    m = TITLE_RE.search(text)
    if m:
        return m.group(1).strip()
    return fallback


def extract_nor(text: str) -> Optional[str]:
    m = NOR_RE.search(text)
    return m.group(1) if m else None


def extract_ministry(text: str) -> Optional[str]:
    m = MINISTRY_RE.search(text)
    return m.group(0).strip() if m else None


def extract_date(text: str) -> Optional[str]:
    m = DATE_RE.search(text)
    return m.group(1) if m else None


def infer_document_type(title: str) -> str:
    title_l = title.lower()
    if "arrêté" in title_l or "arrete" in title_l:
        return "arrete"
    if "décret" in title_l or "decret" in title_l:
        return "decret"
    if "ordonnance" in title_l:
        return "ordonnance"
    if "loi" in title_l:
        return "loi"
    return ""


def build_index_text(chunk: Dict[str, Any]) -> str:
    parts = [
        f"Titre: {chunk.get('title', '')}",
        f"Type: {chunk.get('document_type', '')}",
        f"Date: {chunk.get('date', '')}",
        f"Article: {chunk.get('article', '')}",
        f"NOR: {chunk.get('nor', '')}",
        f"Ministère: {chunk.get('ministry', '')}",
        "",
        chunk.get("text", ""),
    ]
    return "\n".join(p for p in parts if p is not None)


def is_likely_legal_doc(full_text: str, title: str) -> bool:
    """
    Restrict article extraction to documents that look like real legal/normative texts.
    This helps avoid false positives in reports, manuals, academic PDFs, etc.
    """
    text_start = full_text[:4000].lower()
    title_l = title.lower()

    legal_signals = [
        "arrêté",
        "arrete",
        "décret",
        "decret",
        "loi",
        "ordonnance",
        "journal officiel",
        "arrêtent",
        "décrète",
        "décretent",
        "texte général",
        "textes généraux",
        "nor :",
    ]

    return any(sig in text_start or sig in title_l for sig in legal_signals)


def looks_like_real_article_chunk(text: str) -> bool:
    """
    Filter out false positives.
    A real chunk should begin with a true article header and be long enough
    to contain actual legal content.
    """
    stripped = text.strip()

    if not re.match(r"(?im)^\s*(Art\.?|Article)\s*(1er|\d+)\b", stripped):
        return False

    if len(stripped) < 40:
        return False

    return True


def split_articles(full_text: str) -> List[Dict[str, str]]:
    matches = list(ARTICLE_HEADER_RE.finditer(full_text))
    if not matches:
        return []

    articles = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        article_text = full_text[start:end].strip()
        article_id = (match.group(1) or match.group(2)).strip().lower()

        articles.append(
            {
                "article": article_id,
                "text": article_text,
            }
        )

    return articles


def build_article_chunks(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    docs = group_pages_by_doc(pages)
    output: List[Dict[str, Any]] = []

    for doc_id, doc_pages in docs.items():
        full_text = normalize_whitespace("\n".join(page["text"] for page in doc_pages))
        fallback_title = doc_pages[0]["doc_name"]

        title = extract_title(full_text, fallback=fallback_title)
        nor = extract_nor(full_text)
        ministry = extract_ministry(full_text)
        date = extract_date(title) or extract_date(full_text)
        document_type = infer_document_type(title)

        if not is_likely_legal_doc(full_text, title):
            continue

        articles = split_articles(full_text)
        if not articles:
            continue

        for item in articles:
            article_id = item["article"]
            text = item["text"]

            if not looks_like_real_article_chunk(text):
                continue

            row = {
                "chunk_id": f"{Path(doc_id).stem}_article_{article_id}",
                "doc_id": doc_id,
                "doc_name": doc_id,
                "title": title,
                "document_type": document_type,
                "date": date,
                "nor": nor,
                "ministry": ministry,
                "article": article_id,
                "page_start": doc_pages[0]["page"],
                "page_end": doc_pages[-1]["page"],
                "source": doc_pages[0]["source"],
                "split": doc_pages[0]["split"],
                "text": text,
            }
            row["index_text"] = build_index_text(row)
            output.append(row)

    return output


if __name__ == "__main__":
    pages = load_jsonl("data/processed/pages.jsonl")
    article_chunks = build_article_chunks(pages)

    print(f"Created {len(article_chunks)} article chunks.")
    if article_chunks:
        print(json.dumps(article_chunks[0], ensure_ascii=False, indent=2)[:2000])

    save_jsonl(article_chunks, "data/processed/article_chunks.jsonl")
    print("Saved to data/processed/article_chunks.jsonl")