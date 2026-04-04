# from __future__ import annotations

# import json
# import os
# import pickle
# import re
# from typing import Any, Dict, List

# from rank_bm25 import BM25Okapi


# def load_jsonl(input_path: str) -> List[Dict[str, Any]]:
#     rows = []
#     with open(input_path, "r", encoding="utf-8") as f:
#         for line in f:
#             rows.append(json.loads(line))
#     return rows


# def simple_tokenize(text: str) -> List[str]:
#     text = text.lower()
#     return re.findall(r"[a-zàâçéèêëîïôûùüÿñæœ0-9]+", text, flags=re.IGNORECASE)


# def normalize_text(text: str) -> str:
#     return " ".join(simple_tokenize(text))


# class BM25Retriever:
#     def __init__(self, chunks: List[Dict[str, Any]]) -> None:
#         self.chunks = chunks
#         # self.tokenized_corpus = [simple_tokenize(chunk["text"]) for chunk in chunks]
#         self.tokenized_corpus = [
#             simple_tokenize(chunk.get("index_text", chunk["text"]))
#             for chunk in chunks
# ]
#         self.bm25 = BM25Okapi(self.tokenized_corpus)

#     def _metadata_boost(self, query_tokens: List[str], chunk: Dict[str, Any]) -> float:
#         title_tokens = set(simple_tokenize(chunk.get("title", "")))
#         doc_tokens = set(simple_tokenize(chunk.get("doc_name", "")))
#         query_set = set(query_tokens)

#         overlap = len(query_set & (title_tokens | doc_tokens))
#         return 2.0 * overlap

#     def _phrase_boost(self, query: str, chunk: Dict[str, Any]) -> float:
#         score = 0.0
#         q_norm = normalize_text(query)
#         text_norm = normalize_text(chunk.get("index_text", chunk["text"]))
#         title_norm = normalize_text(chunk.get("title", ""))
        

#         # exact normalized query contained
#         if len(q_norm) > 15 and q_norm in text_norm:
#             score += 8.0
#         if len(q_norm) > 15 and q_norm in title_norm:
#             score += 10.0

#         # article number match: "article 2", "art 2"
#         article_match = re.search(r"\barticle\s+(\d+|[a-z]\d+)\b", query.lower())
#         if article_match:
#             article_id = article_match.group(1)
#             if re.search(rf"\b(art|article)\.?\s*{re.escape(article_id)}\b", chunk["text"].lower()):
#                 score += 8.0

#         # french date patterns like "16 janvier 2023"
#         date_match = re.search(
#             r"\b(\d{1,2}\s+(janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre)\s+\d{4})\b",
#             query.lower(),
#         )
#         if date_match:
#             date_str = date_match.group(1)
#             if date_str in chunk["text"].lower() or date_str in chunk.get("title", "").lower():
#                 score += 10.0

#         return score

#     def _noise_penalty(self, chunk: Dict[str, Any]) -> float:
#         """
#         Mild penalty for clearly non-French/non-domain pages that often act as noise.
#         """
#         text = chunk["text"].lower()
#         title = chunk.get("title", "").lower()

#         penalty = 0.0

#         noisy_patterns = [
#             "panasonic",
#             "estech",
#             "smdr",
#             "north america",
#             "call accounting",
#         ]
#         if any(p in text or p in title for p in noisy_patterns):
#             penalty -= 6.0

#         return penalty

#     def search(self, query: str, k: int = 5, per_doc_limit: int = 2) -> List[Dict[str, Any]]:
#         query_tokens = simple_tokenize(query)
#         raw_scores = self.bm25.get_scores(query_tokens)

#         rescored = []
#         for idx, score in enumerate(raw_scores):
#             chunk = self.chunks[idx]
#             boosted = (
#                 float(score)
#                 + self._metadata_boost(query_tokens, chunk)
#                 + self._phrase_boost(query, chunk)
#                 + self._noise_penalty(chunk)
#             )
#             rescored.append((idx, boosted))

#         rescored.sort(key=lambda x: x[1], reverse=True)

#         results = []
#         doc_counts: Dict[str, int] = {}

#         for idx, score in rescored:
#             chunk = dict(self.chunks[idx])
#             doc_id = chunk["doc_id"]

#             if doc_counts.get(doc_id, 0) >= per_doc_limit:
#                 continue

#             chunk["score"] = float(score)
#             results.append(chunk)
#             doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1

#             if len(results) >= k:
#                 break

#         return results

#     def save(self, output_dir: str) -> None:
#         os.makedirs(output_dir, exist_ok=True)

#         with open(os.path.join(output_dir, "chunks.json"), "w", encoding="utf-8") as f:
#             json.dump(self.chunks, f, ensure_ascii=False)

#         with open(os.path.join(output_dir, "bm25.pkl"), "wb") as f:
#             pickle.dump(self.bm25, f)

#     @classmethod
#     def load(cls, output_dir: str) -> "BM25Retriever":
#         with open(os.path.join(output_dir, "chunks.json"), "r", encoding="utf-8") as f:
#             chunks = json.load(f)

#         retriever = cls(chunks)

#         with open(os.path.join(output_dir, "bm25.pkl"), "rb") as f:
#             retriever.bm25 = pickle.load(f)

#         return retriever


# if __name__ == "__main__":
#     chunks = load_jsonl("data/processed/article_chunks.jsonl")
    
#     retriever = BM25Retriever(chunks)

#     test_queries = [
#         "Que dit l’article 2 de l’arrêté du 16 janvier 2023 ?",
#         "Dans quelles conditions les militaires peuvent-ils immobiliser les véhicules routiers automobiles ?",
#         "Qui fixe les conditions et les limites de l’emploi de chaque type de matériel ?",
#     ]

#     for query in test_queries:
#         print(f"\nQuery: {query}\n")
#         results = retriever.search(query, k=5)
#         for i, result in enumerate(results, start=1):
#             print(
#                 f"Result {i} | chunk_id={result['chunk_id']} | doc_id={result['doc_id']} | page={result.get('page')} | score={result['score']:.4f}"
#             )
#             print(result["text"][:700])
#             print("-" * 80)

#     retriever.save("data/processed/bm25_index")
#     print("Saved BM25 index to data/processed/bm25_index")


from __future__ import annotations

import json
import os
import pickle
import re
from typing import Any, Dict, List, Optional

from rank_bm25 import BM25Okapi


def load_jsonl(input_path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def simple_tokenize(text: str) -> List[str]:
    text = text.lower()
    return re.findall(r"[a-zàâçéèêëîïôûùüÿñæœ0-9]+", text, flags=re.IGNORECASE)


def normalize_text(text: str) -> str:
    return " ".join(simple_tokenize(text))


def extract_article_from_query(query: str) -> Optional[str]:
    q = query.lower()

    # direct forms
    m = re.search(r"\barticle\s+(1er|\d+)\b", q)
    if m:
        return m.group(1)

    m = re.search(r"\bart\.?\s*(1er|\d+)\b", q)
    if m:
        return m.group(1)

    # ordinal forms
    ordinal_map = {
        "premier": "1er",
        "première": "1er",
        "deuxième": "2",
        "second": "2",
        "seconde": "2",
        "troisième": "3",
        "quatrième": "4",
        "cinquième": "5",
    }

    for word, value in ordinal_map.items():
        if re.search(rf"\b{word}\s+article\b", q):
            return value
        if re.search(rf"\barticle\s+{word}\b", q):
            return value

    return None


def extract_date_from_query(query: str) -> Optional[str]:
    m = re.search(
        r"\b(\d{1,2}\s+(janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre)\s+\d{4})\b",
        query.lower(),
    )
    return m.group(1) if m else None


class BM25Retriever:
    def __init__(self, chunks: List[Dict[str, Any]]) -> None:
        self.chunks = chunks
        self.tokenized_corpus = [
            simple_tokenize(chunk.get("index_text", chunk["text"]))
            for chunk in chunks
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _metadata_boost(self, query_tokens: List[str], query: str, chunk: Dict[str, Any]) -> float:
        title_tokens = set(simple_tokenize(chunk.get("title", "")))
        doc_tokens = set(simple_tokenize(chunk.get("doc_name", "")))
        query_set = set(query_tokens)

        score = 0.0

        overlap = len(query_set & (title_tokens | doc_tokens))
        score += 2.0 * overlap

        # strong article metadata match
        query_article = extract_article_from_query(query)
        chunk_article = str(chunk.get("article", "")).strip().lower()

        if query_article and chunk_article:
            qa = query_article.lower()
            if qa == chunk_article:
                score += 20.0
            elif qa in {"1", "1er"} and chunk_article in {"1", "1er"}:
                score += 20.0
            else:
                score -= 6.0

        # date match
        query_date = extract_date_from_query(query)
        title = chunk.get("title", "").lower()
        text = chunk.get("text", "").lower()
        if query_date:
            if query_date in title or query_date in text:
                score += 10.0

        # shorthand aliases from title/doc name
        doc_blob = f"{chunk.get('title', '')} {chunk.get('doc_name', '')}".lower()
        alias_terms = [tok for tok in query_tokens if tok not in {"article", "arrêté", "arrete", "du", "de", "des"}]
        alias_overlap = sum(1 for tok in alias_terms if tok in doc_blob)
        score += 3.0 * alias_overlap

        return score

    def _phrase_boost(self, query: str, chunk: Dict[str, Any]) -> float:
        score = 0.0
        q_norm = normalize_text(query)
        text_norm = normalize_text(chunk.get("index_text", chunk["text"]))
        title_norm = normalize_text(chunk.get("title", ""))

        if len(q_norm) > 15 and q_norm in text_norm:
            score += 8.0
        if len(q_norm) > 15 and q_norm in title_norm:
            score += 10.0

        query_article = extract_article_from_query(query)
        if query_article:
            chunk_article = str(chunk.get("article", "")).strip().lower()
            if query_article == chunk_article or (
                query_article in {"1", "1er"} and chunk_article in {"1", "1er"}
            ):
                score += 10.0

        chunk_text_lower = chunk.get("text", "").lower()
        query_lower = query.lower()

        if "journal officiel de la république française" in query_lower:
            if "journal officiel de la république française" in chunk_text_lower:
                score += 8.0

        if (
            "publié au journal officiel" in query_lower
            or "publié au journal officiel de la république française" in query_lower
            or "sera publié au journal officiel" in query_lower
        ):
            if "sera publié au journal officiel de la république française" in chunk_text_lower:
                score += 20.0
            elif "publié au journal officiel de la république française" in chunk_text_lower:
                score += 15.0

        return score

    def _noise_penalty(self, chunk: Dict[str, Any]) -> float:
        text = chunk.get("text", "").lower()
        title = chunk.get("title", "").lower()
        doc_name = chunk.get("doc_name", "").lower()

        penalty = 0.0

        noisy_patterns = [
            "panasonic",
            "estech",
            "smdr",
            "north america",
            "call accounting",
            "manuel",
            "rapport",
            "concept d'emploi",
            "ballons",
        ]
        if any(p in text or p in title or p in doc_name for p in noisy_patterns):
            penalty -= 6.0

        return penalty

    def search(self, query: str, k: int = 5, per_doc_limit: int = 2) -> List[Dict[str, Any]]:
        query_tokens = simple_tokenize(query)
        raw_scores = self.bm25.get_scores(query_tokens)

        rescored = []
        for idx, score in enumerate(raw_scores):
            chunk = self.chunks[idx]
            boosted = (
                float(score)
                + self._metadata_boost(query_tokens, query, chunk)
                + self._phrase_boost(query, chunk)
                + self._noise_penalty(chunk)
            )
            rescored.append((idx, boosted))

        rescored.sort(key=lambda x: x[1], reverse=True)

        results = []
        doc_counts: Dict[str, int] = {}

        for idx, score in rescored:
            chunk = dict(self.chunks[idx])
            doc_id = chunk["doc_id"]

            if doc_counts.get(doc_id, 0) >= per_doc_limit:
                continue

            chunk["score"] = float(score)
            results.append(chunk)
            doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1

            if len(results) >= k:
                break

        return results

    def save(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False)

        with open(os.path.join(output_dir, "bm25.pkl"), "wb") as f:
            pickle.dump(self.bm25, f)

    @classmethod
    def load(cls, output_dir: str) -> "BM25Retriever":
        with open(os.path.join(output_dir, "chunks.json"), "r", encoding="utf-8") as f:
            chunks = json.load(f)

        retriever = cls(chunks)

        with open(os.path.join(output_dir, "bm25.pkl"), "rb") as f:
            retriever.bm25 = pickle.load(f)

        return retriever
    

if __name__ == "__main__":
    chunks = load_jsonl("data/processed/article_chunks.jsonl")
    retriever = BM25Retriever(chunks)