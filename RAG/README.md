# What Current Version Does (Simple Explanation)

## Goal
Answer legal questions by retrieving the **correct article** from legal documents.

---

## End-to-End Flow

### 1. Ingest Data (`ingest.py`)
- Loads documents from dataset
- Cleans text
- Saves as: pages.jsonl

Each row = one page

---

### 2. Article Chunking (`article_chunking.py`)
- Combines all pages of a document
- Splits text into **articles (Art. 1, Art. 2, etc.)**
- Extracts metadata:
- title
- date
- article number
- Saves as: article_chunks.jsonl


Now:

1 chunk = 1 legal article


---

### 3. Retrieval (`retrieve.py`)
- Uses **BM25 (keyword search)** to find relevant articles
- Improves ranking using:
  - article number match (very strong boost)
  - date match
  - title/doc overlap
  - phrase matching (e.g. "journal officiel")
  - penalties for noisy documents

Input: user question

Output: top relevant article chunks


---

### 4. Prompt Building (`generate.py`)
- Formats retrieved articles into a prompt
- Ensures:
  - answers come from context
  - no hallucination
  - sources included

---

### 5. Pipeline (`pipeline.py`)
Runs everything together:
Question → Retrieve → Build prompt → Output


---

### 6. Evaluation (`evaluate.py`)
- Tests system on 20 questions
- Checks if:
  - correct document
  - correct article

Metric: Recall@1 = correct top result / total questions


---

## Results
Recall@1 = 0.95 (19/20 correct)


---

## Key Improvement

Before: Chunks = random page pieces


Now: Chunks = legal articles


This is why retrieval got much better.

---

## Remaining Issue

BM25 struggles with:
- indirect questions
- semantic matching

Example: "Quel article dit que..."


---

## Next Step

Add:
- dense retrieval (embeddings)
- hybrid search (BM25 + dense)
- reranker

---

## Commands

```bash
python src/ingest.py
python src/article_chunking.py
python src/retrieve.py
python src/evaluate.py
```