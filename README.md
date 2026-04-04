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


---

# Legal RAG System – Retrieval Experiments (Part 2)

## Objective

Improve retrieval performance for legal question answering by evaluating:

- Dense retrieval (semantic embeddings)
- Hybrid retrieval (BM25 + dense)
- Reranking (cross-encoder models)
- Gated retrieval (adaptive strategy)

All methods are evaluated using a fixed benchmark.

---

## Evaluation Setup

Dataset: data/processed/eval_questions.jsonl


- 20 hand-written legal questions  
- Each question includes:
  - `query`
  - `gold_doc_id`
  - `gold_article`

### Metric

Recall@1 = correct top result / total questions


A prediction is correct if:
- correct document
- correct article

---

## Experiments

### 1. Dense Retrieval

- Model: `intfloat/multilingual-e5-base`
- Uses semantic similarity instead of keyword matching

#### Result: 
A prediction is correct if:
- correct document
- correct article

---

## 🧪 Experiments

### 1. Dense Retrieval

- Model: `intfloat/multilingual-e5-base`
- Uses semantic similarity instead of keyword matching

#### Result: Recall@1 = 0.45


#### Observations:
- Good for semantic similarity
- Weak for:
  - article numbers
  - dates
  - exact legal references

---

### 2. Hybrid Retrieval (BM25 + Dense)

- Method: Reciprocal Rank Fusion (RRF)
- Combines top results from BM25 and dense retrieval

#### Result: Recall@1 = 0.65


#### Observations:
- Dense introduces noise
- BM25 signal gets diluted
- Naive fusion hurts performance

---

### 3. BM25 + Reranker (English Model)

- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`

#### Result: Recall@1 = 0.75


#### Observations:
- Slight improvement over dense
- Not suitable for French legal data

---

### 4. BM25 + Reranker (Multilingual Model)

- Model: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`

#### Result: Recall@1 = 0.85


#### Observations:
- Better than English reranker
- Still misranks nearby articles
- Does not outperform BM25 baseline

---

### 5. Gated BM25 + Reranker (Final System)

Strategy:
- If query is structured (article/date/NOR/etc):
use BM25 directly
- Else:
rerank BM25 candidates


#### Result: Recall@1 = 0.95


#### Observations:
- Matches BM25 baseline performance
- Preserves exact-match strengths
- Adds flexibility for broader queries

---

## 📈 Summary Table

| Method | Recall@1 |
|------|--------|
| BM25 (article baseline) | **0.95** |
| Dense only | 0.45 |
| Hybrid (BM25 + dense) | 0.65 |
| BM25 + English reranker | 0.75 |
| BM25 + multilingual reranker | 0.85 |
| **Gated BM25 + reranker** | **0.95** |

---

## Key Insights

### 1. Structure-aware chunking is critical
Switching from page chunks → article chunks provides the largest improvement.

---

### 2. BM25 is highly effective for legal retrieval
Legal queries rely heavily on:
- article numbers
- dates
- exact wording

---

### 3. Dense retrieval alone is insufficient
Semantic similarity does not capture:
- symbolic constraints
- exact legal references

---

### 4. Naive hybrid fusion can degrade performance
Combining retrievers without weighting introduces noise.

---

### 5. Reranker effectiveness depends on model choice
- English models underperform on French text
- Multilingual models improve results but still lag behind BM25

---

### 6. Adaptive (gated) retrieval is effective
Different query types require different retrieval strategies:
- structured queries → BM25
- semantic queries → reranking

---

## Final Retrieval Strategy

Best-performing system: Gated BM25 + multilingual reranker


However, the strongest simple baseline remains: Article-aware BM25



---

## Next Steps

- Expand evaluation set:
  - more paraphrases
  - more indirect queries
  - more document diversity
- Add answer generation (LLM)
- Explore:
  - domain-specific rerankers
  - fine-tuned embeddings

---

## Conclusion

This project demonstrates that:

> For legal question answering, retrieval performance depends more on structured, exact matching than purely semantic similarity.

Article-aware BM25 remains a strong and reliable baseline, while advanced methods must be carefully adapted to the domain.