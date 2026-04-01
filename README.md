# EvalLLM2026 RAG

## Overview
This repository contains the **EvalLLM 2026 RAG Challenge**, focused on retrieval-augmented generation (RAG) over a corpus of French defense documents.

---

## Dataset Usage

Load the dataset directly from Hugging Face:

```python
from datasets import load_dataset

ds = load_dataset("kamrulkonok/evalllm2026-french-defense")
print(ds)
```

## Data Format
```
{
  "doc_name": "example.pdf",
  "page": 17,
  "text": "Extracted text from the page..."
}
```

## Repository Structure
.
├── process_pdfs.py       # PDF → JSONL pipeline
├── upload_to_hf.py       # Upload dataset to Hugging Face
├── data/
│   └── pages.jsonl       # Processed dataset
└── README.md