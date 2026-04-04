from __future__ import annotations

from typing import Dict, List


def build_context(chunks: List[Dict], max_chunks: int = 5) -> str:
    selected = chunks[:max_chunks]
    parts = []

    for i, chunk in enumerate(selected, start=1):
        header = (
            f"[Source {i}] "
            f"doc_id={chunk['doc_id']} | "
            f"page={chunk.get('page')} | "
            f"chunk_id={chunk['chunk_id']}"
        )
        parts.append(f"{header}\n{chunk['text']}")

    return "\n\n".join(parts)


def build_prompt(question: str, chunks: List[Dict], max_chunks: int = 5) -> str:
    context = build_context(chunks, max_chunks=max_chunks)

    return f"""Tu es un assistant spécialisé en réponse fondée sur des documents.

Réponds à la question en utilisant uniquement le contexte fourni.
Si la réponse n'est pas présente dans le contexte, dis clairement que l'information n'est pas trouvée dans les documents fournis.
Quand c'est possible, cite la source sous la forme doc_id et page.
N'invente rien.

Question :
{question}

Contexte :
{context}

Réponse :
"""