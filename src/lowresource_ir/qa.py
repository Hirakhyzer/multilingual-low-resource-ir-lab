"""Citation-grounded extractive QA for retrieved multilingual chunks."""

from __future__ import annotations

import re
import pandas as pd

from lowresource_ir.normalization import normalize_text, tokenize

SENTENCE_SPLIT = re.compile(r"(?<=[\.؟?])\s+|\n+")


def answer_from_retrieval(query: str, retrieved: pd.DataFrame, min_score: float = 0.05) -> dict:
    """Create an answer only from retrieved text and cite every source chunk."""
    if retrieved.empty or float(retrieved.iloc[0]["score"]) <= min_score:
        return {"status": "unsupported", "answer": "I don't know from the available multilingual documents.", "citations": []}
    q_tokens = set(tokenize(query))
    candidate_sentences: list[tuple[float, str, str]] = []
    for row in retrieved.itertuples(index=False):
        sentences = [s.strip() for s in SENTENCE_SPLIT.split(str(row.chunk_text)) if s.strip()]
        for sentence in sentences:
            s_tokens = set(tokenize(sentence))
            overlap = len(q_tokens.intersection(s_tokens)) / max(len(q_tokens), 1)
            score = float(getattr(row, "score")) + overlap
            candidate_sentences.append((score, sentence, row.chunk_id))
    if not candidate_sentences:
        return {"status": "unsupported", "answer": "I don't know from the available multilingual documents.", "citations": []}
    candidate_sentences.sort(key=lambda item: item[0], reverse=True)
    selected = candidate_sentences[:2]
    citations = sorted({chunk_id for _, _, chunk_id in selected})
    answer = " ".join(sentence for _, sentence, _ in selected)
    answer += " " + " ".join(f"[{citation}]" for citation in citations)
    return {"status": "answered", "answer": answer, "citations": citations}


def citation_coverage(answer: str, citations: list[str]) -> float:
    if not answer.strip():
        return 0.0
    return 1.0 if citations and all(f"[{citation}]" in answer for citation in citations) else 0.0


def unsupported_claim_score(answer: str, citations: list[str], retrieved: pd.DataFrame) -> float:
    """Conservative lexical unsupported-claim heuristic."""
    if not citations:
        return 1.0 if "don't know" not in answer.lower() else 0.0
    retrieved_text = normalize_text(" ".join(retrieved["chunk_text"].tolist()))
    answer_tokens = [t for t in tokenize(answer) if not t.startswith("[")]
    if not answer_tokens:
        return 0.0
    unsupported = [t for t in answer_tokens if len(t) > 3 and t not in retrieved_text]
    return min(1.0, len(unsupported) / max(len(answer_tokens), 1))
