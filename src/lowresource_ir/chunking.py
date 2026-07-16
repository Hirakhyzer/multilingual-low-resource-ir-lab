"""Citation-ready document chunking."""

from __future__ import annotations

import re
import pandas as pd

SENTENCE_SPLIT = re.compile(r"(?<=[\.؟?])\s+|\n+")


def chunk_documents(documents: pd.DataFrame, max_sentences: int = 2) -> pd.DataFrame:
    """Split documents into stable, citation-ready chunks."""
    rows: list[dict] = []
    for doc in documents.itertuples(index=False):
        sentences = [s.strip() for s in SENTENCE_SPLIT.split(str(doc.content)) if s.strip()]
        if not sentences:
            sentences = [str(doc.content)]
        for i in range(0, len(sentences), max_sentences):
            text = " ".join(sentences[i:i + max_sentences])
            chunk_index = i // max_sentences
            rows.append({
                "chunk_id": f"{doc.doc_id}::C{chunk_index:02d}",
                "doc_id": doc.doc_id,
                "language": doc.language,
                "script": doc.script,
                "topic": doc.topic,
                "title": doc.title,
                "chunk_text": text,
            })
    return pd.DataFrame(rows)
