"""Multilingual embedding extension point.

The runnable lab is fully offline and uses lexical baselines. This module defines
an explicit adapter contract for future multilingual embedding models without
claiming results that have not been executed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingModelSpec:
    model_name: str
    source: str
    expected_languages: tuple[str, ...]
    notes: str = "Requires separate model download, license review, and execution log."


DEFAULT_FUTURE_SPECS = [
    EmbeddingModelSpec("multilingual-e5", "future_adapter", ("urdu", "arabic", "english", "roman_urdu")),
    EmbeddingModelSpec("LaBSE", "future_adapter", ("urdu", "arabic", "english")),
    EmbeddingModelSpec("sentence-transformers multilingual MiniLM", "future_adapter", ("urdu", "arabic", "english")),
]


def planned_embedding_models() -> list[dict[str, object]]:
    """Return documented future-model comparison candidates."""
    return [spec.__dict__ for spec in DEFAULT_FUTURE_SPECS]
