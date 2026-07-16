"""Visualization utilities for multilingual IR experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _path(path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def plot_retrieval_comparison(metrics: pd.DataFrame, path: str | Path) -> None:
    pivot = metrics.pivot_table(index="noise_rate", columns="method", values="top3_accuracy", aggfunc="mean")
    figure, axis = plt.subplots(figsize=(9, 5))
    pivot.plot(marker="o", ax=axis)
    axis.set(xlabel="OCR noise rate", ylabel="Top-3 accuracy", title="Multilingual retrieval robustness under OCR noise", ylim=(0, 1.05))
    axis.grid(True, alpha=0.25)
    figure.tight_layout(); figure.savefig(_path(path), dpi=250); plt.close(figure)


def plot_language_fairness(fairness: pd.DataFrame, path: str | Path) -> None:
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    axis.bar(fairness["language"], fairness["top3_recall"])
    axis.set(xlabel="Expected document language", ylabel="Top-3 recall", title="Language-level retrieval fairness audit", ylim=(0, 1.05))
    axis.grid(True, axis="y", alpha=0.25)
    figure.tight_layout(); figure.savefig(_path(path), dpi=250); plt.close(figure)


def plot_query_status(answers: pd.DataFrame, path: str | Path) -> None:
    counts = answers["status"].value_counts()
    figure, axis = plt.subplots(figsize=(6.5, 4.6))
    axis.bar(counts.index, counts.values)
    axis.set(xlabel="QA status", ylabel="Query count", title="Citation-grounded QA outcomes")
    axis.grid(True, axis="y", alpha=0.25)
    figure.tight_layout(); figure.savefig(_path(path), dpi=250); plt.close(figure)


def plot_citation_grounding(answers: pd.DataFrame, path: str | Path) -> None:
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    axis.scatter(answers["citation_coverage"], answers["unsupported_claim_score"], s=60)
    axis.set(xlabel="Citation coverage", ylabel="Unsupported-claim score", title="QA grounding and hallucination-risk audit", xlim=(-0.05, 1.05), ylim=(-0.05, 1.05))
    axis.grid(True, alpha=0.25)
    figure.tight_layout(); figure.savefig(_path(path), dpi=250); plt.close(figure)
