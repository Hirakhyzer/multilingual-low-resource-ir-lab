"""Markdown report generation for multilingual IR experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd


def write_report(path: str | Path, summary: dict[str, Any], metrics: pd.DataFrame, fairness: pd.DataFrame, answers: pd.DataFrame) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Multilingual Low-Resource IR Lab Report",
        "",
        "> Synthetic research warning: all documents, queries, labels, and answers are fictional. Results are for reproducible method comparison only.",
        "",
        "## Summary",
        "",
        f"- Documents: `{summary['document_count']}`",
        f"- Queries: `{summary['query_count']}`",
        f"- Mean Top-3 accuracy: `{summary['mean_top3_accuracy']:.3f}`",
        f"- Mean reciprocal rank: `{summary['mean_reciprocal_rank']:.3f}`",
        f"- Mean citation coverage: `{summary['mean_citation_coverage']:.3f}`",
        f"- Hallucination-risk rate: `{summary['hallucination_risk_rate']:.3f}`",
        "",
        "## Retrieval metrics by method and OCR noise",
        "",
        metrics.to_markdown(index=False),
        "",
        "## Language fairness audit",
        "",
        fairness.to_markdown(index=False),
        "",
        "## Example answers",
        "",
        "| Query | Status | Answer |",
        "| --- | --- | --- |",
    ]
    for row in answers.head(8).itertuples(index=False):
        safe_answer = str(row.answer).replace("|", " ")
        lines.append(f"| {row.query_id} | {row.status} | {safe_answer} |")
    lines.extend([
        "",
        "## Research boundary",
        "",
        "The lexical and hybrid baselines are offline experiments. They are not production search quality claims, not a complete translation system, and not an OCR benchmark.",
    ])
    destination.write_text("\n".join(lines), encoding="utf-8")
