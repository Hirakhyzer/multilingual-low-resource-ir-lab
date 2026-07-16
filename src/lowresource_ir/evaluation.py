"""Evaluation metrics for multilingual low-resource IR and QA."""

from __future__ import annotations

import pandas as pd


def retrieval_metrics(results: pd.DataFrame) -> dict[str, float | int]:
    """Compute top-k accuracy, MRR, QA coverage, and language fairness gaps."""
    expected = results.loc[results["expected_doc_id"] != "NONE"].copy()
    if expected.empty:
        return {}
    grouped = expected.groupby(["method", "noise_rate", "query_id"], dropna=False)
    rows = []
    for (method, noise_rate, query_id), group in grouped:
        target = group.iloc[0]["expected_doc_id"]
        hits = group.loc[group["doc_id"] == target].sort_values("rank")
        reciprocal = 0.0 if hits.empty else 1.0 / float(hits.iloc[0]["rank"])
        rows.append({
            "method": method,
            "noise_rate": noise_rate,
            "query_id": query_id,
            "query_language": group.iloc[0]["query_language"],
            "expected_language": group.iloc[0]["expected_language"],
            "top1_hit": int(group.sort_values("rank").iloc[0]["doc_id"] == target),
            "top3_hit": int((group.loc[group["rank"] <= 3, "doc_id"] == target).any()),
            "reciprocal_rank": reciprocal,
        })
    detail = pd.DataFrame(rows)
    clean = detail.loc[detail["noise_rate"] == detail["noise_rate"].min()]
    by_lang = clean.groupby("expected_language")["top3_hit"].mean()
    return {
        "query_count": int(detail["query_id"].nunique()),
        "mean_top1_accuracy": float(detail["top1_hit"].mean()),
        "mean_top3_accuracy": float(detail["top3_hit"].mean()),
        "mean_reciprocal_rank": float(detail["reciprocal_rank"].mean()),
        "clean_language_fairness_gap_top3": float(by_lang.max() - by_lang.min()) if len(by_lang) > 1 else 0.0,
    }


def retrieval_metric_table(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, noise_rate), group in results.loc[results["expected_doc_id"] != "NONE"].groupby(["method", "noise_rate"]):
        per_query = []
        for query_id, qgroup in group.groupby("query_id"):
            target = qgroup.iloc[0]["expected_doc_id"]
            hits = qgroup.loc[qgroup["doc_id"] == target].sort_values("rank")
            rr = 0.0 if hits.empty else 1.0 / float(hits.iloc[0]["rank"])
            per_query.append({"top1": int(qgroup.sort_values("rank").iloc[0]["doc_id"] == target), "top3": int((qgroup.loc[qgroup["rank"] <= 3, "doc_id"] == target).any()), "rr": rr})
        frame = pd.DataFrame(per_query)
        rows.append({"method": method, "noise_rate": noise_rate, "top1_accuracy": frame["top1"].mean(), "top3_accuracy": frame["top3"].mean(), "mrr": frame["rr"].mean()})
    return pd.DataFrame(rows).sort_values(["method", "noise_rate"]).reset_index(drop=True)


def qa_metrics(answers: pd.DataFrame) -> dict[str, float | int]:
    answered = answers["status"].eq("answered")
    return {
        "answer_count": int(answered.sum()),
        "refusal_count": int((~answered).sum()),
        "mean_citation_coverage": float(answers["citation_coverage"].mean()) if len(answers) else 0.0,
        "mean_unsupported_claim_score": float(answers["unsupported_claim_score"].mean()) if len(answers) else 0.0,
        "hallucination_risk_rate": float(answers["hallucination_risk"].mean()) if len(answers) else 0.0,
    }


def fairness_audit(results: pd.DataFrame) -> pd.DataFrame:
    base = results.loc[(results["expected_doc_id"] != "NONE") & (results["rank"] <= 3)].copy()
    if base.empty:
        return pd.DataFrame(columns=["language", "top3_recall", "query_count"])
    base["hit"] = base["doc_id"] == base["expected_doc_id"]
    rows = []
    for language, group in base.groupby("expected_language"):
        rows.append({"language": language, "top3_recall": float(group.groupby("query_id")["hit"].max().mean()), "query_count": int(group["query_id"].nunique())})
    return pd.DataFrame(rows).sort_values("language").reset_index(drop=True)
