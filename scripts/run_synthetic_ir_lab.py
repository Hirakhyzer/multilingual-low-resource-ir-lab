"""Run the complete synthetic multilingual low-resource IR and QA laboratory.

The lab uses fictional documents and queries. It compares transparent lexical and
hybrid retrieval baselines, tests OCR-noise robustness, produces citation-grounded
answers, audits language-level fairness, and writes local reports and figures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lowresource_ir.audit import append_record, verify_log
from lowresource_ir.chunking import chunk_documents
from lowresource_ir.config import ensure_output_dirs, set_seed
from lowresource_ir.evaluation import fairness_audit, qa_metrics, retrieval_metric_table, retrieval_metrics
from lowresource_ir.noise import corrupt_documents
from lowresource_ir.qa import answer_from_retrieval, citation_coverage, unsupported_claim_score
from lowresource_ir.reporting import write_report
from lowresource_ir.retrieval import BM25Retriever, TfidfRetriever, hybrid_retrieve
from lowresource_ir.synthetic import SyntheticIRConfig, generate_synthetic_corpus
from lowresource_ir.visualization import plot_citation_grounding, plot_language_fairness, plot_query_status, plot_retrieval_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a synthetic multilingual low-resource IR/QA lab.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--noise-rates", default="0,0.08,0.16", help="Comma-separated OCR noise rates.")
    args = parser.parse_args()

    set_seed(args.seed)
    documents, queries = generate_synthetic_corpus(SyntheticIRConfig(seed=args.seed))
    noise_rates = [float(value.strip()) for value in args.noise_rates.split(",") if value.strip()]

    retrieval_rows: list[dict] = []
    answer_rows: list[dict] = []
    clean_chunks = None

    for noise_rate in noise_rates:
        docs_for_run = documents if noise_rate == 0 else corrupt_documents(documents, noise_rate=noise_rate, seed=args.seed)
        chunks = chunk_documents(docs_for_run)
        if noise_rate == 0:
            clean_chunks = chunks.copy()
        tfidf = TfidfRetriever(chunks, expand_queries=True).fit()
        bm25 = BM25Retriever(chunks, expand_queries=True).fit()
        for query in queries.itertuples(index=False):
            method_frames = [
                tfidf.retrieve(query.query, top_k=args.top_k),
                bm25.retrieve(query.query, top_k=args.top_k),
                hybrid_retrieve(query.query, tfidf, bm25, top_k=args.top_k),
            ]
            for frame in method_frames:
                for row in frame.itertuples(index=False):
                    retrieval_rows.append({
                        "query_id": query.query_id,
                        "query_language": query.query_language,
                        "query": query.query,
                        "expected_doc_id": query.expected_doc_id,
                        "expected_language": query.expected_language,
                        "noise_rate": noise_rate,
                        "method": row.method,
                        "rank": int(row.rank),
                        "chunk_id": row.chunk_id,
                        "doc_id": row.doc_id,
                        "language": row.language,
                        "script": row.script,
                        "topic": row.topic,
                        "score": float(row.score),
                    })
            if noise_rate == 0:
                retrieved = hybrid_retrieve(query.query, tfidf, bm25, top_k=args.top_k)
                response = answer_from_retrieval(query.query, retrieved)
                coverage = citation_coverage(response["answer"], response["citations"])
                unsupported = unsupported_claim_score(response["answer"], response["citations"], retrieved)
                answer_rows.append({
                    "query_id": query.query_id,
                    "query_language": query.query_language,
                    "query": query.query,
                    "expected_doc_id": query.expected_doc_id,
                    "status": response["status"],
                    "answer": response["answer"],
                    "citations": json.dumps(response["citations"], ensure_ascii=False),
                    "top_doc_id": str(retrieved.iloc[0]["doc_id"]) if not retrieved.empty else "NONE",
                    "citation_coverage": coverage,
                    "unsupported_claim_score": unsupported,
                    "hallucination_risk": int(coverage < 1.0 or unsupported > 0.35),
                })

    retrieval_results = pd.DataFrame(retrieval_rows)
    answers = pd.DataFrame(answer_rows)
    metric_table = retrieval_metric_table(retrieval_results)
    fairness = fairness_audit(retrieval_results)
    summary = {
        **retrieval_metrics(retrieval_results),
        **qa_metrics(answers),
        "seed": args.seed,
        "document_count": int(len(documents)),
        "chunk_count": int(len(clean_chunks)) if clean_chunks is not None else 0,
        "query_count": int(len(queries)),
        "languages": sorted(documents["language"].unique().tolist()),
        "data_origin": "synthetic fictional multilingual corpus",
        "research_boundary": "offline baseline only; not a production search or translation benchmark",
    }

    outputs = ensure_output_dirs(args.output_dir)
    documents.to_csv(outputs["results"] / "synthetic_documents.csv", index=False)
    queries.to_csv(outputs["results"] / "synthetic_queries.csv", index=False)
    if clean_chunks is not None:
        clean_chunks.to_csv(outputs["results"] / "synthetic_chunks.csv", index=False)
    retrieval_results.to_csv(outputs["results"] / "synthetic_retrieval_results.csv", index=False)
    answers.to_csv(outputs["results"] / "synthetic_qa_answers.csv", index=False)
    metric_table.to_csv(outputs["results"] / "synthetic_retrieval_metrics.csv", index=False)
    fairness.to_csv(outputs["results"] / "synthetic_fairness_audit.csv", index=False)

    audit_path = outputs["audit"] / "ir_audit_log.jsonl"
    append_record(audit_path, {**summary, "boundary": "synthetic low-resource IR experiment only"})
    summary["audit_log"] = verify_log(audit_path)
    (outputs["results"] / "synthetic_ir_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    write_report(outputs["reports"] / "synthetic_ir_report.md", summary, metric_table, fairness, answers)

    plot_retrieval_comparison(metric_table, outputs["figures"] / "synthetic_retrieval_robustness.png")
    plot_language_fairness(fairness, outputs["figures"] / "synthetic_language_fairness.png")
    plot_query_status(answers, outputs["figures"] / "synthetic_qa_status.png")
    plot_citation_grounding(answers, outputs["figures"] / "synthetic_citation_grounding.png")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
