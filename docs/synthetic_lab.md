# Synthetic multilingual IR lab

## Command

```bash
python scripts/run_synthetic_ir_lab.py
```

Optional controls:

```bash
python scripts/run_synthetic_ir_lab.py --top-k 5 --noise-rates 0,0.10,0.20 --seed 42
```

## Generated outputs

```text
outputs/results/synthetic_documents.csv
outputs/results/synthetic_queries.csv
outputs/results/synthetic_chunks.csv
outputs/results/synthetic_retrieval_results.csv
outputs/results/synthetic_qa_answers.csv
outputs/results/synthetic_retrieval_metrics.csv
outputs/results/synthetic_fairness_audit.csv
outputs/results/synthetic_ir_summary.json
outputs/reports/synthetic_ir_report.md
outputs/audit/ir_audit_log.jsonl

outputs/figures/synthetic_retrieval_robustness.png
outputs/figures/synthetic_language_fairness.png
outputs/figures/synthetic_qa_status.png
outputs/figures/synthetic_citation_grounding.png
```

## Interpretation rules

- All documents and queries are fictional.
- Cross-lingual query expansion is a transparent baseline, not translation.
- OCR noise is simulated.
- No external benchmark result is claimed.
- Future real-corpus experiments require licensing, privacy review, and careful language sampling.
