# Methodology

## Scope

This repository is a synthetic-first multilingual information retrieval and citation-grounded QA lab. It is designed for reproducible research on Urdu, Arabic, English, Roman Urdu, and regional-language placeholders without requiring a licensed external corpus.

It does not claim production search quality, OCR benchmark performance, translation quality, or official linguistic coverage.

## Corpus design

The default corpus contains fictional short documents with document IDs, language labels, script labels, topic labels, titles, content, and answer facts. The query set includes monolingual and cross-lingual questions, including English-to-Urdu, English-to-Arabic, Urdu-to-English, and Roman Urdu retrieval.

## Normalization

The lab applies Unicode normalization, Arabic-script diacritic removal, selected Urdu/Arabic character normalization, punctuation cleanup, and whitespace normalization. It avoids destructive transliteration by default.

## Retrieval baselines

| Method | Role |
| --- | --- |
| `tfidf_char_xlingual` | Character n-gram TF-IDF baseline for mixed scripts and noisy text |
| `bm25_xlingual` | Transparent BM25-style lexical baseline |
| `hybrid_lexical_xlingual` | Weighted combination of TF-IDF and BM25 evidence |
| Embedding adapter | Documented future extension for multilingual embeddings; no result claimed until executed |

A small transparent query-expansion bridge is included for the synthetic cross-lingual task. It is not a translation system.

## OCR-noise robustness

The lab simulates character confusion, punctuation loss, spacing corruption, and dropped characters. These are stress tests, not claims about any real OCR engine.

## Citation-grounded QA

Answers are extractive. The system only answers from retrieved chunks and appends chunk citations such as `UR-HEALTH-001::C00`. Unsupported questions should return a fallback rather than an invented answer.

## Fairness audit

The fairness audit compares retrieval recall across expected document languages. This is a regression signal for synthetic experiments, not a legal or social-scientific fairness conclusion.

## Limitations

- Synthetic documents are too small to represent real Urdu or Arabic corpora.
- Query expansion is a research convenience, not translation.
- BM25 and TF-IDF are strong baselines but do not model deep semantics.
- OCR noise is simulated and simplified.
- Language fairness gaps require much larger datasets and careful sampling in real studies.
