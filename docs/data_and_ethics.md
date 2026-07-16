# Data, licensing, and ethics policy

## Synthetic-first boundary

The repository ships only fictional multilingual content. Real documents, scans, OCR outputs, or regional-language corpora must not be committed unless the license explicitly permits redistribution.

## Real corpus requirements

Before adding a real corpus adapter:

1. Document the source, license, collection date, language, script, domain, and preprocessing steps.
2. Keep raw files under `data/raw/`, which is ignored by Git.
3. Remove or protect personal information.
4. Separate evaluation labels from training data where applicable.
5. Report language and script coverage honestly.
6. Avoid claiming fairness or robustness from tiny or biased samples.

## OCR and regional-language caution

OCR errors can vary by font, scanner quality, script, page layout, and dialect. The simulated noise in this repository is a research stress test only.

## Deployment boundary

This lab is not a legal, medical, government, or citizen-service retrieval system. Real deployment requires evaluation with native speakers, accessibility review, privacy review, monitoring, and human escalation pathways.
