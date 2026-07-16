# Multilingual corpus data boundary

The default project does not require external data. It generates a fictional multilingual corpus locally.

Real corpora, scans, OCR outputs, annotations, or benchmark datasets must stay under:

```text
data/raw/
```

This folder is ignored by Git.

## Future adapter fields

| Field | Meaning |
| --- | --- |
| doc_id | Stable source identifier |
| language | Urdu, Arabic, English, Roman Urdu, or regional label |
| script | Arabic, Latin, or other script label |
| topic | Domain/topic label |
| title | Document title |
| content | OCR or clean text |
| license | Redistribution and usage license |
| source | Dataset or archive source |

Do not mix real documents with synthetic results unless the report labels them clearly.
