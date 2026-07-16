"""Script-aware normalization and tokenization utilities."""

from __future__ import annotations

import re
import unicodedata

ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
PUNCTUATION = re.compile(r"[\.,;:!؟?،؛\(\)\[\]\{\}\"'`“”‘’]")

ARABIC_CHAR_MAP = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ئ": "ي", "ؤ": "و", "ة": "ه",
    "ک": "ك", "ی": "ي", "ھ": "ه", "ۀ": "ه",
})

ROMAN_URDU_HINTS = {
    "sehat": "health", "camp": "camp", "kab": "when", "shuru": "start", "hota": "held",
    "class": "class", "haftay": "week", "din": "days", "pani": "water", "school": "school",
}

CROSS_LINGUAL_HINTS = {
    "health": "صحت sehat مرکز center", "water": "پانی water filtration", "school": "مدارس school class", "agriculture": "زراعت agriculture crops",
    "flood": "سیلاب flood shelter", "registration": "registration تسجيل رجسٹریشن", "camp": "camp sehat صحت", "class": "class taleemi تعلیم",
    "grades": "الصف grades school", "crops": "محاصيل crops irrigation", "shelter": "پناہ shelter school",
}


def normalize_text(text: str, language: str | None = None) -> str:
    """Normalize multilingual text without destructive transliteration."""
    value = unicodedata.normalize("NFKC", str(text)).lower()
    value = ARABIC_DIACRITICS.sub("", value)
    value = value.translate(ARABIC_CHAR_MAP)
    value = PUNCTUATION.sub(" ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def tokenize(text: str, language: str | None = None) -> list[str]:
    value = normalize_text(text, language)
    return [token for token in re.split(r"\s+", value) if token]


def expand_cross_lingual_query(query: str) -> str:
    """Add a tiny transparent synonym bridge for synthetic cross-lingual retrieval.

    This is not a real translation system. It is an auditable baseline that makes
    the synthetic lab runnable offline before adding multilingual embeddings.
    """
    normalized = normalize_text(query)
    extra_terms: list[str] = []
    for key, hints in CROSS_LINGUAL_HINTS.items():
        if key in normalized:
            extra_terms.append(hints)
    for key, hint in ROMAN_URDU_HINTS.items():
        if key in normalized:
            extra_terms.append(hint)
    return " ".join([query, *extra_terms]).strip()
