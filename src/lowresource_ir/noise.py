"""OCR-noise simulation for Urdu/Arabic/Latin text."""

from __future__ import annotations

import numpy as np

CONFUSIONS = {
    "ب": "پ", "پ": "ب", "ت": "ث", "ث": "ت", "ج": "ح", "ح": "ج", "د": "ذ", "ر": "ز", "س": "ش", "ص": "ض", "ط": "ظ", "ع": "غ", "ف": "ق", "ك": "ک", "ي": "ی",
    "ا": "l", "o": "0", "i": "l", "rn": "m", "cl": "d",
}

PUNCT = ".,;:!?؟،؛"


def apply_ocr_noise(text: str, noise_rate: float = 0.10, seed: int = 42) -> str:
    """Apply deterministic character-level OCR-like corruption.

    This models missing punctuation, character confusion, and spacing errors. It is
    a robustness stress test, not a claim about any OCR engine.
    """
    rng = np.random.default_rng(seed)
    chars: list[str] = []
    for char in str(text):
        if char in PUNCT and rng.random() < noise_rate * 1.8:
            continue
        if char == " " and rng.random() < noise_rate * 0.9:
            continue
        if char in CONFUSIONS and rng.random() < noise_rate:
            chars.append(CONFUSIONS[char])
        elif rng.random() < noise_rate * 0.08:
            continue
        else:
            chars.append(char)
    return "".join(chars)


def corrupt_documents(documents, noise_rate: float, seed: int = 42):
    noisy = documents.copy()
    noisy["clean_content"] = noisy["content"]
    noisy["content"] = [apply_ocr_noise(text, noise_rate=noise_rate, seed=seed + idx) for idx, text in enumerate(noisy["content"].tolist())]
    noisy["ocr_noise_rate"] = noise_rate
    return noisy
