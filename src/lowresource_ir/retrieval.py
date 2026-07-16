"""Lexical and hybrid multilingual retrieval baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from lowresource_ir.normalization import expand_cross_lingual_query, normalize_text


class TfidfRetriever:
    """Character/word TF-IDF retriever with optional cross-lingual expansion."""

    def __init__(self, chunks: pd.DataFrame, expand_queries: bool = True):
        self.chunks = chunks.reset_index(drop=True).copy()
        self.expand_queries = expand_queries
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), lowercase=False)
        self.matrix = None

    def fit(self) -> "TfidfRetriever":
        texts = [normalize_text(value, lang) for value, lang in zip(self.chunks["chunk_text"], self.chunks["language"])]
        self.matrix = self.vectorizer.fit_transform(texts)
        return self

    def retrieve(self, query: str, top_k: int = 5) -> pd.DataFrame:
        if self.matrix is None:
            raise RuntimeError("Call fit() before retrieve().")
        q = expand_cross_lingual_query(query) if self.expand_queries else query
        query_vec = self.vectorizer.transform([normalize_text(q)])
        scores = cosine_similarity(query_vec, self.matrix).ravel()
        order = np.argsort(scores)[::-1][:top_k]
        out = self.chunks.iloc[order].copy()
        out["score"] = scores[order]
        out["rank"] = range(1, len(out) + 1)
        out["method"] = "tfidf_char_xlingual" if self.expand_queries else "tfidf_char"
        return out.reset_index(drop=True)


class BM25Retriever:
    """Small BM25-style lexical baseline with transparent offline implementation."""

    def __init__(self, chunks: pd.DataFrame, expand_queries: bool = True, k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks.reset_index(drop=True).copy()
        self.expand_queries = expand_queries
        self.k1 = k1
        self.b = b
        self.vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b", preprocessor=normalize_text)
        self.counts = None
        self.idf = None
        self.doc_len = None
        self.avgdl = 1.0

    def fit(self) -> "BM25Retriever":
        texts = self.chunks["chunk_text"].tolist()
        self.counts = self.vectorizer.fit_transform(texts).astype(float)
        df = np.asarray((self.counts > 0).sum(axis=0)).ravel()
        n_docs = self.counts.shape[0]
        self.idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
        self.doc_len = np.asarray(self.counts.sum(axis=1)).ravel()
        self.avgdl = float(np.mean(self.doc_len)) if len(self.doc_len) else 1.0
        return self

    def retrieve(self, query: str, top_k: int = 5) -> pd.DataFrame:
        if self.counts is None or self.idf is None or self.doc_len is None:
            raise RuntimeError("Call fit() before retrieve().")
        q = expand_cross_lingual_query(query) if self.expand_queries else query
        query_counts = self.vectorizer.transform([q])
        q_indices = query_counts.nonzero()[1]
        scores = np.zeros(self.counts.shape[0])
        for term_idx in q_indices:
            tf = self.counts[:, term_idx].toarray().ravel()
            denom = tf + self.k1 * (1 - self.b + self.b * self.doc_len / max(self.avgdl, 1e-9))
            scores += self.idf[term_idx] * (tf * (self.k1 + 1)) / np.maximum(denom, 1e-9)
        order = np.argsort(scores)[::-1][:top_k]
        out = self.chunks.iloc[order].copy()
        out["score"] = scores[order]
        out["rank"] = range(1, len(out) + 1)
        out["method"] = "bm25_xlingual" if self.expand_queries else "bm25"
        return out.reset_index(drop=True)


def hybrid_retrieve(query: str, tfidf: TfidfRetriever, bm25: BM25Retriever, top_k: int = 5) -> pd.DataFrame:
    """Combine normalized TF-IDF and BM25 ranks into a hybrid score."""
    t = tfidf.retrieve(query, top_k=max(top_k, 8)).rename(columns={"score": "tfidf_score"})
    b = bm25.retrieve(query, top_k=max(top_k, 8)).rename(columns={"score": "bm25_score"})
    merged = t.merge(b[["chunk_id", "bm25_score"]], on="chunk_id", how="outer")
    merged["tfidf_score"] = merged["tfidf_score"].fillna(0.0)
    merged["bm25_score"] = merged["bm25_score"].fillna(0.0)
    for column in ["tfidf_score", "bm25_score"]:
        max_score = float(merged[column].max())
        if max_score > 0:
            merged[column] = merged[column] / max_score
    merged["score"] = 0.58 * merged["tfidf_score"] + 0.42 * merged["bm25_score"]
    merged["method"] = "hybrid_lexical_xlingual"
    merged = merged.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)
    merged["rank"] = range(1, len(merged) + 1)
    return merged
