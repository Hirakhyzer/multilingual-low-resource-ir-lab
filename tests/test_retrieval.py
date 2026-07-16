from lowresource_ir.chunking import chunk_documents
from lowresource_ir.retrieval import BM25Retriever, TfidfRetriever, hybrid_retrieve
from lowresource_ir.synthetic import generate_synthetic_corpus


def test_roman_urdu_query_retrieves_matching_document():
    documents, _ = generate_synthetic_corpus()
    chunks = chunk_documents(documents)
    tfidf = TfidfRetriever(chunks).fit()
    result = tfidf.retrieve("Sehat camp kab shuru hota hai?", top_k=3)
    assert "RU-HEALTH-003" in set(result["doc_id"])


def test_hybrid_retrieval_returns_ranked_rows():
    documents, _ = generate_synthetic_corpus()
    chunks = chunk_documents(documents)
    tfidf = TfidfRetriever(chunks).fit()
    bm25 = BM25Retriever(chunks).fit()
    result = hybrid_retrieve("When does registration close for the youth skills workshop?", tfidf, bm25, top_k=5)
    assert len(result) == 5
    assert result["rank"].tolist() == [1, 2, 3, 4, 5]
    assert result.iloc[0]["score"] >= result.iloc[-1]["score"]
