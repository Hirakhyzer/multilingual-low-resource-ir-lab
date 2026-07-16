from lowresource_ir.chunking import chunk_documents
from lowresource_ir.qa import answer_from_retrieval, citation_coverage
from lowresource_ir.retrieval import BM25Retriever, TfidfRetriever, hybrid_retrieve
from lowresource_ir.synthetic import generate_synthetic_corpus


def test_answer_contains_citation_when_supported():
    documents, _ = generate_synthetic_corpus()
    chunks = chunk_documents(documents)
    tfidf = TfidfRetriever(chunks).fit()
    bm25 = BM25Retriever(chunks).fit()
    retrieved = hybrid_retrieve("Sehat camp kab shuru hota hai?", tfidf, bm25, top_k=4)
    response = answer_from_retrieval("Sehat camp kab shuru hota hai?", retrieved)
    assert response["status"] == "answered"
    assert response["citations"]
    assert citation_coverage(response["answer"], response["citations"]) == 1.0
