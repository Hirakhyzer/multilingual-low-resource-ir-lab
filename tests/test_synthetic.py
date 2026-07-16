from lowresource_ir.synthetic import generate_synthetic_corpus


def test_synthetic_corpus_has_multilingual_coverage():
    documents, queries = generate_synthetic_corpus()
    assert {"urdu", "arabic", "english", "roman_urdu"}.issubset(set(documents["language"]))
    assert len(queries) >= 8
    assert (documents["doc_id"].is_unique)


def test_queries_include_cross_lingual_cases():
    _, queries = generate_synthetic_corpus()
    cross = queries.loc[queries["query_language"] != queries["expected_language"]]
    assert len(cross) >= 3
