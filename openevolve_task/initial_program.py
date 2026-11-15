from rank_bm25 import BM25Okapi


# EVOLVE-BLOCK-START
def retrieve(corpus: dict, queries: dict) -> dict:
    doc_ids = list(corpus.keys())
    tokenized_corpus = [
        (corpus[doc_id]["title"] + " " + corpus[doc_id]["text"]).lower().split()
        for doc_id in doc_ids
    ]
    bm25 = BM25Okapi(tokenized_corpus)

    results = {}
    for qid, query_text in queries.items():
        tokenized_query = query_text.lower().split()
        scores = bm25.get_scores(tokenized_query)
        top_docs = sorted(zip(doc_ids, scores), key=lambda pair: pair[1], reverse=True)[:100]
        results[qid] = {doc_id: float(score) for doc_id, score in top_docs}

    return results
# EVOLVE-BLOCK-END