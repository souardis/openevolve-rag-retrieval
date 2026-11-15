"""
Phase 1: hands-on exploration of BEIR's `scifact` dataset, and a hand-written
BM25 baseline — before any LLM or OpenEvolve gets involved.

Run with: .venv/bin/python exploration/scifact_baseline.py
"""

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from rank_bm25 import BM25Okapi

DATASET = "scifact"
DATA_DIR = "datasets"


def load_scifact(split: str):
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{DATASET}.zip"
    data_path = util.download_and_unzip(url, DATA_DIR)
    return GenericDataLoader(data_folder=data_path).load(split=split)


def inspect(corpus: dict, queries: dict, qrels: dict) -> None:
    print(f"corpus: {len(corpus)} documents")
    print(f"queries: {len(queries)} queries")
    print(f"qrels: {len(qrels)} queries with relevance judgments")

    some_doc_id = next(iter(corpus))
    print(f"\nexample document ({some_doc_id}):")
    print(corpus[some_doc_id])

    some_qid = next(iter(queries))
    print(f"\nexample query ({some_qid}):")
    print(queries[some_qid])
    print(f"its qrels: {qrels.get(some_qid)}")


# ---------------------------------------------------------------------------
# TODO (this is the part to write yourself): a plain BM25 baseline retriever.
#
# Signature: retrieve(corpus: dict, queries: dict) -> dict
#   - corpus[doc_id] = {"title": str, "text": str}
#   - queries[qid]   = str
#   - return shape must be BEIR's expected results format:
#       {qid: {doc_id: score, ...}, ...}
#     with the top 100 scored docs per query (BEIR's own recall@100 convention
#     — don't return the full corpus score for every query, it's wasteful and
#     BEIR's evaluate() only looks at the ranks within what you give it anyway)
#
# rank_bm25's BM25Okapi has exactly two methods you need:
#   - BM25Okapi(tokenized_corpus) to build the index
#   - bm25.get_scores(tokenized_query) to score one query against every document
#
# Tokenization is up to you — simplest is `text.lower().split()`. You decide
# whether to concatenate title+text, and whether to weight title differently
# from body text (that's the kind of knob OpenEvolve will later discover it
# can turn — but right now, keep it simple, this is just the baseline).
# ---------------------------------------------------------------------------
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


def main():
    corpus, queries, qrels = load_scifact(split="train")
    inspect(corpus, queries, qrels)

    results = retrieve(corpus, queries)
    evaluator = EvaluateRetrieval()
    ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, k_values=[10])
    print("\nbaseline scores (train split):")
    print(ndcg)
    print(recall)


if __name__ == "__main__":
    main()
