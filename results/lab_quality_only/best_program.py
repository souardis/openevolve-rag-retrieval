from rank_bm25 import BM25Okapi


# EVOLVE-BLOCK-START
import re
import numpy as np

_STOP = set(
    "a an the of in on at to for and or with by from is are was were be been "
    "this that these those it its as which who whom than then there their they "
    "we our using used via into over under can not no".split()
)


def _tok(text: str) -> list:
    out = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w in _STOP:
            continue
        if len(w) > 4 and w.endswith("ies"):
            out.append(w[:-3] + "y")
            continue
        for suf in ("ing", "ed", "es", "s"):
            if len(w) > 4 and w.endswith(suf):
                w = w[: -len(suf)]
                break
        out.append(w)
    return out


def retrieve(corpus: dict, queries: dict) -> dict:
    doc_ids = list(corpus.keys())
    # Title is repeated to give it extra weight in the term statistics.
    tokenized_corpus = [
        _tok(corpus[doc_id]["title"] + " " + corpus[doc_id]["title"] + " " + corpus[doc_id]["text"])
        for doc_id in doc_ids
    ]
    bm25 = BM25Okapi(tokenized_corpus, k1=0.9, b=0.4)
    title_sets = [set(_tok(corpus[doc_id]["title"])) for doc_id in doc_ids]
    # Adjacent-term pairs per document, used as a cheap phrase-proximity signal.
    bigram_sets = [set(zip(t, t[1:])) for t in tokenized_corpus]
    doc_sets = [set(t) for t in tokenized_corpus]

    results = {}
    for qid, query_text in queries.items():
        raw_query = _tok(query_text)
        tokenized_query = list(dict.fromkeys(raw_query))
        scores = np.asarray(bm25.get_scores(tokenized_query), dtype=float)
        qset = set(tokenized_query)
        qbigrams = set(zip(raw_query, raw_query[1:]))
        if qset:
            # Title coverage: fraction of distinct query terms present in the title.
            tb = np.array([len(qset & t) for t in title_sets], dtype=float) / len(qset)
            scores = scores / (scores.max() or 1.0) + 0.3 * tb
            if qbigrams:
                # Phrase proximity: fraction of query bigrams appearing adjacently in the doc.
                bb = np.array([len(qbigrams & b) for b in bigram_sets], dtype=float) / len(qbigrams)
                scores = scores + 0.25 * bb
            # Pseudo-relevance feedback: boost docs whose vocabulary overlaps the top-3 first-pass docs.
            top5 = np.argsort(-scores)[:5]
            fset = set().union(*(doc_sets[i] for i in top5))
            if fset:
                fb = np.array([len(fset & d) for d in doc_sets], dtype=float) / len(fset)
                scores = scores + 0.15 * fb
        top_docs = sorted(zip(doc_ids, scores), key=lambda pair: pair[1], reverse=True)[:100]
        results[qid] = {doc_id: float(score) for doc_id, score in top_docs}

    return results
# EVOLVE-BLOCK-END