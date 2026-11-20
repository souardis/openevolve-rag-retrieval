# Phase 1 notes — BM25 baseline on BEIR SciFact

## Data

- Split used for this baseline: `train` (5,183 corpus docs, 809 queries with qrels).
- `corpus[doc_id]` = `{"title": str, "text": str}` — `text` is the paper's abstract, not full text.
- `queries[qid]` = a scientific claim (str).
- `qrels[qid]` = `{doc_id: relevance}` — binary (`1` = relevant), usually one relevant doc per query.
- Doc ids and query ids are both strings. (Checked explicitly in the REPL — this is the #1 silent-bug source: a string/int mismatch between `corpus`, `queries`, `qrels` would make BM25 "work" but score everything 0.)

## Implementation choices (`exploration/scifact_baseline.py`)

- **Tokenization**: `.lower().split()` — plain whitespace split, no stemming/stopword removal. Deliberately simple for a baseline.
- **Field handling**: concatenated `title + " " + text` into one field per document, no separate weighting between the two. This is exactly the kind of knob an evolutionary search (OpenEvolve, later) could discover is worth splitting out.
- **Library**: `rank_bm25`'s `BM25Okapi` (the original/canonical BM25 formula, as opposed to the `BM25L`/`BM25+` variants also in that package).
- **Output truncation**: top 100 scored docs per query, matching BEIR's own recall@100 convention — not the full corpus.

## Result

- `NDCG@10 = 0.58992`
- `Recall@10 = 0.72015`
- `BASELINE_LATENCY_MS = 9.386` ms/query (7.59s total over 809 queries, measured with `time.perf_counter()` around the `retrieve()` call — wall-clock, single run, no repeated trials, on this laptop at the time of measurement)
- (on the **train** split — see caveat below)

This `BASELINE_LATENCY_MS` is the fixed speed reference point Lab B's fitness formula compares every evolved candidate against (`speed_ratio = BASELINE_LATENCY_MS / candidate_latency_ms`). It's a laptop wall-clock measurement, not an isolated/averaged-over-many-runs benchmark — noisy by production standards, but consistent enough for comparing programs run sequentially on the same machine, which is all this needs.

## Sanity check against the literature

Published BEIR results for plain BM25 on `scifact` are typically in the ~0.60-0.70 nDCG@10 range. 0.59 is close enough to confirm the harness (tokenization, id bookkeeping, BEIR evaluation call) is correct, not a bug.

## Caveat: train vs. test

This number is on the **train** split (809 queries), not the official **test** split that published literature numbers and leaderboards use. That's intentional, not an oversight: train is reserved for fast iteration feedback during evolution (Phase 2+), and test stays untouched until the final validation pass (Phase 3) — mirroring the Nature paper's separation between cheap iteration feedback and a rigorous final check. Don't read the 0.59 vs. ~0.60-0.70 gap as a real discrepancy; it's an apples-to-oranges comparison until we re-run this same baseline on the test split during Phase 3.
