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

### Re-measured under OpenEvolve (2026-11-25): `BASELINE_LATENCY_MS = 9.71`

The Phase 1 number above is a single standalone run. Lab B compares candidates timed *inside* OpenEvolve, so the reference was re-measured the same way: Lab B's evaluator (median of 3 timed `retrieve()` calls per evaluation) on the unmodified `initial_program.py`, nothing else running.

| Path | 5 evaluations (ms/query) |
|---|---|
| Main process (`openevolve-run --iterations 0`, how the starting program is scored) | 9.66, 9.70, 9.54, 9.72, 9.78 |
| Spawned worker process with OpenEvolve's `Evaluator` (how every candidate is scored) | 9.68, 9.81, 9.61, 9.79, 9.76 |

Median of all 10: **9.71 ms/query** (range 9.54-9.81, so ±1.4%). The two paths agree, so a worker process adds no measurable overhead. The 10.67 ms seen in the first Lab B smoke run (2026-11-20, started right after Lab A's run) was an outlier, not systematic overhead. With 9.386 the unmodified baseline scored about -0.0025 in Lab B; with 9.71 it scores ~0, as designed.

## Sanity check against the literature

Published BEIR results for plain BM25 on `scifact` are typically in the ~0.60-0.70 nDCG@10 range. 0.59 is close enough to confirm the harness (tokenization, id bookkeeping, BEIR evaluation call) is correct, not a bug.

## Caveat: train vs. test

This number is on the **train** split (809 queries), not the official **test** split that published literature numbers and leaderboards use. That's intentional, not an oversight: train is reserved for fast iteration feedback during evolution (Phase 2+), and test stays untouched until the final validation pass (Phase 3) — mirroring the Nature paper's separation between cheap iteration feedback and a rigorous final check. Don't read the 0.59 vs. ~0.60-0.70 gap as a real discrepancy; it's an apples-to-oranges comparison until we re-run this same baseline on the test split during Phase 3.
