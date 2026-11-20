"""
Lab B evaluator: quality + latency fitness.

Identical to Lab A's evaluator (openevolve_task/lab_quality_only/evaluator.py)
except for two deliberate changes: `fitness()`, and timing retrieve()
TIMING_REPEATS times and keeping the median (latency is part of the score
here, so its noise matters; in Lab A it's only reported). Same data, same
source check, same validation. If you fix something in the shared part, fix
it in both files.

Standalone check (no OpenEvolve, no API calls):
    uv run python openevolve_task/lab_quality_latency/evaluator.py [program.py]
"""

import copy
import importlib.util
import math
import re
import statistics
import sys
import time
import traceback
from pathlib import Path

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from openevolve.evaluation_result import EvaluationResult

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "datasets"  # absolute: OpenEvolve doesn't guarantee our cwd
DATASET_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
SPLIT = "train"  # evolution feedback only; test split stays untouched until Phase 3
TOP_K_MAX = 100  # BEIR convention: at most 100 ranked docs per query

# Candidates only get corpus + queries as arguments. Anything that could reach
# the relevance labels on disk (datasets/scifact/qrels/*.tsv) or escape the
# sandbox is rejected before the program is even imported. A candidate that
# reads qrels scores a perfect 1.0, and so would the test-split check in
# Phase 3, because those labels sit in the same folder. This is a cheap string
# check, not a real sandbox: read the winning programs' code in Phase 3 too.
FORBIDDEN_PATTERNS = [
    r"\bopen\s*\(",
    r"\bos\b",
    r"\bsys\b",
    r"\bpathlib\b",
    r"\bshutil\b",
    r"\bsubprocess\b",
    r"\bsocket\b",
    r"\burllib\b",
    r"\brequests\b",
    r"\bimportlib\b",
    r"__import__",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\bglobals\s*\(",
    r"\bbeir\b",
    r"qrels",
    r"datasets",
]

# ---------------------------------------------------------------------------
# THE PART THAT MAKES THIS LAB B. Everything else in this file is Lab A's code.
#
# Design (plan, Phase 2, step 5, with the clip changed to log2, see below):
#   speed_term     = clip(log2(BASELINE_LATENCY_MS / latency_ms), -1, 1)
#   combined_score = ndcg_at_10 + LATENCY_WEIGHT * speed_term
# combined_score always stays within ndcg_at_10 +/- LATENCY_WEIGHT.
#
# Why log2 and not the plan's (ratio - 1): ratio - 1 is lopsided. Its bonus
# caps at 2x faster, but 2x slower costs only -0.5 and 10x slower -0.9, so an
# expensive add-on (query expansion, a second scoring pass) that buys +0.03
# nDCG@10 would win even at 3x slower. log2 makes 2x faster and 2x slower
# equal and opposite (+1 / -1).
# ---------------------------------------------------------------------------
# Unmodified initial_program.py on scifact/train, measured on this laptop (exploration/NOTES.md).
BASELINE_LATENCY_MS = 9.386
LATENCY_WEIGHT = 0.05  # 2x faster than baseline = +0.05 nDCG@10, 2x slower = -0.05
MIN_LATENCY_MS = 1e-6  # floor: an instant candidate (returns {}) must not divide by zero
# Single runs of the baseline vary 9.1-9.6 ms/query (~+/-0.002 combined_score).
# The median of 3 damps that, at ~2x extra evaluation time.
TIMING_REPEATS = 3


def fitness(ndcg_10: float, latency_ms: float) -> float:
    """Lab B's combined_score: nDCG@10 plus a bounded bonus/penalty for speed vs. baseline.

    speed_term: +1 at >= 2x faster, 0 at baseline speed, -1 at >= 2x slower.
    """
    speed_ratio = BASELINE_LATENCY_MS / max(latency_ms, MIN_LATENCY_MS)
    speed_term = max(-1.0, min(1.0, math.log2(speed_ratio)))
    return ndcg_10 + LATENCY_WEIGHT * speed_term


_DATA_CACHE = None


def load_data():
    """(corpus, queries, qrels) for scifact/train, loaded once per process.

    Never hand these cached objects to a candidate directly: OpenEvolve reuses
    this module for many evaluations, so an in-place edit by one candidate
    would silently corrupt the data every later candidate gets scored on.
    """
    global _DATA_CACHE
    if _DATA_CACHE is None:
        data_path = util.download_and_unzip(DATASET_URL, str(DATA_DIR))
        _DATA_CACHE = GenericDataLoader(data_folder=data_path).load(split=SPLIT)
    return _DATA_CACHE


def load_program(program_path: str):
    """Import a candidate .py file as a fresh module and return it."""
    spec = importlib.util.spec_from_file_location("candidate_program", program_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_source(program_path: str):
    """Return an error message if the candidate's source uses a forbidden construct, else None."""
    source = Path(program_path).read_text()
    hits = [p for p in FORBIDDEN_PATTERNS if re.search(p, source)]
    if hits:
        return (
            "Rejected before running: the program must not do file, OS, network or "
            f"dynamic-import access. Forbidden patterns found: {hits}"
        )
    return None


def normalize_results(results, queries: dict) -> dict:
    """Validate retrieve()'s output and coerce it to BEIR's {qid: {doc_id: float}} format.

    Every query gets an entry. Missing ones become {} and score 0. BEIR only
    averages over the queries present in `results`, so without this a candidate
    could drop the queries it gets wrong and inflate nDCG@10 (measured: 0.59 -> 0.99).
    """
    if not isinstance(results, dict):
        raise TypeError(f"retrieve() must return a dict, got {type(results).__name__}")
    normalized = {}
    for qid in queries:
        ranked = results.get(qid, {})
        if not isinstance(ranked, dict):
            raise TypeError(f"results[{qid!r}] must be a dict of doc_id -> score, got {type(ranked).__name__}")
        if len(ranked) > TOP_K_MAX:
            raise ValueError(f"results[{qid!r}] has {len(ranked)} docs, max is {TOP_K_MAX}")
        normalized[qid] = {str(doc_id): float(score) for doc_id, score in ranked.items()}
    return normalized


def failure(message: str) -> EvaluationResult:
    """Bottom-ranked score plus the reason, which OpenEvolve puts into the next mutation prompt."""
    return EvaluationResult(
        metrics={
            "ndcg_at_10": 0.0,
            "recall_at_10": 0.0,
            "latency_ms_per_query": 0.0,
            "combined_score": 0.0,
        },
        artifacts={"error": message},
    )


def evaluate(program_path: str):
    rejection = check_source(program_path)
    if rejection:
        return failure(rejection)

    try:
        cached_corpus, cached_queries, qrels = load_data()

        # Lab B: time retrieve() TIMING_REPEATS times, keep the median, score
        # the first run's results. Each repeat re-imports the program and gets
        # fresh data copies, all outside the timer. The re-import matters: on a
        # reused module, a candidate could cache results in a module-level dict
        # so repeats 2 and 3 take ~0 ms and the median rewards the cache.
        timings = []
        results = None
        for _ in range(TIMING_REPEATS):
            corpus = copy.deepcopy(cached_corpus)
            queries = copy.deepcopy(cached_queries)
            program = load_program(program_path)

            start = time.perf_counter()
            run_results = program.retrieve(corpus, queries)
            timings.append(time.perf_counter() - start)
            if results is None:
                results = run_results
        latency = statistics.median(timings) / len(cached_queries) * 1000

        results = normalize_results(results, cached_queries)
        ndcg, _map, recall, _precision = EvaluateRetrieval.evaluate(qrels, results, k_values=[10])
    except Exception as exc:
        tb = traceback.format_exc(limit=5)
        return failure(f"{type(exc).__name__}: {exc}\n{tb}")

    ndcg_10 = ndcg["NDCG@10"]
    return {
        "ndcg_at_10": ndcg_10,
        "recall_at_10": recall["Recall@10"],
        "latency_ms_per_query": latency,
        "combined_score": fitness(ndcg_10, latency),  # Lab B: quality + latency
    }


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else str(REPO_ROOT / "openevolve_task" / "initial_program.py")
    outcome = evaluate(target)
    if isinstance(outcome, EvaluationResult):
        print(outcome.metrics)
        print("error:", outcome.artifacts.get("error"))
    else:
        print(outcome)
