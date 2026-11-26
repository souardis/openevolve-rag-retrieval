"""
Summarize one OpenEvolve run into files small enough to commit.

OpenEvolve's output folder (logs, checkpoints, ~30 MB) is gitignored. This
script reads the run log plus the last checkpoint's program database and
writes, into RESULTS_DIR:

    iterations.csv          one row per iteration: status, metrics, tokens, model
    summary.json            headline numbers (best program, failures, tokens, milestones)
    progress.png            candidates and best-so-far per iteration, plus latency
    best_program.py         copy of the winning program
    best_program_info.json  its metrics, as OpenEvolve saved them

Usage (no API calls, no evaluation, just reads files):
    uv run python scripts/summarize_run.py <openevolve_output_dir> <results_dir> [--baseline-ndcg 0.58992]
"""

import argparse
import csv
import json
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only, never open a window
import matplotlib.pyplot as plt  # noqa: E402

ITER_OK = re.compile(
    r"Iteration (?P<it>\d+): Program (?P<id>[0-9a-f-]{36}) \(parent: (?P<parent>[0-9a-f-]{36})\) "
    r"completed in (?P<secs>[\d.]+)s \| tokens: (?P<tok>\d+) \(prompt: (?P<prompt>\d+), completion: (?P<comp>\d+)\)"
)
ITER_ERR = re.compile(
    r"Iteration (?P<it>\d+) error \[(?P<mode>\w+)\]: (?P<msg>.*?)"
    r"(?: \| tokens: (?P<tok>\d+) \(prompt: (?P<prompt>\d+), completion: (?P<comp>\d+)\))?$"
)
METRICS = re.compile(r" - Metrics: (?P<body>.*)$")
INITIAL = re.compile(r"Evaluated program (?P<id>[0-9a-f-]{36}) in [\d.]+s: (?P<body>.*)$")
TOTAL_TOKENS = re.compile(r"Total LLM Token Usage \((?P<calls>\d+) calls\): total_tokens=(?P<tot>\d+) \(prompt_tokens=(?P<p>\d+), completion_tokens=(?P<c>\d+)\)")


def parse_metrics(body: str) -> dict:
    return {k: float(v) for k, v in re.findall(r"(\w+)=([-\d.eE]+)", body)}


def parse_log(log_path: Path):
    rows, initial, totals = [], None, None
    pending = None  # an OK iteration waits for its "Metrics:" line
    for line in log_path.read_text().splitlines():
        if initial is None and (m := INITIAL.search(line)):
            initial = {"id": m["id"], **parse_metrics(m["body"])}
        elif m := ITER_OK.search(line):
            pending = {
                "iteration": int(m["it"]), "status": "ok", "program_id": m["id"], "parent_id": m["parent"],
                "seconds": float(m["secs"]), "prompt_tokens": int(m["prompt"]), "completion_tokens": int(m["comp"]),
            }
        elif pending and (m := METRICS.search(line)):
            pending.update(parse_metrics(m["body"]))
            rows.append(pending)
            pending = None
        elif m := ITER_ERR.search(line):
            rows.append({
                "iteration": int(m["it"]), "status": "error", "error": m["msg"],
                "prompt_tokens": int(m["prompt"]) if m["prompt"] else None,
                "completion_tokens": int(m["comp"]) if m["comp"] else None,
            })
        elif m := TOTAL_TOKENS.search(line):
            totals = {"calls": int(m["calls"]), "total_tokens": int(m["tot"]),
                      "prompt_tokens": int(m["p"]), "completion_tokens": int(m["c"])}
    rows.sort(key=lambda r: r["iteration"])
    return rows, initial, totals


def load_models(checkpoint: Path) -> dict:
    """program_id -> LLM model, for the programs still in the final database."""
    models = {}
    for f in (checkpoint / "programs").glob("*.json"):
        p = json.loads(f.read_text())
        models[p["id"]] = (p.get("metadata") or {}).get("llm_model")
    return models


def last_checkpoint(output_dir: Path) -> Path:
    cps = sorted((output_dir / "checkpoints").glob("checkpoint_*"), key=lambda p: int(p.name.split("_")[1]))
    return cps[-1]


def plot(rows, initial, out_png: Path, title: str):
    ok = [r for r in rows if r["status"] == "ok"]
    err = [r for r in rows if r["status"] == "error"]
    lab_b = any(abs(r["combined_score"] - r["ndcg_at_10"]) > 1e-9 for r in ok)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    xs = [r["iteration"] for r in ok]
    ax1.scatter(xs, [r["ndcg_at_10"] for r in ok], s=12, alpha=0.5, color="#4c78a8", label="candidate nDCG@10")
    best, curve = initial["combined_score"], []
    by_it = {r["iteration"]: r for r in ok}
    for it in range(1, max(r["iteration"] for r in rows) + 1):
        if it in by_it:
            best = max(best, by_it[it]["combined_score"])
        curve.append(best)
    ax1.step(range(1, len(curve) + 1), curve, where="post", color="#e45756", lw=2,
             label="best combined_score so far" if lab_b else "best nDCG@10 so far")
    ax1.axhline(initial["ndcg_at_10"], color="gray", ls="--", lw=1, label=f"baseline {initial['ndcg_at_10']:.4f}")
    if err:
        lo = min(r["ndcg_at_10"] for r in ok)
        ax1.scatter([r["iteration"] for r in err], [lo - 0.01] * len(err), marker="x", color="black", s=18,
                    label=f"failed iteration ({len(err)})")
    ax1.set_ylabel("score (train split)")
    ax1.set_title(title)
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.scatter(xs, [r["latency_ms_per_query"] for r in ok], s=12, alpha=0.5, color="#54a24b")
    ax2.axhline(initial["latency_ms_per_query"], color="gray", ls="--", lw=1,
                label=f"baseline {initial['latency_ms_per_query']:.1f} ms")
    ax2.set_yscale("log")
    ax2.set_ylabel("latency ms/query (log)")
    ax2.set_xlabel("iteration")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    logs = sorted((args.output_dir / "logs").glob("*.log"))
    if len(logs) != 1:
        raise SystemExit(f"expected exactly 1 log in {args.output_dir/'logs'}, found {len(logs)}")
    rows, initial, totals = parse_log(logs[0])
    checkpoint = last_checkpoint(args.output_dir)
    models = load_models(checkpoint)
    for r in rows:
        r["model"] = models.get(r.get("program_id"), "") if r["status"] == "ok" else ""

    args.results_dir.mkdir(parents=True, exist_ok=True)
    cols = ["iteration", "status", "ndcg_at_10", "recall_at_10", "latency_ms_per_query", "combined_score",
            "model", "seconds", "prompt_tokens", "completion_tokens", "program_id", "parent_id", "error"]
    with open(args.results_dir / "iterations.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r["status"] == "ok"]
    err = [r for r in rows if r["status"] == "error"]
    best_info = json.loads((args.output_dir / "best" / "best_program_info.json").read_text())
    errors = {}
    for r in err:
        errors[r["error"]] = errors.get(r["error"], 0) + 1
    milestones = {}
    for target in (0.65, 0.68, 0.69):
        hit = next((r["iteration"] for r in ok if r["ndcg_at_10"] >= target), None)
        milestones[f"first_ndcg_ge_{target}"] = hit
    model_counts = {}
    for r in ok:
        model_counts[r["model"] or "pruned from database"] = model_counts.get(r["model"] or "pruned from database", 0) + 1
    summary = {
        "iterations": len(rows),
        "succeeded": len(ok),
        "failed": len(err),
        "failures_by_error": errors,
        "failed_at_max_tokens": sum(1 for r in err if r.get("completion_tokens") == max(
            (x.get("completion_tokens") or 0) for x in rows)),
        "max_completion_tokens_seen": max((r.get("completion_tokens") or 0) for r in rows),
        "successful_programs_by_model": model_counts,
        "baseline_in_run": initial,
        "best": {"iteration": best_info.get("iteration"), "generation": best_info.get("generation"),
                 "id": best_info.get("id"), **best_info["metrics"]},
        "milestones": milestones,
        "token_usage": totals,
        "checkpoint_used": checkpoint.name,
        "log": logs[0].name,
    }
    (args.results_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    shutil.copy(args.output_dir / "best" / "best_program.py", args.results_dir / "best_program.py")
    shutil.copy(args.output_dir / "best" / "best_program_info.json", args.results_dir / "best_program_info.json")
    plot(rows, initial, args.results_dir / "progress.png", args.title or args.output_dir.parent.name)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
