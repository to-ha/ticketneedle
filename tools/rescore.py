#!/usr/bin/env python3
"""Re-score existing benchmark JSON dumps with the current scorer.

Compares old (stored) vs new (current scorer) numbers per file. Does NOT
overwrite the input JSONs — prints the diff and writes summary to stdout.
Pass --update to rewrite the JSONs with new scores in place.

Usage:
    python tools/rescore.py results/*.json
    python tools/rescore.py --update results/*.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.extract import load_corpus
from bench.scorer import score


def aggregate(rs: list[dict]) -> dict:
    n = len(rs)
    return {
        "n": n,
        "primary_matched": sum(r["primary_matched"] for r in rs),
        "primary_total": sum(r["primary_total"] for r in rs),
        "halluc": sum(r["primary_hallucinated"] for r in rs),
        "rc": sum(1 for r in rs if r["root_cause_match"]),
        "kc": sum(1 for r in rs if r["key_command_match"]),
        "esc": sum(1 for r in rs if r["escalation_path_passed"]),
        "ts": sum(1 for r in rs if r["incident_timestamp_match"]),
        "pass": sum(1 for r in rs if r["passed"]),
        "cluster_n": sum(1 for r in rs if r.get("cluster")),
        "cluster_pass": sum(1 for r in rs if r.get("cluster") and r["primary_passed"]),
    }


def fmt_agg(a: dict) -> str:
    return (
        f"primary {a['primary_matched']:>3}/{a['primary_total']:<3} "
        f"halluc={a['halluc']:>4} pass={a['pass']:>2}/{a['n']:<2} "
        f"rc={a['rc']}/{a['n']} kc={a['kc']}/{a['n']} "
        f"esc={a['esc']}/{a['n']} ts={a['ts']}/{a['n']} "
        f"cluster={a['cluster_pass']}/{a['cluster_n']}"
    )


def fmt_delta(old: dict, new: dict) -> str:
    parts = []
    for k in ["primary_matched", "halluc", "rc", "kc", "esc", "ts", "pass", "cluster_pass"]:
        d = new[k] - old[k]
        if d != 0:
            sym = "+" if d > 0 else ""
            parts.append(f"{k}={sym}{d}")
    return "  Δ " + ", ".join(parts) if parts else "  Δ (no change)"


def rescore_file(path: Path, corpus_cache: dict, update: bool) -> tuple[dict, dict]:
    data = json.load(open(path))
    corpus_dir = Path(data["corpus"])
    corpus_dir = corpus_dir.resolve()
    if str(corpus_dir) not in corpus_cache:
        corpus_cache[str(corpus_dir)] = load_corpus(corpus_dir)
    corpus = corpus_cache[str(corpus_dir)]

    target_by_id = {t.ticket_id: t for t in corpus.targets}
    relax_indent = data.get("relax_indent", True)

    old_results = data["results"]
    new_results: list[dict] = []
    for r in old_results:
        if r.get("error") or not r.get("response"):
            new_results.append(r)
            continue
        target = target_by_id.get(r["ticket_id"])
        if target is None:
            new_results.append(r)
            continue
        sc = score(ticket=target, response=r["response"], relax_indent=relax_indent)
        new_r = dict(r)
        new_r["primary_matched"] = sc.primary_matched
        new_r["primary_total"] = sc.primary_total
        new_r["primary_hallucinated"] = sc.primary_hallucinated
        new_r["primary_passed"] = sc.primary_passed
        new_r["root_cause_match"] = sc.root_cause_match
        new_r["key_command_match"] = sc.key_command_match
        new_r["escalation_path_score"] = sc.escalation_path_score
        new_r["escalation_path_passed"] = sc.escalation_path_passed
        new_r["incident_timestamp_match"] = sc.incident_timestamp_match
        new_r["bonus_matched"] = sc.bonus_matched
        new_r["passed"] = sc.passed
        new_results.append(new_r)

    if update:
        data["results"] = new_results
        path.write_text(json.dumps(data, indent=2))
    return aggregate(old_results), aggregate(new_results)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--update", action="store_true",
                   help="rewrite JSONs in place with new scores (default: dry-run, print diff only)")
    p.add_argument("files", nargs="+")
    args = p.parse_args()

    corpus_cache: dict = {}
    paths = sorted({Path(f) for f in args.files})

    print(f"Re-scoring {len(paths)} JSON file(s) — mode: {'UPDATE in place' if args.update else 'dry-run'}\n")

    n_changed = 0
    for path in paths:
        if not path.exists() or not path.suffix == ".json":
            continue
        try:
            old, new = rescore_file(path, corpus_cache, args.update)
        except Exception as e:
            print(f"{path.name}: ERROR — {e}")
            continue

        changed = any(old[k] != new[k] for k in ["primary_matched", "halluc", "pass", "cluster_pass", "rc", "kc", "esc", "ts"])
        if not changed:
            print(f"{path.name}: unchanged  {fmt_agg(new)}")
            continue
        n_changed += 1
        print(f"{path.name}:")
        print(f"  OLD  {fmt_agg(old)}")
        print(f"  NEW  {fmt_agg(new)}")
        print(fmt_delta(old, new))
        print()

    print(f"\nSummary: {n_changed}/{len(paths)} files would change scores")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
