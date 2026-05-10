#!/usr/bin/env python3
"""Approximate TPS / prefill-time per model from existing benchmark JSON dumps.

Trick: each ticket-run has the same prompt size (whole corpus + same task
suffix), so prefill cost is constant. Output length varies by ticket. A
linear fit `latency = prefill_seconds + output_tokens / tps_generation`
gives both quantities.

Token counts are approximated as `chars / 4` — good enough to compare
models on the same corpus, not absolute.

Usage:
    python tools/speed_stats.py results/small_30__*.json
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

CHARS_PER_TOKEN = 4.0


def fit_prefill_tps(latencies: list[float], out_tokens: list[float]) -> tuple[float, float]:
    """Least-squares fit: latency = a + b * out_tokens. Returns (a, 1/b)."""
    n = len(latencies)
    if n < 2:
        return latencies[0] if latencies else 0.0, 0.0
    sx = sum(out_tokens)
    sy = sum(latencies)
    sxx = sum(x * x for x in out_tokens)
    sxy = sum(x * y for x, y in zip(out_tokens, latencies))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-9:
        return sy / n, 0.0
    b = (n * sxy - sx * sy) / denom
    a = (sy - b * sx) / n
    tps = 1.0 / b if b > 1e-9 else 0.0
    return a, tps


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    args = p.parse_args()

    all_files: list[Path] = []
    for pat in args.files:
        all_files.extend(Path(p) for p in glob.glob(pat))
    all_files = [f for f in all_files if f.is_file()]

    print(
        f"{'Model':40s} {'in_tok':>8s} {'avg_out_tok':>12s} {'avg_lat':>9s} "
        f"{'naive_TPS':>11s} {'prefill_s':>10s} {'gen_TPS':>9s}"
    )
    print("-" * 105)

    for f in sorted(all_files):
        data = json.load(open(f))
        rs = [r for r in data["results"] if not r.get("error") and r.get("response")]
        if not rs:
            continue
        prompt_tokens = rs[0]["prompt_chars"] / CHARS_PER_TOKEN
        out_toks = [len(r["response"]) / CHARS_PER_TOKEN for r in rs]
        lats = [r["latency_s"] for r in rs]
        avg_out = sum(out_toks) / len(out_toks)
        avg_lat = sum(lats) / len(lats)
        naive_tps = avg_out / avg_lat if avg_lat else 0.0
        prefill_s, gen_tps = fit_prefill_tps(lats, out_toks)
        print(
            f"{data['model']:40s} {prompt_tokens:>8.0f} {avg_out:>12.0f} {avg_lat:>7.1f}s "
            f"{naive_tps:>9.1f} {prefill_s:>9.1f}s {gen_tps:>8.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
