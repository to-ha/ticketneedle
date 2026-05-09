#!/usr/bin/env python3
"""ticketneedle benchmark CLI.

Usage:
    python bench.py --corpus small_30 --model qwen3.6-coding-mxfp8
    python bench.py --corpus small_30 --model llama3.3-70b --k 5
    python bench.py --corpus medium_80 --model gpt-5.1 --k 16
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"


def main() -> int:
    p = argparse.ArgumentParser(description="ticketneedle benchmark runner.")
    p.add_argument("--corpus", required=True,
                   help="corpus name (configs/corpora/<name>.toml) or path to TOML")
    p.add_argument("--model", required=True,
                   help="model name (configs/models/<name>.toml), path to TOML, or raw model id")
    p.add_argument("--k", type=int, default=None,
                   help="how many tickets to sample (default: from corpus config)")
    p.add_argument("--seed", type=int, default=None,
                   help="sample seed (default: from corpus config)")
    p.add_argument("--ticket", action="append",
                   help="run only this ticket id (repeatable). Skips sampling.")
    p.add_argument("--base-url", help="override model base_url")
    p.add_argument("--api-key", help="override API key")
    p.add_argument("--temperature", type=float)
    p.add_argument("--max-tokens", type=int)
    p.add_argument("--timeout", type=float)
    p.add_argument("--pacing", type=float, default=None,
                   help="seconds to sleep between calls (Tier-1 pacing)")
    p.add_argument("--think", action="store_true",
                   help="don't suppress reasoning (default suppresses)")
    p.add_argument("--strict-indent", action="store_true",
                   help="don't relax leading whitespace / numbering on resolution_steps")
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--no-dump", action="store_true",
                   help="skip writing the per-run JSON dump")
    args = p.parse_args()

    from bench.config import auto_dump_path, load_corpus, load_model
    from bench.runner import corpus_from_dir, run_benchmark

    corpus_cfg = load_corpus(args.corpus)
    print(
        f"Corpus config: {corpus_cfg.name} -> {corpus_cfg.directory} "
        f"(k={corpus_cfg.k}, seed={corpus_cfg.seed})",
        flush=True,
    )

    model_cfg, found = load_model(args.model)
    if not found:
        print(
            f"  (no model config '{args.model}'; using as raw model id with local-Ollama defaults)",
            file=sys.stderr,
        )

    if args.base_url:
        model_cfg.client.base_url = args.base_url
    if args.api_key:
        model_cfg.client.api_key = args.api_key
    if args.temperature is not None:
        model_cfg.client.temperature = args.temperature
    if args.max_tokens is not None:
        model_cfg.client.max_tokens = args.max_tokens
    if args.timeout is not None:
        model_cfg.client.timeout = args.timeout

    suppress_thinking = model_cfg.suppress_thinking and not args.think

    k = args.k if args.k is not None else corpus_cfg.k
    seed = args.seed if args.seed is not None else corpus_cfg.seed
    pacing = args.pacing if args.pacing is not None else corpus_cfg.pacing_seconds

    corpus = corpus_from_dir(corpus_cfg.directory)
    dump_path = (
        None
        if args.no_dump
        else auto_dump_path(corpus_cfg.name, model_cfg.name, args.results_dir)
    )

    run_benchmark(
        corpus,
        model_cfg.client,
        k=k,
        seed=seed,
        dump_path=dump_path,
        ticket_filter=args.ticket,
        suppress_thinking=suppress_thinking,
        relax_indent=not args.strict_indent,
        pacing_seconds=pacing,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
