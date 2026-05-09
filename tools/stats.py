#!/usr/bin/env python3
"""Print statistics for a generated corpus directory.

Usage:
    python tools/stats.py out/smoke
    python tools/stats.py ../ticketneedle-corpus/medium_80
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("corpus_dir", type=Path)
    args = p.parse_args()

    index_path = args.corpus_dir / "index.json"
    if not index_path.exists():
        print(f"No index.json in {args.corpus_dir}")
        return 1
    index = json.loads(index_path.read_text())

    print(f"=== {args.corpus_dir} ({len(index)} tickets) ===")
    print()

    print("Domain distribution:")
    for d, n in Counter(t["metadata"]["domain"] for t in index).most_common():
        print(f"  {d:15s} {n:3d}  ({n*100//len(index):2d} %)")
    print()

    print("Length bucket distribution:")
    for b, n in sorted(Counter(t["metadata"]["length_bucket"] for t in index).items()):
        print(f"  {b:8s} {n:3d}  ({n*100//len(index):2d} %)")
    print()

    print("Priority distribution:")
    for prio, n in sorted(Counter(t["metadata"]["priority"] for t in index).items()):
        print(f"  {prio:4s} {n:3d}  ({n*100//len(index):2d} %)")
    print()

    print("Cluster distribution:")
    cl = Counter(t["metadata"]["cluster"] for t in index)
    for c, n in sorted(cl.items(), key=lambda x: (x[0] is None, x[0] or "")):
        label = c if c else "(standalone)"
        print(f"  {label:20s} {n:3d}")
    print()

    print("Resolution-step count:")
    for k, n in sorted(
        Counter(len(t["primary"]["resolution_steps"]) for t in index).items()
    ):
        print(f"  {k} steps: {n}")
    print()

    sizes: list[int] = []
    words: list[int] = []
    for t in index:
        fp = args.corpus_dir / t["file"]
        if fp.exists():
            text = fp.read_text()
            sizes.append(len(text))
            words.append(len(text.split()))

    if sizes:
        print("File size:")
        print(
            f"  chars: min={min(sizes)} avg={sum(sizes)//len(sizes)} "
            f"max={max(sizes)}"
        )
        print(
            f"  words: min={min(words)} avg={sum(words)//len(words)} "
            f"max={max(words)}"
        )
        # Rough token estimate: words * 1.4 (English-ish heuristic)
        approx_tokens = int(sum(words) * 1.4)
        print(
            f"Total corpus: {sum(sizes):,} chars / {sum(words):,} words / "
            f"~{approx_tokens:,} tokens"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
