#!/usr/bin/env python3
"""Re-run validation on an existing corpus directory.

Reads index.json and the per-ticket Markdown files and runs the same
validators the generator runs (section presence, vendor regex, needle
presence). Useful after manual edits or when verifying a corpus that
came from a different generator run.

Usage:
    python tools/recheck.py out/smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make `ticketneedle` importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ticketneedle.types import BonusNeedles, Needles, PrimaryNeedles
from ticketneedle.validate import validate_ticket


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("corpus_dir", type=Path)
    args = p.parse_args()

    index = json.loads((args.corpus_dir / "index.json").read_text())

    failures: list[tuple[str, list[str]]] = []
    for entry in index:
        needles = Needles(
            primary=PrimaryNeedles(
                resolution_steps=entry["primary"]["resolution_steps"]
            ),
            bonus=BonusNeedles(**entry["bonus"]),
        )
        fp = args.corpus_dir / entry["file"]
        if not fp.exists():
            failures.append((entry["ticket_id"], ["file missing"]))
            continue
        errs = validate_ticket(fp.read_text(), needles)
        if errs:
            failures.append((entry["ticket_id"], errs))

    if not failures:
        print(f"OK — all {len(index)} tickets validate.")
        return 0

    print(f"FAIL — {len(failures)}/{len(index)} tickets failed validation:")
    for tid, errs in failures:
        print(f"\n  {tid}:")
        for e in errs:
            print(f"    - {e}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
