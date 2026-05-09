#!/usr/bin/env python3
"""Generate a synthetic ticket corpus.

Examples:
    python generate.py --size small --output ./out/small_30
    python generate.py --size medium --output ../ticketneedle-corpus/medium_80
    python generate.py --size large  --output ../ticketneedle-corpus/large_150
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from ticketneedle.distribution import build_specs
from ticketneedle.loader import (
    load_clusters,
    load_domains,
    load_personas,
    load_prompt_system,
)
from ticketneedle.needles import build_needles
from ticketneedle.ollama_client import OllamaClient, OllamaOptions
from ticketneedle.render import render_phase2_user
from ticketneedle.types import CORPUS_SIZES
from ticketneedle.validate import validate_ticket


_BAR_WIDTH = 30


def progress_bar(done: int, total: int, width: int = _BAR_WIDTH) -> str:
    """ASCII progress bar like '###########...................'."""
    if total <= 0:
        return "." * width
    filled = min(width, (done * width) // total)
    return "#" * filled + "." * (width - filled)


def progress_prefix(done: int, total: int) -> str:
    """Prefix like ' 4/30 ###########...................' (counter is space-padded so prefix width is stable across the run)."""
    digits = len(str(total))
    return f"{done:>{digits}}/{total} {progress_bar(done, total)}"


def fmt_duration(seconds: float) -> str:
    """Compact human-readable duration: '47s', '5m12s', '1h23m'."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def _insert_root_cause(markdown: str, root_cause: str) -> str:
    """Append root_cause verbatim as a final sentence in the Post-mortem note."""
    if root_cause in markdown:
        return markdown
    marker = "## Post-mortem note"
    if marker not in markdown:
        return markdown.rstrip() + f"\n\n{marker}\n\n{root_cause}\n"
    # Post-mortem is conventionally the last section — append at file end.
    return markdown.rstrip() + f" {root_cause}\n"


def _insert_key_command(markdown: str, key_command: str) -> str:
    """Insert a fenced code block containing key_command into the Diagnosis section."""
    if key_command in markdown:
        return markdown
    marker = "## Diagnosis steps and hypotheses"
    idx = markdown.find(marker)
    if idx < 0:
        return markdown  # bail: no section to insert into
    insertion_point = markdown.find("\n", idx) + 1
    snippet = f"\n```\n{key_command}\n```\n"
    return markdown[:insertion_point] + snippet + markdown[insertion_point:]


def attempt_repair(
    markdown: str,
    needles,  # Needles
    errors: list[str],
):
    """Try to repair single-string verbatim failures (root_cause / key_command).

    Returns (repaired_markdown, remaining_errors, did_repair). Larger
    failures (missing section, vendor name, missing resolution_steps or
    escalation_path) are not repaired — they need a real retry.
    """
    repaired = markdown
    did_repair = False
    for err in errors:
        if err.startswith("root_cause not verbatim"):
            repaired = _insert_root_cause(repaired, needles.bonus.root_cause)
            did_repair = True
        elif err.startswith("key_command not verbatim"):
            repaired = _insert_key_command(repaired, needles.bonus.key_command)
            did_repair = True
    remaining = validate_ticket(repaired, needles)
    return repaired, remaining, did_repair


def main() -> int:
    p = argparse.ArgumentParser(description="Generate a ticketneedle corpus.")
    p.add_argument("--size", choices=list(CORPUS_SIZES.keys()), required=True)
    p.add_argument("--output", type=Path, required=True, help="Output directory")
    p.add_argument("--model", default="llama3.3:70b")
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--num-ctx", type=int, default=16384)
    p.add_argument("--limit", type=int, default=None, help="Stop after N tickets")
    p.add_argument(
        "--skip-existing", action="store_true",
        help="Skip tickets whose output file already exists",
    )
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    domains = load_domains()
    clusters = load_clusters()
    personas = load_personas()
    system_prompt = load_prompt_system()
    name_pool = list(personas["first_names"])

    client = OllamaClient(model=args.model, base_url=args.ollama_url)

    specs = build_specs(
        size=args.size, domains=domains, clusters=clusters, seed=args.seed
    )
    if args.limit is not None:
        specs = specs[: args.limit]

    print(
        f"Plan: {len(specs)} tickets, size={args.size}, seed={args.seed}, "
        f"model={args.model}",
        flush=True,
    )

    seeds_log = {
        "size": args.size,
        "base_seed": args.seed,
        "model": args.model,
        "n_tickets": len(specs),
        "specs": [
            {
                "ticket_id": s.ticket_id,
                "domain": s.domain_key,
                "length_bucket": s.length_bucket,
                "priority": s.priority,
                "cluster": s.cluster_name,
                "cluster_variant": s.cluster_variant,
                "ticket_seed": args.seed + i,
            }
            for i, s in enumerate(specs)
        ],
    }
    (args.output / "generation_seeds.json").write_text(
        json.dumps(seeds_log, indent=2) + "\n"
    )

    index_records: list[dict] = []
    failures: list[dict] = []
    t_start = time.time()

    for i, spec in enumerate(specs):
        ticket_seed = args.seed + i
        out_path = args.output / f"{spec.ticket_id}.md"

        if args.skip_existing and out_path.exists():
            print(f"[{i+1}/{len(specs)}] {spec.ticket_id} — exists, skipping",
                  flush=True)
            continue

        domain = domains[spec.domain_key]
        cluster = clusters[spec.cluster_name] if spec.cluster_name else None

        elapsed = time.time() - t_start
        eta_str = (
            fmt_duration((elapsed / i) * (len(specs) - i)) if i > 0 else "?"
        )
        prefix_start = progress_prefix(i, len(specs))
        print(
            f"{prefix_start} start {spec.ticket_id} "
            f"{spec.domain_key}/{spec.length_bucket} "
            f"cluster={spec.cluster_name or '-'}  "
            f"elapsed {fmt_duration(elapsed)}, eta {eta_str}",
            flush=True,
        )

        try:
            needles, focus = build_needles(
                spec=spec,
                domain=domain,
                cluster=cluster,
                client=client,
                personas=personas,
                base_seed=args.seed,
                ticket_index=i,
            )
        except RuntimeError as e:
            print(f"  ERROR phase-1: {e}", flush=True)
            failures.append(
                {"ticket_id": spec.ticket_id, "phase": "phase-1", "error": str(e)}
            )
            continue

        user_prompt = render_phase2_user(
            spec=spec,
            domain=domain,
            cluster=cluster,
            needles=needles,
            focus=focus,
            name_pool=name_pool,
        )

        body_opts = OllamaOptions(
            temperature=0.7,
            seed=ticket_seed,
            num_ctx=args.num_ctx,
            top_p=0.9,
        )

        markdown = ""
        validation_errors: list[str] = []
        for attempt in range(args.max_retries):
            body_opts.seed = ticket_seed + 999_000 + attempt * 1000
            markdown = client.chat(
                system=system_prompt,
                user=user_prompt,
                options=body_opts,
            )
            validation_errors = validate_ticket(markdown, needles)
            if not validation_errors:
                break
            print(
                f"  retry {attempt+1}/{args.max_retries}: "
                f"{len(validation_errors)} validation errors "
                f"(first: {validation_errors[0]})",
                flush=True,
            )
        else:
            # All retries failed. Last-resort: try to repair single-string
            # verbatim failures (root_cause / key_command) by direct insertion.
            repaired, remaining, did_repair = attempt_repair(
                markdown, needles, validation_errors
            )
            if did_repair and not remaining:
                print(
                    f"  repaired {len(validation_errors)} verbatim issues "
                    f"by direct insertion",
                    flush=True,
                )
                markdown = repaired
                validation_errors = []
            else:
                print(
                    f"  FAILED phase-2 after {args.max_retries} attempts "
                    f"(repair did not resolve all errors)",
                    flush=True,
                )
                failures.append(
                    {
                        "ticket_id": spec.ticket_id,
                        "phase": "phase-2",
                        "errors": validation_errors,
                        "remaining_after_repair": remaining,
                        "raw_tail": markdown[-500:],
                    }
                )
                continue

        out_path.write_text(markdown)
        index_records.append(
            {
                "ticket_id": spec.ticket_id,
                "primary": {"resolution_steps": needles.primary.resolution_steps},
                "bonus": asdict(needles.bonus),
                "metadata": {
                    "domain": spec.domain_key,
                    "length_bucket": spec.length_bucket,
                    "priority": spec.priority,
                    "cluster": spec.cluster_name,
                },
                "file": out_path.name,
            }
        )

        elapsed = time.time() - t_start
        eta = (elapsed / (i + 1)) * (len(specs) - i - 1)
        words = len(markdown.split())
        prefix_done = progress_prefix(i + 1, len(specs))
        print(
            f"{prefix_done} done  {spec.ticket_id} "
            f"{len(markdown)}c {words}w  "
            f"elapsed {fmt_duration(elapsed)}, eta {fmt_duration(eta)}",
            flush=True,
        )

    (args.output / "index.json").write_text(
        json.dumps(index_records, indent=2) + "\n"
    )
    if failures:
        (args.output / "failures.json").write_text(
            json.dumps(failures, indent=2) + "\n"
        )

    total = time.time() - t_start
    print()
    print(
        f"Done in {fmt_duration(total)}. "
        f"{len(index_records)}/{len(specs)} ok, {len(failures)} failed."
    )
    print(f"Output: {args.output}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
