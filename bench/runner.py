"""Orchestrate a benchmark run on a ticket corpus."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .client import ClientConfig, chat_complete
from .extract import TicketCorpus, TicketTarget, load_corpus, stratified_sample
from .report import render_summary, render_ticket
from .scorer import TicketScore, score


# Corpus first, task suffix last — same KV-cache-friendly shape as codeneedle.
# We ask for five fields in one call (multi-slot). The scorer matches each
# needle by substring in the full response, so the model is free to format
# the sections however it likes as long as each value appears verbatim.
PROMPT_TEMPLATE = (
    "{corpus_text}\n"
    "\n"
    "---\n"
    "\n"
    "Task: from the corpus above, locate ticket `{ticket_id}` and extract "
    "the following five fields exactly as they appear in that ticket. "
    "Output them under the five labeled headings shown below, in this exact "
    "order, and reproduce each value verbatim.\n"
    "\n"
    "## resolution_steps\n"
    "(the numbered list of steps from that ticket's `## Resolution steps` "
    "section, verbatim, in the original order)\n"
    "\n"
    "## root_cause\n"
    "(the one-sentence root-cause statement from that ticket's "
    "`## Post-mortem note` section, verbatim, including its final period)\n"
    "\n"
    "## key_command\n"
    "(the diagnostic command from the fenced code block inside that ticket's "
    "`## Diagnosis steps and hypotheses` section, verbatim)\n"
    "\n"
    "## escalation_path\n"
    "(the bulleted list from that ticket's `## Escalation timeline` section, "
    "verbatim, in the original order — each entry is `HH:MM — Role: Name`)\n"
    "\n"
    "## incident_timestamp\n"
    "(the `Reported` value from that ticket's header, verbatim, in the form "
    "`YYYY-MM-DD HH:MM UTC`)\n"
    "\n"
    "Rules:\n"
    "- Output ONLY the five labeled sections above, in that order.\n"
    "- Reproduce each value verbatim — do NOT paraphrase, do NOT shorten.\n"
    "- Do NOT add commentary, do NOT add markdown code fences around the whole reply.\n"
    "- If the ticket does not exist or a field cannot be found, output the literal "
    "string `NOT_FOUND` for that section.\n"
    "{thinking_suffix}"
)

NO_THINK_SUFFIX = "\n/no_think\n"


@dataclass
class _Run:
    ticket_id: str
    prompt_chars: int
    response: str
    latency_s: float
    error: str | None = None


def _build_prompt(target: TicketTarget, corpus_text: str, suppress_thinking: bool) -> str:
    return PROMPT_TEMPLATE.format(
        corpus_text=corpus_text,
        ticket_id=target.ticket_id,
        thinking_suffix=NO_THINK_SUFFIX if suppress_thinking else "",
    )


def run_benchmark(
    corpus: TicketCorpus,
    cfg: ClientConfig,
    *,
    k: int = 16,
    seed: int = 42,
    dump_path: Path | None = None,
    ticket_filter: list[str] | None = None,
    suppress_thinking: bool = True,
    relax_indent: bool = True,
    pacing_seconds: float = 0.0,
) -> list[TicketScore]:
    text = corpus.text
    print(
        f"Corpus: {corpus.display_name}  ({len(text):,} chars, "
        f"{len(corpus.targets)} tickets)",
        flush=True,
    )

    if ticket_filter:
        wanted = set(ticket_filter)
        chosen = [t for t in corpus.targets if t.ticket_id in wanted]
        missing = wanted - {t.ticket_id for t in chosen}
        if missing:
            print(f"WARNING: requested but not found: {sorted(missing)}", flush=True)
    else:
        chosen = stratified_sample(corpus.targets, k=k, seed=seed)

    print(f"Selected {len(chosen)} ticket(s):", flush=True)
    for t in chosen:
        cl = f" [{t.cluster}]" if t.cluster else ""
        print(f"  - {t.ticket_id}  {t.domain}/{t.length_bucket}{cl}", flush=True)

    scores: list[TicketScore] = []
    runs: list[_Run] = []
    t_start = time.time()

    for i, t in enumerate(chosen, 1):
        prompt = _build_prompt(t, text, suppress_thinking)
        print(
            f"\n[{i}/{len(chosen)}] {t.ticket_id} — prompt {len(prompt):,} chars, "
            f"waiting on model...",
            flush=True,
        )
        start = time.monotonic()
        request_error: str | None = None
        try:
            resp = chat_complete(cfg, system=None, user=prompt)
        except Exception as e:
            request_error = str(e)
            print(f"  ERROR: {request_error}", flush=True)
            resp = ""
        latency = time.monotonic() - start
        print(f"  response: {len(resp)} chars in {latency:.1f}s", flush=True)

        if resp.strip() == "" and request_error is None:
            request_error = (
                "empty response (200 OK but no content; reasoning models often "
                "need more max_tokens — try --max-tokens 8000)"
            )
            print(f"  warn: {request_error}", flush=True)

        sc = score(ticket=t, response=resp, relax_indent=relax_indent)
        if request_error:
            sc.error = request_error
        scores.append(sc)
        runs.append(
            _Run(
                ticket_id=t.ticket_id,
                prompt_chars=len(prompt),
                response=resp,
                latency_s=latency,
                error=request_error,
            )
        )
        print(render_ticket(sc), flush=True)

        if pacing_seconds > 0 and i < len(chosen):
            print(f"  pacing {pacing_seconds:.0f}s...", flush=True)
            time.sleep(pacing_seconds)

    print(render_summary(scores), flush=True)
    total = time.time() - t_start
    print(f"\nDone in {total/60:.1f}m.", flush=True)

    if dump_path:
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "corpus": str(corpus.corpus_dir),
            "model": cfg.model,
            "base_url": cfg.base_url,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "relax_indent": relax_indent,
            "k": k,
            "seed": seed,
            "results": [
                {
                    "ticket_id": sc.ticket_id,
                    "domain": sc.domain,
                    "cluster": sc.cluster,
                    "passed": sc.passed,
                    "error": sc.error,
                    "primary_matched": sc.primary_matched,
                    "primary_total": sc.primary_total,
                    "primary_hallucinated": sc.primary_hallucinated,
                    "primary_passed": sc.primary_passed,
                    "root_cause_match": sc.root_cause_match,
                    "key_command_match": sc.key_command_match,
                    "escalation_path_score": sc.escalation_path_score,
                    "escalation_path_passed": sc.escalation_path_passed,
                    "incident_timestamp_match": sc.incident_timestamp_match,
                    "bonus_matched": sc.bonus_matched,
                    "latency_s": r.latency_s,
                    "prompt_chars": r.prompt_chars,
                    "response": r.response,
                }
                for sc, r in zip(scores, runs)
            ],
        }
        dump_path.write_text(json.dumps(payload, indent=2))
        print(f"Results dumped to {dump_path}", flush=True)

    return scores


def corpus_from_dir(path: Path) -> TicketCorpus:
    return load_corpus(path)
