# ticketneedle

Hallucination benchmark for IT operations tickets — applies the
[codeneedle](https://github.com/alexziskind1/codeneedle) methodology
(needle-in-a-codebase) to long-context retrieval over synthetic
major-incident tickets.

## Status

Early work in progress. Generator and benchmark are being built; first
results expected in the LinkedIn Hero-03 post.

## Why

LLMs are increasingly used in IT operations to triage incidents, suggest
resolutions, and summarize ticket history. Existing benchmarks measure
code retrieval (codeneedle) or factual QA, but not how reliably models
extract specific information from long ticket logs — and how often they
hallucinate plausible-sounding but wrong answers when the right
information is buried in a 100k-token corpus.

## Methodology

1. **Synthetic ticket corpus** — see
   [ticketneedle-corpus](https://github.com/to-ha/ticketneedle-corpus).
   Three sizes (30 / 80 / 150 tickets) with embedded *needles* (specific
   resolution steps, root cause, escalation path, command, timestamp)
   and 10–20 % hallucination traps (similar symptoms, different
   resolutions across tickets).
2. **Prompt** — the model sees the full corpus and is asked to recall a
   specific field from a specific ticket.
3. **Scorer** — primary recall on the resolution-step block (multi-line,
   directly comparable to codeneedle's `primary_lines`); bonus slots
   scored per type (text-block / short-string / list / timestamp).
4. **Metrics** — recall, hallucination rate, per-slot accuracy,
   context-length scaling, hallucination-trap discrimination.

## Repo layout

```
templates/      prompts and structure templates fed to the generator
generate.py     corpus generator (Ollama / llama3.3:70b)
bench/          benchmark runner (extractor, scorer, runner) — to come
tools/          validation, stats, plot
```

## Quickstart

The generator targets a local Ollama endpoint (`http://localhost:11434`)
running `llama3.3:70b` (chosen for family-neutrality vs. the planned
benchmark models — Qwen / GPT / Claude — and to keep the corpus
generation pipeline fully local).

```bash
# One-time setup
python3 -m venv .venv
.venv/bin/pip install -e .

# Pull the generator model (one-time, ~42 GB)
ollama pull llama3.3:70b

# Smoke test (1 ticket, ~80s on M5 Max)
.venv/bin/python generate.py --size small --output ./out/smoke --limit 1

# Full small corpus (30 tickets, ~40 min)
.venv/bin/python generate.py --size small --output ../ticketneedle-corpus/small_30

# Inspect
.venv/bin/python tools/stats.py ./out/smoke
.venv/bin/python tools/recheck.py ./out/smoke
```

### Two-stage generation

1. **Phase 1** — a JSON-mode call generates the *needles* for the ticket
   (`resolution_steps`, `root_cause`, `key_command`). Schema-validated,
   vendor-filtered, retried on failure.
2. **Phase 2** — a second call generates the full Markdown ticket body
   around the needles, with strict instructions to embed every needle
   verbatim. The output is post-validated against the index of needles;
   non-conformant tickets are retried.

Escalation paths and incident timestamps are generated deterministically
(no LLM call) from the per-ticket seed, so they are perfectly
reproducible.

### Reproducibility

`generation_seeds.json` (written into the output directory) records the
base seed plus every per-ticket parameter and seed slot, so the corpus
can be regenerated bit-identical.

## Inspiration and credit

This project ports
[Alex Ziskind's codeneedle](https://github.com/alexziskind1/codeneedle)
methodology to the IT-operations domain. Upstream contributions made
during this work: see
[codeneedle PR #3](https://github.com/alexziskind1/codeneedle/pull/3)
(cloud 429-retry with `Retry-After` honoring, `omit_temperature` flag
for GPT-5 / Opus-4.7).

## License

MIT — see [LICENSE](LICENSE).
