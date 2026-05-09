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

To be filled as the tool stabilizes. The generator targets a local
Ollama endpoint (`http://localhost:11434`) running `llama3.3:70b`.

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
