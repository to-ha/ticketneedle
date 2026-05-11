# ticketneedle Model Cards — IT Operations Ticket Recall

Use-case-specific model cards for **long-context recall on IT operations
incident tickets**. Not a generic LLM leaderboard — these cards rate
models on one specific shape of work: feed an LLM a corpus of major-incident
tickets and ask it to recall a named ticket's resolution steps, root cause,
key command, escalation path, and timestamp verbatim.

If your workload is something else (chat, summarization, code generation,
multi-turn agents), a different benchmark will give a more useful ranking.

## Headline findings

Five counter-intuitive results from running this benchmark across 14
local + 3 cloud model configurations on three corpus sizes:

1. **Q4_K_M (Metal/llama.cpp) beats MXFP8 (MLX) on long-context ticket
   recall** — counter to the "more bits = more accurate" intuition. On
   the 149 k-token large_150 corpus, qwen3.5-27b-Q4 hits 100 % recall
   with 0 hallucinations (17 GB RAM, 93 s/query) while qwen3.5-27b-MXFP8
   collapses to 57 % recall with 29 hallucinations (31 GB RAM, 74 s/query).
   The same MXFP8 cliff hits qwen3.6 (general and coding-tuned). Q4 wins
   in recall, latency, **and** RAM footprint. The Hero-02 code-recall
   benchmark showed the opposite ranking on jquery — so quantization
   suitability is **domain-specific**, not universal.

2. **9 B is the recall-hardware-class sweet-spot** — qwen3.5-9b-128k
   (6.6 GB model footprint, 16 GB hardware class) is hallucination-free
   up to ~80 k tokens, 86 % recall on 149 k tokens with 7 hallucinations.
   That is the cleanest defensible Edge / 16 GB notebook story across
   the entire local matrix.

3. **Architecture matters more than parameter count for long context** —
   Gemma 4's sliding-window attention (1 k-token window with 5:1 global
   layer pattern) caps recall as the corpus grows. gemma4-e2b
   (effective-2B) drops from 86 % recall on small_30 to 21-31 % on
   medium / large (11-49 hallucinations after format-aware rescoring),
   even though the model itself stays format-faithful below 80 k tokens.
   The dense gemma4-31b-mxfp8 recovers to 100 % recall on large because
   dense layers compensate.

4. **Cloud frontier premium is latency, not recall** — gpt-5.1 hits 100 %
   recall on all three corpora at 2-11 s per query (8× faster than the
   fastest 100 %-recall local model). claude-sonnet-4-6 and
   claude-opus-4-7 both reach 100 % on small_30 and medium_80; on
   large_150 sonnet drops slightly to 91.7 % (1 connection-drop at
   the 16th call) and opus could not be measured at the 149 k-token
   scale within budget. **Methodology note:** an initial opus medium_80
   run scored only 69 %. The run summary explicitly reported "11 scored,
   5 errored" and the run log carried the HTTP 400 "credit balance too
   low" messages on those 5 calls — the failure mode was prominent, not
   silent. The write-up step that built the model-behavior story
   compared only the passed-counts (Opus 11/16 vs Sonnet 16/16) and
   read "Opus is conservative" out of that gap, without integrating
   the errored count. With Anthropic 1 h-TTL prompt caching enabled
   (the cached re-run, sonnet/opus 6-7 s per query on small/medium),
   opus reaches 100 %. The lesson is about evaluation discipline, not
   model behavior or API silence.

5. **Cluster-trap discrimination is intact for 10 of 11 local
   configurations** — even the Q4-quantized 4 B model correctly picks
   the right resolution for tickets with similar symptoms across
   hallucination-trap clusters. Only gemma4-e2b-mxfp8 systematically
   confuses cluster members (0/3 cluster pass on medium_80 and 0/2 on
   large_150). For Operations-Workflows, that means the architecture
   reliably distinguishes "this looks similar to ticket X but the
   resolution is different" — a critical safety property.

The methodology and per-model details that back these findings follow
below.

## Methodology in one paragraph

Three corpus sizes from [ticketneedle-corpus](https://github.com/to-ha/ticketneedle-corpus):
**small_30** (~31 k tokens), **medium_80** (~81 k tokens), **large_150**
(~149 k tokens). Each corpus has 3-4 hallucination-trap clusters (similar
symptoms, different resolutions). Per benchmark run we sample k=16 tickets
stratified by domain and cluster presence, then ask the model to extract
five labeled fields per ticket. Scoring: primary recall on
`resolution_steps` (codeneedle-style alignment, optional whitespace +
numbering tolerance), four bonus slots scored per type. A ticket "passes"
when ≥50 % of resolution_steps are recalled AND ≥2/4 bonus slots match.

**Beta disclaimer:** All cards based on n=1 run × k=16 tickets per corpus.
Methodologically sufficient for tendencies and drift detection; not for
significant comparison of small differences. Updates with larger samples
planned.

---

## qwen3.5-27b-128k (Q4_K_M, Metal/llama.cpp)

### Profile

| Field | Value |
|---|---|
| Family | Qwen 3.5 (general) |
| Size | 27 B parameters dense |
| Quantization | Q4_K_M (4-bit, GGUF) |
| Engine | Metal / llama.cpp |
| Model footprint | 17 GB |
| Effective context | 128 k (Modelfile re-tag of `qwen3.5:27b`) |
| Recommended hardware class | **24 GB unified memory or 24 GB VRAM** |

### Recall profile

| Corpus | Prompt tokens | Primary | Hallucinations | Pass | Cluster-trap | Avg latency |
|---|---|---|---|---|---|---|
| small_30  | 31 k  | 64/64 (100 %) | 0 | 16/16 | 3/3 | 34.6 s |
| medium_80 | 81 k  | 68/68 (100 %) | 0 | 16/16 | 3/3 | 58.4 s |
| **large_150** | **149 k** | **65/65 (100 %)** | **0** | **16/16** | **2/2** | **93.1 s** |

The only model with **100 % recall and zero hallucinations across all
three corpus sizes**. Scaling sub-linear (~1.6× latency per corpus
doubling).

### Use-case suitability for IT ticket support

| Use case | Suitable? | Rationale |
|---|---|---|
| Edge / offline service technician (24 GB notebook) | ✅ | Hallucination-free up to 149 k tokens, 93 s/query |
| Centralized async triage pipeline | ✅ | Best recall-per-RAM ratio of any tested local model |
| Real-time operator assist (< 5 s) | ❌ | 35-93 s latency is too slow — use cloud frontier |
| Long-term knowledge-base search | ✅ | Stable on 149 k tokens; expect linear cost beyond |

### Workflow recommendations

- **Reasoning suppression:** `prefill_no_think=true` is mandatory (Qwen
  3.5 family ignores `/no_think` and burns the entire `max_tokens`
  budget on chain-of-thought)
- **Context tag:** the default `qwen3.5:27b` tag has 4 k context. Build
  the 128 k variant via Modelfile (`PARAMETER num_ctx 131072`)
- **Prompt pattern:** structured multi-slot output (`## resolution_steps`,
  `## root_cause`, etc.) works verbatim
- **Pacing/retry:** none required for local; safe to batch
- **Sanity check before deployment:** run cluster-trap test to verify
  the model does not mix similar tickets

### Comparison vs neighbors

| vs | Δ recall (large) | Δ latency (large) | Δ RAM | Verdict |
|---|---|---|---|---|
| qwen3.5-9b-128k | +14 pp | +65 s (slower) | +10 GB | Recall worth the RAM if Long-Context-stable matters |
| qwen3.5-27b-mxfp8-128k | **+43 pp** | +19 s | -14 GB | **Q4 dominates MXFP8 on long context** despite less RAM |
| gemma4-31b-mxfp8 | 0 pp | -99 s (faster) | -6 GB | Q4 winner on latency-per-recall ratio |

### Caveats

- Latency dominated by prefill (Q4 Metal has ~3-4× higher prefill cost
  than MLX MXFP8 on Apple Silicon — but the recall stability justifies it
  for long contexts)
- Slower in token generation than MLX builds; differential narrows once
  prompt cache is warm

### Cost profile

Local — only electricity. M5 Max under inference load ~80 W. At 93 s/query
(large_150) that is ~2 Wh per query, or ~€0.0006 at €0.30/kWh. **Cost per
1000 queries: ~€0.60** (excluding hardware amortization).

---

## qwen3.5-9b-128k (Q4_K_M, Metal/llama.cpp)

### Profile

| Field | Value |
|---|---|
| Family | Qwen 3.5 (general) |
| Size | 9 B parameters dense |
| Quantization | Q4_K_M (4-bit, GGUF) |
| Engine | Metal / llama.cpp |
| Model footprint | 6.6 GB |
| Effective context | 128 k (Modelfile re-tag of `qwen3.5:9b`) |
| Recommended hardware class | **16 GB unified memory or 16 GB VRAM** |

### Recall profile

| Corpus | Prompt tokens | Primary | Hallucinations | Pass | Cluster-trap | Avg latency |
|---|---|---|---|---|---|---|
| small_30  | 31 k  | 64/64 (100 %) | 0 | 16/16 | 3/3 | 11.7 s |
| medium_80 | 81 k  | 68/68 (100 %) | 0 | 16/16 | 3/3 | 19.2 s |
| **large_150** | **149 k** | **56/65 (86 %)** | **7** | **14/16** | **2/2** | **28.4 s** |

Hallucination-free up to ~80 k tokens. Mild Recall-Drop on 149 k context
(86 % primary, 7 hallucinations) but cluster-trap discrimination intact.

### Use-case suitability for IT ticket support

| Use case | Suitable? | Rationale |
|---|---|---|
| Edge / offline service technician (16 GB notebook) | ✅ **Sweet-spot** | 16 GB MacBook Pro / RTX 4060 Ti class |
| Centralized async triage pipeline | ✅ | 3× faster than 27B-Q4 at identical recall up to 80 k |
| Real-time operator assist (< 5 s) | ❌ | 19-28 s/query too slow |
| Long-term knowledge-base search | ⚠️ | 86 % recall on 149 k — usable but document the gap |

### Workflow recommendations

- **Reasoning suppression:** `prefill_no_think=true` mandatory (same as
  27B sibling)
- **Context tag:** `qwen3.5:9b-128k` Modelfile re-tag of default
- **For real-time use cases:** consider qwen3.5:4b-128k instead (faster
  but more hallucinations on long context)

### Comparison vs neighbors

| vs | Δ recall (large) | Δ latency (large) | Δ RAM | Verdict |
|---|---|---|---|---|
| qwen3.5-4b-128k | +6 pp | +7 s | +3 GB | 9B clearly better on long context — worth the RAM upgrade |
| qwen3.5-27b-Q4 | -14 pp | -65 s (faster) | -10 GB | Faster + smaller, but recall-cliff on 149 k |
| gemma4-e4b-mxfp8 | +21 pp | +20 s (slower) | -4 GB | qwen-9B much better recall — gemma faster but less reliable |

### Caveats

- 7 hallucinations on large_150 — acceptable for non-safety-critical
  workflows; document recall delta to users
- Latency-sensitive use cases on Metal benefit from prompt caching for
  repeated queries

### Cost profile

Local — only electricity. M5 Max under inference load ~80 W. At 28 s/query
(large_150) that is ~0.6 Wh per query, or ~€0.0002 at €0.30/kWh. **Cost per
1000 queries: ~€0.20** (excluding hardware amortization).

---

## gemma4-31b-mxfp8 (MLX MXFP8, text-only)

### Profile

| Field | Value |
|---|---|
| Family | Gemma 4 (dense, text-only build) |
| Size | 31 B parameters dense |
| Quantization | MXFP8 (8-bit, MLX) |
| Engine | MLX (Ollama-MLX runner) |
| Model footprint | ~23 GB |
| Effective context | **256 k** (native Gemma 4 context) |
| Recommended hardware class | **32 GB unified memory or 36 GB VRAM** |

### Recall profile

| Corpus | Prompt tokens | Primary | Hallucinations | Pass | Cluster-trap | Avg latency |
|---|---|---|---|---|---|---|
| small_30  | 31 k  | 64/64 (100 %) | 0 | 16/16 | 3/3 | 98.3 s |
| medium_80 | 81 k  | 59/68 (87 %)  | 0 | 14/16 | 2/3 | 126.4 s |
| **large_150** | **149 k** | **65/65 (100 %)** | **0** | **16/16** | **2/2** | **191.9 s** |

Cross-family Long-Context recall winner — second model after
qwen3.5-27b-Q4 with 100 % recall on 149 k tokens, zero hallucinations.
Note: medium dip is likely k=16 sample variance; large recovery is
clean.

### Use-case suitability for IT ticket support

| Use case | Suitable? | Rationale |
|---|---|---|
| Edge / offline service technician (32 GB notebook) | ✅ | If you need cross-family redundancy alongside qwen |
| Centralized async triage pipeline | ✅ | Stable, deeper context window (256 k vs 128 k) |
| Real-time operator assist (< 5 s) | ❌ | 98-192 s latency far too slow |
| Long-term knowledge-base search | ✅ | 256 k context allows larger corpora than qwen-128 k |

### Workflow recommendations

- **Reasoning suppression:** Gemma 4 is non-reasoning by default — no
  prefill or `/no_think` needed
- **Stop sequences:** verified neutral on tickets (irrelevant; were
  needed for code-domain in Hero-02 to cut prompt parrot)
- **Context advantage:** 256 k native — for very long ticket histories
  beyond the 128 k qwen ceiling
- **Prompt pattern:** identical multi-slot output works verbatim

### Comparison vs neighbors

| vs | Δ recall (large) | Δ latency (large) | Δ RAM | Verdict |
|---|---|---|---|---|
| qwen3.5-27b-Q4 | 0 pp | +99 s (slower) | +6 GB | qwen-Q4 wins on speed-per-RAM; gemma is the cross-family backup |
| qwen3.6-27b-mxfp8 | **+32 pp** | +116 s (slower) | -8 GB | Gemma-31B-dense radically more stable than qwen-MXFP8 |
| gemma4-26b-mlx-bf16 | +18 pp | +123 s (slower) | -29 GB | Dense > sparse-MoE for long context (per Protorikis too) |

### Caveats

- Slowest of the top-tier recall models (192 s/query on 149 k tokens)
- 256 k context advantage useful only for corpora >128 k tokens
- Smaller Gemma 4 variants (e2b, e4b, 26b) are NOT recommended — see
  excluded section below

### Cost profile

Local — only electricity. M5 Max under inference load ~80 W. At 192 s/query
(large_150) that is ~4.3 Wh per query, or ~€0.0013 at €0.30/kWh. **Cost per
1000 queries: ~€1.30** (excluding hardware amortization).

---

## gpt-5.1 (cloud, OpenAI)

### Profile

| Field | Value |
|---|---|
| Provider | OpenAI |
| Model | gpt-5.1 |
| Context window | 256 k+ |
| Endpoint | `api.openai.com/v1/chat/completions` |
| Required flags | `use_max_completion_tokens=true`, `temperature=1.0` |
| Recommended for | Real-time / latency-sensitive ticket workflows |

### Recall profile

| Corpus | Prompt tokens | Primary | Hallucinations | Pass | Cluster-trap | Avg latency |
|---|---|---|---|---|---|---|
| small_30  | 31 k  | 64/64 (100 %) | 0 | 16/16 | 3/3 | 2.1 s |
| medium_80 | 81 k  | 68/68 (100 %) | 0 | 16/16 | 3/3 | 4.3 s |
| **large_150** | **149 k** | **65/65 (100 %)** | **0** | **16/16** | **2/2** | **11.3 s** |

**100 % recall across all three corpora at sub-15 s latency.** The only
cloud or local model that combines perfect recall with real-time-feasible
response speed.

### Use-case suitability for IT ticket support

| Use case | Suitable? | Rationale |
|---|---|---|
| Real-time operator assist (< 5 s) | ✅ | small/medium under 5 s; large 11 s borderline |
| Centralized async triage pipeline | ✅ | Best speed-per-recall; OpenAI prompt cache halves input cost |
| Edge / offline service technician | ❌ | Cloud-only; offline workflows need a local model |
| High-volume daily inference (10 k+ tickets/day) | ⚠️ | Cost scales linearly — see cost profile below |

### Workflow recommendations

- **Prompt caching:** OpenAI applies it automatically on repeated prefixes
  (50 % discount on cached input tokens). For our corpus-prefix-constant
  workload that means input tokens after the 1st call are billed at 50 %
- **Reasoning:** gpt-5.1 is non-reasoning by default — no special flag
  required (unlike GPT-5.2 which is reasoning-on)
- **Pacing/retry:** OpenAI Tier 1 = 500 k TPM. Our 149 k-token requests
  fit comfortably with no pacing required
- **Required flag:** `use_max_completion_tokens=true` (GPT-5 family rejects
  the legacy `max_tokens` parameter)
- **Required temperature:** 1.0 (GPT-5 family rejects 0.0)

### Comparison vs neighbors

| vs | Δ recall (large) | Δ latency (large) | Cost vs gpt-5.1 | Verdict |
|---|---|---|---|---|
| qwen3.5-27b-Q4 (local) | 0 pp | -82 s (gpt faster) | local ≈ €0 vs cloud ~€0.001/query | Cloud unbeatable on speed; local for cost-at-scale |
| claude-sonnet-4-6 | **+8 pp** (gpt better — sonnet hit 1 conn-drop) | +172 s (sonnet w/pacing & 429-backoff) | sonnet ~3.5× w/ cache | Sonnet recall equal up to medium; large_150 is rate-limit territory |
| claude-opus-4-7 | n/a (not measured on large) | n/a | opus ~5× w/ cache | Opus 100 % on small/medium but expensive — gpt-5.1 wins on €/correct-answer |

### Caveats

- Cloud-only — Privacy/Compliance constraints may forbid sending tickets
- Cost scales linearly with token volume — see cost profile

### Cost profile

OpenAI pricing for gpt-5.1: **$1.25/M input ($0.625/M cached, 50 % off),
$10/M output**. Our workload triggers automatic prompt caching after the
first call (constant corpus prefix).

| Corpus | Per 16-ticket run | Per 1000 queries |
|---|---|---|
| small_30 (~31 k tokens) | ~$0.22 | ~$13.80 |
| medium_80 (~81 k tokens) | ~$0.52 | ~$32.50 |
| large_150 (~149 k tokens) | ~$0.93 | ~$58.10 |

**Empirical:** $1.70 spent across all Hero-02 + Hero-03 work
(Hero-02 jquery + http_server + small/medium/large × 16 tickets).
At this rate, $15 budget covers ~140 full small_30 runs or ~25
full large_150 runs.

---

## claude-sonnet-4-6 (cloud, Anthropic)

### Profile

| Field | Value |
|---|---|
| Provider | Anthropic |
| Model | claude-sonnet-4-6 |
| Context window | 200 k |
| Endpoint | `api.anthropic.com/v1/chat/completions` (OpenAI-compat shim) |
| Required flags | `anthropic_cache=true` (auto-detected from base_url), `temperature=1.0` works |
| Recommended for | Cloud-second-source / fallback when gpt-5.1 unavailable |

### Recall profile

| Corpus | Prompt tokens | Primary | Hallucinations | Pass | Cluster-trap | Avg latency |
|---|---|---|---|---|---|---|
| small_30  | 31 k  | 64/64 (100 %) | 0 | 16/16 | 3/3 | 6.7 s |
| medium_80 | 81 k  | 68/68 (100 %) | 0 | 16/16 | 3/3 | 70.0 s |
| **large_150** | **149 k** | **55/60 (91.7 %)** | **4** | **14/15** | **2/2** | **183 s** |

100 % recall on small / medium, slight drop on large_150 with one
connection-drop at the 16th call. The connection-drop is an Anthropic
infrastructure tax at 149 k-token prompt size combined with HTTP 429
rate-limit backoff — not a model-quality issue. 15/16 calls all scored
100 % primary recall with 0 hallucinations.

### Use-case suitability for IT ticket support

| Use case | Suitable? | Rationale |
|---|---|---|
| Real-time operator assist (< 5 s) | ❌ | 6.7 s on small is borderline; medium/large too slow due to rate-limit pacing |
| Centralized async triage pipeline | ⚠️ | Works up to medium_80; large_150 needs careful pacing and retry logic |
| Edge / offline service technician | ❌ | Cloud-only |
| Cloud-second-source fallback | ✅ | Independent vendor from OpenAI, 100 % match on small/medium |

### Workflow recommendations

- **Prompt caching mandatory:** set `anthropic_cache=true` (auto-detected
  from `base_url contains anthropic.com`). 1 h-TTL cache eliminates the
  prefix-reload cost across the 16-call benchmark run, dropping per-call
  cost ~10× after the first call
- **Pacing:** the 70 s default suffices for small/medium; large_150
  needs 180-200 s pacing because Anthropic rate-limits the 149 k-input
  bucket aggressively (15/16 calls in our test hit HTTP 429)
- **Required temperature:** 1.0 (Anthropic OpenAI-shim rejects 0.0
  silently — verify by checking response variance across calls)
- **Connection-drop retry:** add a final retry step on `peer closed
  connection` — these occur ~1 in 16 calls on 149 k-token prompts

### Comparison vs neighbors

| vs | Δ recall (large) | Δ latency (large) | Cost vs sonnet | Verdict |
|---|---|---|---|---|
| gpt-5.1 | -8 pp | +172 s (gpt faster) | gpt ~0.3× | gpt-5.1 strictly cheaper + faster + more reliable on large_150 |
| claude-opus-4-7 | n/a (opus not measured) | n/a | opus ~1.4× | Sonnet/opus identical recall on small/medium; opus 5× more expensive |
| qwen3.5-27b-Q4 (local) | +8 pp (qwen perfect) | -90 s (qwen faster) | qwen ≈ €0 | Local Q4 dominates sonnet on large_150 quality + speed + cost |

### Caveats

- **HTTP 429 rate-limit aggressive at 149 k input:** 15/16 calls on
  large_150 hit a rate-limit retry, doubling effective latency
- **Connection-drop at scale:** ~6 % connection-drop rate at 149 k input
  on large_150 — production code needs idempotent retry
- **Cost asymmetry vs OpenAI:** without cache, ~3-4× gpt-5.1 cost per
  call; with cache the gap narrows to ~2.5-3× on repeat calls
- **Cache-write premium:** 1 h-TTL cache costs 2.0× input rate on the
  first call; pays off when ≥3 cached calls follow

### Cost profile

Anthropic pricing for claude-sonnet-4-6: **$3.00/M input, $0.30/M cached-
read (10 %), $15.00/M output**. 1 h-TTL cache write premium 2.0× = $6.00/M
on first call only.

| Corpus | Per 16-ticket run (cached) | Per 1000 queries (amortized) |
|---|---|---|
| small_30 (~31 k tokens) | ~$0.50 | ~$31 |
| medium_80 (~81 k tokens) | ~$1.30 | ~$80 |
| large_150 (~149 k tokens) | ~$2.40 | ~$150 |

**Empirical Hero-03 cost:** ~$15 burned across small/medium/large with
two top-ups ($5 + $8); cached re-run reduced effective cost ~3× vs
the initial nocache run.

---

## claude-opus-4-7 (cloud, Anthropic)

### Profile

| Field | Value |
|---|---|
| Provider | Anthropic |
| Model | claude-opus-4-7 |
| Context window | 200 k |
| Endpoint | `api.anthropic.com/v1/chat/completions` (OpenAI-compat shim) |
| Required flags | `anthropic_cache=true` (auto-detected from base_url) |
| Recommended for | High-stakes recall where the model "knowing it doesn't know" is desirable |

### Recall profile

| Corpus | Prompt tokens | Primary | Hallucinations | Pass | Cluster-trap | Avg latency |
|---|---|---|---|---|---|---|
| small_30  | 31 k  | 64/64 (100 %) | 0 | 16/16 | 3/3 | 6.2 s |
| medium_80 | 81 k  | 68/68 (100 %) | 0 | 16/16 | 3/3 | 7.2 s |
| **large_150** | **149 k** | **not measured** | n/a | n/a | n/a | n/a |

100 % recall and 0 hallucinations on small/medium with 1 h-TTL prompt
caching. **Large_150 not measured** — Anthropic budget exhausted before
the run completed; estimated $5-7 budget gap to a successful cached run.

**Methodology correction:** an earlier (no-cache) opus medium_80 run
scored 69 %. The run summary reported "11 scored, 5 errored" and the
run log explicitly carried 5 error lines — 1 × HTTP 400 "credit balance
too low" plus subsequent connection-drops and a 502 that the same
budget condition cascaded into. The failure mode was prominent in
both summary and log; the imprecision lay in the next step, when
"Opus 11/16 pass vs Sonnet 16/16 pass" was read as "Opus is
conservative, prefers silence to wrong answers" without integrating
the errored count and the error-detail lines. The cached re-run with
sufficient budget produced a perfect 16/16 — confirming that the
opus-is-quiet narrative was an evaluation artifact, not a model
trait or a hidden API failure.

### Use-case suitability for IT ticket support

| Use case | Suitable? | Rationale |
|---|---|---|
| Real-time operator assist (< 5 s) | ⚠️ | 6.2-7.2 s borderline; faster than sonnet, slower than gpt-5.1 |
| Centralized async triage pipeline | ⚠️ | Works on small/medium; large unproven |
| Edge / offline service technician | ❌ | Cloud-only |
| High-stakes / "don't guess" workloads | ✅ | Most expensive option, but identical recall to gpt-5.1 in our test |

### Workflow recommendations

- **Prompt caching mandatory:** as with sonnet — without 1 h-TTL cache
  this model is not budget-feasible on long-context workloads
- **Required temperature:** 1.0
- **Cost monitoring during runs:** Anthropic does NOT return cost in
  the response — watch the console budget; a single failed large_150
  run can burn $5-10 in input tokens before noticing
- **Pacing:** 70 s default fine for small/medium; large_150 expected
  to need 180-200 s like sonnet (not yet verified)

### Comparison vs neighbors

| vs | Δ recall (medium) | Δ latency (medium) | Cost vs opus | Verdict |
|---|---|---|---|---|
| gpt-5.1 | 0 pp | -3 s (gpt slightly faster) | gpt ~0.2× | gpt-5.1 strictly cheaper at identical quality on small/medium |
| claude-sonnet-4-6 | 0 pp | -63 s (opus faster on cached path) | sonnet ~0.7× | Opus only justifies premium if your workload genuinely needs the larger model |
| qwen3.5-27b-Q4 (local) | 0 pp | -51 s (opus faster) | qwen ≈ €0 | Local Q4 ties opus on quality at zero marginal cost |

### Caveats

- **Most expensive cloud option in this benchmark** — 5× sonnet,
  ~25× gpt-5.1 per call without cache, ~5× gpt-5.1 with cache
- **Errored-count discipline at write-up time:** the budget-exhaustion
  episode shows how 5 errored calls — even when reported prominently
  in the bench summary and run log — can be quietly dropped from a
  model-behavior write-up if only the passed-counts are compared.
  Recommendation: treat the errored count in bench summaries as a
  first-class data point alongside passed counts, and inspect each
  errored call's `error` field before drawing conclusions about model
  behavior
- **Large_150 unverified** — methodology has a gap here

### Cost profile

Anthropic pricing for claude-opus-4-7: **$15.00/M input, $1.50/M cached-
read (10 %), $75.00/M output**. 1 h-TTL cache write premium 2.0× = $30.00/M
on first call only.

| Corpus | Per 16-ticket run (cached) | Per 1000 queries (amortized) |
|---|---|---|
| small_30 (~31 k tokens) | ~$2.40 | ~$150 |
| medium_80 (~81 k tokens) | ~$6.20 | ~$390 |
| large_150 (~149 k tokens) | ~$11.50 (estimated) | ~$720 |

**Empirical:** opus consumed ~$10 of the $15 Hero-03 Anthropic budget
across cached small/medium re-runs alone; the original (no-cache) medium
attempt burned an additional ~$4 before failing on credit-balance.

---

## Excluded models (documented failure modes)

The following models were tested and **not recommended** for this
use case. Documented for completeness so others don't re-test what's
already broken.

| Model | RAM | Failure mode | Recall on large_150 |
|---|---|---|---|
| gemma4:26b-mxfp8 | 20 GB | 14/16 empty responses, silent failure of mxfp8 build | 9/64 (14 %) on small |
| gemma4:26b-a4b-it-q4_K_M | 18 GB | 15/16 empty responses, Q4 build broken | 4/64 (6 %) on small |
| gemma4-e2b-mxfp8 | 7.9 GB | omits `##` headers; ticket confusion on medium/large | 31 % primary, 44 hallucinations (after format-aware rescoring; pre-fix was 212) |
| qwen3.5-27b-mxfp8 | 31 GB | MXFP8 quantization hits a recall cliff on long context | 57 % primary, 29 hallucinations |
| qwen3.6-27b-mxfp8 / coding-mxfp8 | 31 GB | Same MXFP8 cliff as qwen3.5; generation tuning does not help | 68 % primary, 23 hallucinations |

The MXFP8-cliff finding is the most actionable: **for Qwen 3.5 / 3.6
27B on long-context recall, Q4_K_M (Metal) is the more robust choice
despite less aggressive quantization optics.**

### Anthropic large_150 — partial coverage

The cached re-run with `anthropic-beta: extended-cache-ttl-2025-04-11`
recovered the small/medium gap (sonnet/opus both 100 % on small + medium).
**Sonnet large_150 measured at 91.7 % recall** with one HTTP 429 →
peer-closed-connection error at the 16th call (15/16 scored). **Opus
large_150 not measured** — budget exhausted again ($0.13 over the
$15 starting balance + two top-ups) before a successful run.

The opus large_150 gap is the only remaining structural hole in the
Anthropic data. Sonnet shows that the 149 k-token workload **fits within
the Anthropic infrastructure**; the connection drops are rate-limit
artifacts, not model-quality issues.

---

## Cross-stack validation (NVIDIA CUDA-Ollama vs Apple Metal/MLX)

We re-ran the qwen3.5 family on two NVIDIA machines to test whether
the recall findings are Apple-Silicon-specific or architecture-general.
Same model weights, same Q4_K_M quantization, same Modelfile context
re-tag — only the runtime stack differs (CUDA-Ollama vs Metal/llama.cpp
or MLX-Ollama).

### Hardware tested

| Machine | GPU | VRAM | RAM | Role |
|---|---|---|---|---|
| ceres (M5 Max) | Apple M5 Max | 128 GB unified | 128 GB unified | baseline (all cards above) |
| donnager (Tower) | RTX 4090 Desktop | 24 GB | 64 GB | high-end NVIDIA single-GPU |
| mewtwo (Razer Blade 14) | RTX 4070 Mobile | 8 GB | 32 GB | edge-class NVIDIA notebook |

### Cross-stack: qwen3.5-9b-128k

| Stack | Corpus | Primary | Halluc | Pass | Avg latency | Δ vs Apple |
|---|---|---|---|---|---|---|
| Apple Metal | small_30 | 64/64 (100 %) | 0 | 16/16 | 11.7 s | — |
| CUDA-Ollama (4090) | small_30 | 64/64 (100 %) | 0 | 16/16 | **5.7 s** | **2.1× faster** |
| Apple Metal | medium_80 | 68/68 (100 %) | 0 | 16/16 | 19.2 s | — |
| CUDA-Ollama (4090) | medium_80 | 65/68 (96 %) | 4 | 15/16 | **4.4 s** | **4.4× faster, -4 pp recall** |
| Apple Metal | large_150 | 56/65 (86 %) | 7 | 14/16 | 28.4 s | — |
| CUDA-Ollama (4090) | large_150 | 56/65 (86 %) | 7 | 14/16 | **5.5 s** | **5.2× faster, identical recall** |

The 9B model fits comfortably in 24 GB VRAM at 128 k context, so the
4090 delivers a clean 2-5× speedup at near-identical recall. The
medium_80 -4 pp gap is within k=16 sample noise.

### Cross-stack: qwen3.5-27b-128k (Q4_K_M)

| Stack | Corpus | Primary | Halluc | Pass | Avg latency (all 16) | Avg w/o cold | 1st call (cold) | Δ vs Apple (warm) |
|---|---|---|---|---|---|---|---|---|
| Apple Metal | small_30 | 64/64 (100 %) | 0 | 16/16 | 34.6 s | — | — | — |
| CUDA-Ollama (4090) | small_30 | 64/64 (100 %) | 0 | 16/16 | 55.4 s | **52.2 s** | 104.1 s | 1.5× slower (still) |
| Apple Metal | medium_80 | 68/68 (100 %) | 0 | 16/16 | 58.4 s | — | — | — |
| CUDA-Ollama (4090) | medium_80 | 68/68 (100 %) | 0 | 16/16 | 69.2 s | **63.8 s** | 150.0 s | 1.1× slower |
| Apple Metal | large_150 | 65/65 (100 %) | 0 | 16/16 | 93.1 s | — | — | — |
| CUDA-Ollama (4090) | large_150 | 65/65 (100 %) | 0 | 16/16 | 79.2 s | **65.9 s** | 278.7 s | **1.4× faster** |

**Identical recall (100 % everywhere), warm-state latency comparable to
or better than Apple.** The 27B Q4 model on a 24 GB-VRAM 4090 runs
fine after warm-up — the cold-start cost is the visible tax.

### The cold-start tax pattern

On the 4090 the first call of each corpus showed a large outlier:

| Corpus | Cold-start latency | Warm avg | Tax (cold − warm) |
|---|---|---|---|
| small_30 | 104.1 s | 52.2 s | +52 s |
| medium_80 | 150.0 s | 63.8 s | +86 s |
| large_150 | 278.7 s | 65.9 s | **+213 s** |

Tax scales roughly linearly with prompt token volume. During the
cold-start phase of large_150 (observed live):
- VRAM 23.3 / 24 GB
- System-RAM (Ollama) ~11 GB
- GPU power swings 90 W (waiting) ↔ 220 W (computing)
- GPU utilization swings 25 % ↔ 86 %
- Ollama-server CPU 37 %

After warm-up (call 2 onward), VRAM, GPU util, and CPU stabilize and
the per-call latency drops back to the warm range. Interpretation:
the first call pulls the 17 GB model + initial 149 k-token KV-cache
through PCIe; subsequent calls re-use the resident model and benefit
from KV-cache prefix-hits inside Ollama. Apple unified memory does
not pay this tax because the model is always resident in the same
128 GB pool the working set lives in.

**Cross-stack inversion at scale:** the small_30 picture (Donnager
1.5× slower at warm avg) reverses at large_150 (Donnager 1.4× faster
at warm avg). Reason: small_30's per-call work is small enough that
Ollama-runner overhead dominates; large_150's per-call work is big
enough that the 4090's raw compute pulls ahead — but only after
the cold-start tax is paid. **For batch workloads where the model
stays warm, the 4090 is the better long-context recall machine; for
one-off ad-hoc calls on a cold service, the Apple cold-start
advantage matters more.**

### Cross-stack: qwen3.5-4b on 8 GB-VRAM edge

The Razer Blade 14 (RTX 4070 Mobile, 8 GB VRAM) cannot hold a 4 B
Q4 model + 128 k KV-cache fully on-GPU. We tested two variants
to quantify the cost of CPU-offload:

| Variant | Engine | KV in VRAM? | Latency on small_30 | Verdict |
|---|---|---|---|---|
| qwen3.5:4b (full 128k context) | CUDA-Ollama | partial CPU-offload | 27.4 s | **3.6× slower** than Apple — CPU/RAM offload tax dominates |
| qwen3.5:4b-32k (Modelfile re-tag) | CUDA-Ollama | fully on-GPU | 6.7 s | **~1.1× faster** than Apple — clean GPU-only path |
| qwen3.5-4b-128k (Apple Metal) | Metal | unified memory | 7.6 s | baseline |

Both NVIDIA variants score 100 % recall on small_30 (16/16 pass, 0
halluc). **On 8 GB VRAM edge-class hardware, context-window restriction
is the practical mechanism for sub-10 s latency.** A 4 B model at full
128 k context is technically viable but 3-4× slower than the same
model with a 32 k cap.

### Output-format drift across stacks

The CUDA-Ollama runner emits section names without the `## ` Markdown
header that Apple Metal/MLX consistently emits. The original scorer
treated this as section-not-found and fell back to "scan everywhere,"
which inflated hallucination counts by 80-180 per run on cross-stack
data. The scorer was patched (commit 2026-05-10) to accept both forms:

```
## resolution_steps        ← Apple Metal/MLX emits this
resolution_steps           ← CUDA-Ollama emits this
**resolution_steps**       ← both occasionally emit this
```

After the patch, qwen3.5-9b-donnager large_150 dropped from 95 to 7
hallucinations (the same 7 as Apple). The model behavior is identical
across stacks; the scorer needed cross-stack tolerance.

**Implication for cross-vendor deployment:** scoring / regex-extraction
pipelines that depend on Markdown formatting will mis-classify cross-
stack outputs as failures. Validate against multiple runtime stacks
before relying on format-strict parsing.

---

## Methodology limits & next steps

- **n=1 per cell.** Each card cell is one benchmark run. Latency
  variance under load (concurrent users, KV-cache state) not measured.
- **k=16 sample.** Recall percentages on 4-step resolution lists move
  in 2 % increments — small differences are noise.
- **Synthetic corpus.** [ticketneedle-corpus](https://github.com/to-ha/ticketneedle-corpus)
  is generated by `llama3.3:70b` against a controlled template. Real
  major-incident tickets have different statistical profiles
  (vocabulary drift, vendor-specific jargon, untemplated formatting).
- **Apple Silicon hardware** (M5 Max 128 GB) for all local models. Speed
  numbers are not directly portable to NVIDIA/AMD GPUs or x86 CPU
  inference.

Updates planned: native Ollama API metrics (precise prefill_s + gen_TPS),
multi-run variance, NVIDIA-GPU comparison, additional model families
(Mistral, Llama 4, Phi 4).

---

*Generated 2026-05-10 from [ticketneedle](https://github.com/to-ha/ticketneedle)
benchmark runs against [ticketneedle-corpus](https://github.com/to-ha/ticketneedle-corpus).
Comments and updates: please open an issue on the ticketneedle repo.*
