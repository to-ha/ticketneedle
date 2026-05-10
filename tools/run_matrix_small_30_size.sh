#!/usr/bin/env bash
# Size-axis benchmark on small_30: 2 qwen3.5 small + 5 gemma4 variants.
# Each model run waits for the corresponding Ollama tag to appear, so this
# script can be started while the gemma4 pulls are still in flight.

cd "$(dirname "$0")/.."

PY=.venv/bin/python
LOG_DIR=out
mkdir -p "$LOG_DIR" results

ORCH_LOG="$LOG_DIR/run_matrix_small_30_size.log"

ts() { date +"%H:%M:%S"; }

# Match an Ollama tag exactly at the start of a `ollama list` line.
wait_for_model() {
    local tag="$1"
    local tag_re=$(printf '%s' "$tag" | sed 's/[][\\.*^$/]/\\&/g')
    while ! ollama list | grep -qE "^${tag_re}[[:space:]]"; do
        echo "[$(ts)] waiting for ollama tag: $tag"
        sleep 30
    done
}

run() {
    local cfg="$1"
    local tag="$2"
    {
        echo
        echo "=== $(ts) WAIT  $tag ==="
    } | tee -a "$ORCH_LOG"
    wait_for_model "$tag"
    {
        echo "=== $(ts) START $cfg ==="
    } | tee -a "$ORCH_LOG"
    $PY bench.py --corpus small_30 --model "$cfg" --k 16 \
        > "$LOG_DIR/bench_small_30__${cfg}.log" 2>&1
    local rc=$?
    {
        echo "=== $(ts) DONE  $cfg (exit $rc) ==="
    } | tee -a "$ORCH_LOG"
}

{
    echo "==================================================================="
    echo "  small_30 SIZE-AXIS matrix — 2 qwen + 5 gemma4 — $(date)"
    echo "==================================================================="
} | tee -a "$ORCH_LOG"

# Smallest first so the fast ones complete while bigger ones are still pulling.
run qwen3.5-4b-128k             qwen3.5:4b-128k
run qwen3.5-9b-128k             qwen3.5:9b-128k
run gemma4-e2b-mxfp8            gemma4:e2b-mxfp8
run gemma4-e4b-mxfp8            gemma4:e4b-mxfp8
run gemma4-26b-mxfp8            gemma4:26b-mxfp8
run gemma4-31b-mxfp8            gemma4:31b-mxfp8
run gemma4-31b-coding-mtp-bf16  gemma4:31b-coding-mtp-bf16

{
    echo
    echo "==================================================================="
    echo "  ALL SIZE-AXIS MODELS DONE @ $(ts)"
    echo "==================================================================="
} | tee -a "$ORCH_LOG"
