#!/usr/bin/env bash
# Test gemma4 26b in alternative quantizations (mlx-bf16, q4_K_M) to isolate
# whether the silent-failure on 26b-mxfp8 is quant-specific.
# Polls until all earlier bench.py / pull processes have finished.

cd "$(dirname "$0")/.."

PY=.venv/bin/python
LOG_DIR=out
mkdir -p "$LOG_DIR" results

ORCH_LOG="$LOG_DIR/run_26b_alt_quants.log"

ts() { date +"%H:%M:%S"; }

wait_for_idle_bench() {
    while pgrep -f 'bench.py --corpus small_30' >/dev/null 2>&1; do
        echo "[$(ts)] waiting for prior bench.py to finish..."
        sleep 30
    done
}

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
    echo "  small_30 GEMMA4-26b alt-quant runs — $(date)"
    echo "==================================================================="
} | tee -a "$ORCH_LOG"

wait_for_idle_bench

run gemma4-26b-mlx-bf16        gemma4:26b-mlx-bf16
run gemma4-26b-a4b-it-q4_K_M   gemma4:26b-a4b-it-q4_K_M

{
    echo
    echo "==================================================================="
    echo "  ALL 26B ALT-QUANT RUNS DONE @ $(ts)"
    echo "==================================================================="
} | tee -a "$ORCH_LOG"
