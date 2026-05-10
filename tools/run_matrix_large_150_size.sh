#!/usr/bin/env bash
# Long-context stress test on large_150 (~125k tokens) — full 11-model local matrix.
# Excludes the broken gemma4-26b-mxfp8 and 26b-q4_K_M variants (diagnosed
# unrecoverable on small_30). 26b-mlx-bf16 included as the partially-working
# 26b reference.

cd "$(dirname "$0")/.."

PY=.venv/bin/python
LOG_DIR=out
mkdir -p "$LOG_DIR" results

ORCH_LOG="$LOG_DIR/run_matrix_large_150_size.log"

ts() { date +"%H:%M:%S"; }

wait_for_idle_bench() {
    while pgrep -f 'bench.py --corpus large_150' >/dev/null 2>&1; do
        echo "[$(ts)] waiting for prior large_150 bench.py to finish..."
        sleep 30
    done
}

run() {
    local cfg="$1"
    {
        echo
        echo "=== $(ts) START $cfg ==="
    } | tee -a "$ORCH_LOG"
    $PY bench.py --corpus large_150 --model "$cfg" --k 16 \
        > "$LOG_DIR/bench_large_150__${cfg}.log" 2>&1
    local rc=$?
    {
        echo "=== $(ts) DONE  $cfg (exit $rc) ==="
    } | tee -a "$ORCH_LOG"
}

{
    echo "==================================================================="
    echo "  large_150 LONG-CONTEXT matrix — 11 local models — $(date)"
    echo "==================================================================="
} | tee -a "$ORCH_LOG"

wait_for_idle_bench

# Smallest first so first datapoints arrive within minutes.
run qwen3.5-4b-128k
run qwen3.5-9b-128k
run gemma4-e2b-mxfp8
run gemma4-e4b-mxfp8
run qwen3.5-27b-128k
run gemma4-26b-mlx-bf16
run qwen3.5-27b-mxfp8-128k
run qwen3.6-27b-mxfp8-128k
run qwen3.6-27b-coding-mxfp8-128k
run gemma4-31b-mxfp8
run gemma4-31b-coding-mtp-bf16-nostop

{
    echo
    echo "==================================================================="
    echo "  ALL LARGE_150 MODELS DONE @ $(ts)"
    echo "==================================================================="
} | tee -a "$ORCH_LOG"
