#!/usr/bin/env bash
# Cross-stack validation on donnager (RTX 4090, 24 GB VRAM).
# 9B model first (lighter, faster, less risk of VRAM pressure).
# 27B with 128k context may pressure 24 GB VRAM at large_150 — watch
# for latency cliff if KV-cache overflows to CPU/RAM.

cd "$(dirname "$0")/.."

PY=.venv/bin/python
LOG_DIR=out
mkdir -p "$LOG_DIR" results

ORCH_LOG="$LOG_DIR/run_donnager_cross_stack.log"

ts() { date +"%H:%M:%S"; }

run() {
    local corpus="$1"
    local cfg="$2"
    {
        echo
        echo "=== $(ts) START $corpus/$cfg ==="
    } | tee -a "$ORCH_LOG"
    $PY bench.py --corpus "$corpus" --model "$cfg" --k 16 \
        > "$LOG_DIR/bench_${corpus}__${cfg}.log" 2>&1
    local rc=$?
    {
        echo "=== $(ts) DONE  $corpus/$cfg (exit $rc) ==="
    } | tee -a "$ORCH_LOG"
}

{
    echo "==================================================================="
    echo "  donnager (RTX 4090) cross-stack matrix — $(date)"
    echo "==================================================================="
} | tee -a "$ORCH_LOG"

# 9B first across all three corpus sizes
run small_30  qwen3.5-9b-donnager
run medium_80 qwen3.5-9b-donnager
run large_150 qwen3.5-9b-donnager

# Then 27B
run small_30  qwen3.5-27b-donnager
run medium_80 qwen3.5-27b-donnager
run large_150 qwen3.5-27b-donnager

{
    echo
    echo "==================================================================="
    echo "  ALL DONNAGER RUNS DONE @ $(ts)"
    echo "==================================================================="
} | tee -a "$ORCH_LOG"
