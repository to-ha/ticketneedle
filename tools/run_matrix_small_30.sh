#!/usr/bin/env bash
# Run the small_30 benchmark for the LOCAL models in sequence.
# Designed to run unattended overnight: each model failure is logged and the
# script continues with the next model.
#
# CLOUD MODELS (gpt-5.1, claude-sonnet-4-6, claude-opus-4-7) are intentionally
# NOT in this script. Run them manually when awake to monitor for runaway
# cost — see tools/run_matrix_small_30_cloud.sh.
#
# Outputs:
#   results/small_30__<model>.json   — per-run JSON dumps (used by analysis)
#   out/bench_small_30__<model>.log  — per-run console log
#   out/run_matrix_small_30.log      — overall orchestrator log

set -u

cd "$(dirname "$0")/.."

PY=.venv/bin/python
LOG_DIR=out
mkdir -p "$LOG_DIR" results

ORCH_LOG="$LOG_DIR/run_matrix_small_30.log"

ts() { date +"%H:%M:%S"; }

run() {
    local model="$1"
    local model_safe="${model//\//_}"
    {
        echo
        echo "=== $(ts) START  $model ==="
    } | tee -a "$ORCH_LOG"

    $PY bench.py --corpus small_30 --model "$model" --k 16 \
        > "$LOG_DIR/bench_small_30__${model_safe}.log" 2>&1
    local rc=$?

    {
        echo "=== $(ts) DONE   $model  (exit $rc) ==="
    } | tee -a "$ORCH_LOG"
}

{
    echo "==================================================================="
    echo "  small_30 LOCAL benchmark matrix — 4 qwen variants — $(date)"
    echo "==================================================================="
} | tee -a "$ORCH_LOG"

# Three benchmark axes from Hero-02:
#   - quant + engine: qwen3.5-q4 (Metal/Q4_K_M) vs qwen3.5-mxfp8 (MLX/MXFP8)
#   - generation:     qwen3.5-mxfp8 vs qwen3.6-mxfp8
#   - domain tuning:  qwen3.6-mxfp8 vs qwen3.6-coding-mxfp8
# All four model tags use 128k Ollama context (Modelfile re-tag).
run qwen3.5-27b-128k
run qwen3.5-27b-mxfp8-128k
run qwen3.6-27b-mxfp8-128k
run qwen3.6-27b-coding-mxfp8-128k

{
    echo
    echo "==================================================================="
    echo "  ALL LOCAL MODELS DONE @ $(ts)"
    echo "==================================================================="
} | tee -a "$ORCH_LOG"
