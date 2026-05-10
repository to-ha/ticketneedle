#!/usr/bin/env bash
# Cloud half of the small_30 benchmark matrix.
#
# RUN THIS MANUALLY (not from an unattended overnight script) so cost
# can be monitored. Total expected cost: ~$1-3 (gpt-5.1 ~$0.30,
# sonnet ~$0.50, opus ~$1.50).
#
# Anthropic Tier 1 = 30k ITPM. small_30 prompt is ~30k tokens, so pace
# 70s between calls (handled per-model below).
#
# Outputs same shape as the local matrix script.

cd "$(dirname "$0")/.."

PY=.venv/bin/python
LOG_DIR=out
mkdir -p "$LOG_DIR" results

ORCH_LOG="$LOG_DIR/run_matrix_small_30_cloud.log"

ts() { date +"%H:%M:%S"; }

run() {
    local model="$1"
    local pacing="${2:-0}"
    local pacing_arg=()
    if [ "$pacing" != "0" ]; then
        pacing_arg=(--pacing "$pacing")
    fi
    local model_safe="${model//\//_}"
    {
        echo
        echo "=== $(ts) START  $model  (pacing=$pacing) ==="
    } | tee -a "$ORCH_LOG"

    $PY bench.py --corpus small_30 --model "$model" --k 16 "${pacing_arg[@]}" \
        > "$LOG_DIR/bench_small_30__${model_safe}.log" 2>&1
    local rc=$?

    {
        echo "=== $(ts) DONE   $model  (exit $rc) ==="
    } | tee -a "$ORCH_LOG"
}

{
    echo "==================================================================="
    echo "  small_30 CLOUD benchmark matrix — 3 cloud models — $(date)"
    echo "==================================================================="
} | tee -a "$ORCH_LOG"

run gpt-5.1            0
run claude-sonnet-4-6  70
run claude-opus-4-7    70

{
    echo
    echo "==================================================================="
    echo "  ALL CLOUD MODELS DONE @ $(ts)"
    echo "==================================================================="
} | tee -a "$ORCH_LOG"
