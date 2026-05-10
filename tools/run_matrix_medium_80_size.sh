#!/usr/bin/env bash
# Size-axis benchmark on medium_80 — extends the local matrix:
#   2 small qwen + 4 working gemma4 variants + 1 alt-quant gemma4-26b
# Excluded: gemma4-26b-mxfp8 + 26b-q4_K_M (both diagnosed as broken on small_30).
#
# Polls until any prior bench.py for medium_80 has finished.

cd "$(dirname "$0")/.."

PY=.venv/bin/python
LOG_DIR=out
mkdir -p "$LOG_DIR" results

ORCH_LOG="$LOG_DIR/run_matrix_medium_80_size.log"

ts() { date +"%H:%M:%S"; }

wait_for_idle_bench() {
    while pgrep -f 'bench.py --corpus medium_80' >/dev/null 2>&1; do
        echo "[$(ts)] waiting for prior medium_80 bench.py to finish..."
        sleep 30
    done
}

run() {
    local cfg="$1"
    {
        echo
        echo "=== $(ts) START $cfg ==="
    } | tee -a "$ORCH_LOG"
    $PY bench.py --corpus medium_80 --model "$cfg" --k 16 \
        > "$LOG_DIR/bench_medium_80__${cfg}.log" 2>&1
    local rc=$?
    {
        echo "=== $(ts) DONE  $cfg (exit $rc) ==="
    } | tee -a "$ORCH_LOG"
}

{
    echo "==================================================================="
    echo "  medium_80 SIZE-AXIS matrix — 7 small/cross-family models — $(date)"
    echo "==================================================================="
} | tee -a "$ORCH_LOG"

wait_for_idle_bench

# Smallest first so we get first datapoints fast.
run qwen3.5-4b-128k
run qwen3.5-9b-128k
run gemma4-e2b-mxfp8
run gemma4-e4b-mxfp8
run gemma4-26b-mlx-bf16
run gemma4-31b-mxfp8
run gemma4-31b-coding-mtp-bf16-nostop  # confirmed equivalent to with-stop on small_30

{
    echo
    echo "==================================================================="
    echo "  ALL MEDIUM_80 SIZE-AXIS MODELS DONE @ $(ts)"
    echo "==================================================================="
} | tee -a "$ORCH_LOG"
