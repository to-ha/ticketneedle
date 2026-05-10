#!/usr/bin/env bash
# Re-run the gemma4 26b / 31b / 31b-coding-bf16 variants on small_30 without
# the stop=["\n---", ...] sequences. Started after the main size-axis run is
# done (it polls for the previous bench.py to exit before starting its own).

cd "$(dirname "$0")/.."

PY=.venv/bin/python
LOG_DIR=out
mkdir -p "$LOG_DIR" results

ORCH_LOG="$LOG_DIR/run_size_nostop_followup.log"

ts() { date +"%H:%M:%S"; }

wait_for_idle_bench() {
    while pgrep -f 'bench.py --corpus small_30 --model gemma4-3' >/dev/null 2>&1; do
        echo "[$(ts)] waiting for current size-axis bench.py to finish..."
        sleep 30
    done
}

run() {
    local cfg="$1"
    {
        echo
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
    echo "  small_30 NOSTOP follow-up — gemma4 26b/31b/coding-bf16 — $(date)"
    echo "==================================================================="
} | tee -a "$ORCH_LOG"

wait_for_idle_bench

run gemma4-26b-mxfp8-nostop
run gemma4-31b-mxfp8-nostop
run gemma4-31b-coding-mtp-bf16-nostop

{
    echo
    echo "==================================================================="
    echo "  ALL NOSTOP FOLLOW-UPS DONE @ $(ts)"
    echo "==================================================================="
} | tee -a "$ORCH_LOG"
