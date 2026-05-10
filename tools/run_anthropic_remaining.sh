#!/usr/bin/env bash
# Anthropic re-run with 1h-TTL cache.
# Order: opus small first (validation $0.60), then medium + large for both
# models. Total estimated cost ~$6 with 1h-TTL cache.
# Polls until any prior bench.py for cloud models has finished.

cd "$(dirname "$0")/.."

PY=.venv/bin/python
LOG_DIR=out
mkdir -p "$LOG_DIR" results

ORCH_LOG="$LOG_DIR/run_anthropic_remaining.log"

ts() { date +"%H:%M:%S"; }

wait_for_idle() {
    while pgrep -f 'bench.py.*claude-' >/dev/null 2>&1; do
        echo "[$(ts)] waiting for prior anthropic bench.py to finish..."
        sleep 30
    done
}

run() {
    local corpus="$1"
    local model="$2"
    {
        echo
        echo "=== $(ts) START $corpus/$model ==="
    } | tee -a "$ORCH_LOG"
    $PY bench.py --corpus "$corpus" --model "$model" --k 16 --pacing 70 \
        > "$LOG_DIR/bench_${corpus}__${model}-1htl.log" 2>&1
    local rc=$?
    {
        echo "=== $(ts) DONE  $corpus/$model (exit $rc) ==="
    } | tee -a "$ORCH_LOG"
}

{
    echo "==================================================================="
    echo "  Anthropic re-run with 1h-TTL cache — $(date)"
    echo "==================================================================="
} | tee -a "$ORCH_LOG"

wait_for_idle

# Validation first: opus small (cheapest opus run, ~$0.60)
run small_30 claude-opus-4-7

# Full re-run order
run medium_80 claude-opus-4-7
run large_150 claude-sonnet-4-6
run large_150 claude-opus-4-7

{
    echo
    echo "==================================================================="
    echo "  ALL ANTHROPIC 1H-TTL RUNS DONE @ $(ts)"
    echo "==================================================================="
} | tee -a "$ORCH_LOG"
