#!/usr/bin/env bash
# Cloud-Modelle auf medium_80 + large_150.
# Anthropic Tier-1-Pacing 70s; gpt-5.1 ohne Pacing.
# Manuell zu starten — Cost-Watch (vermutlich ~$5-15 gesamt).

cd "$(dirname "$0")/.."

PY=.venv/bin/python
LOG_DIR=out
mkdir -p "$LOG_DIR" results

ORCH_LOG="$LOG_DIR/run_matrix_cloud_medium_large.log"

ts() { date +"%H:%M:%S"; }

run() {
    local corpus="$1"
    local model="$2"
    local pacing="${3:-0}"
    local pacing_arg=()
    if [ "$pacing" != "0" ]; then
        pacing_arg=(--pacing "$pacing")
    fi
    local model_safe="${model//\//_}"
    {
        echo
        echo "=== $(ts) START $corpus/$model (pacing=$pacing) ==="
    } | tee -a "$ORCH_LOG"
    $PY bench.py --corpus "$corpus" --model "$model" --k 16 "${pacing_arg[@]}" \
        > "$LOG_DIR/bench_${corpus}__${model_safe}.log" 2>&1
    local rc=$?
    {
        echo "=== $(ts) DONE  $corpus/$model (exit $rc) ==="
    } | tee -a "$ORCH_LOG"
}

{
    echo "==================================================================="
    echo "  Cloud × medium_80 + large_150 — 3 models × 2 corpora — $(date)"
    echo "==================================================================="
} | tee -a "$ORCH_LOG"

# medium_80 first (smaller, faster). gpt fastest, then sonnet, then opus.
run medium_80 gpt-5.1            0
run medium_80 claude-sonnet-4-6  70
run medium_80 claude-opus-4-7    70

# large_150 — same order
run large_150 gpt-5.1            0
run large_150 claude-sonnet-4-6  70
run large_150 claude-opus-4-7    70

{
    echo
    echo "==================================================================="
    echo "  ALL CLOUD MEDIUM+LARGE DONE @ $(ts)"
    echo "==================================================================="
} | tee -a "$ORCH_LOG"
