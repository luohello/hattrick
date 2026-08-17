#!/usr/bin/env bash
set -Eeuo pipefail

# Run both GEANT evaluations with the trained 4-path Hattrick model:
#   1) normalized MLU
#   2) admitted-flow / fulfill ratio

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="${HATTRICK_VENV:-/mnt/data0/helo}"
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "Virtual environment not found: $VENV_DIR" >&2
    exit 1
fi
source "$VENV_DIR/bin/activate"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# PyTorch deterministic mode requires one of these CuBLAS workspace settings.
# It must be exported before the Python process starts.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1

TEST_START="${TEST_START:-8618}"
TEST_END="${TEST_END:-10772}"
TEST_CLUSTER="${TEST_CLUSTER:-0}"
NUM_PATHS="${NUM_PATHS:-4}"
PRED_TYPE="${PRED_TYPE:-esm}"

RESULT_DIR="results/geant/${NUM_PATHS}sp/${TEST_CLUSTER}"
EXPECTED_SNAPSHOTS=$((TEST_END - TEST_START))
EXPECTED_VALUE_LINES=$((EXPECTED_SNAPSHOTS * 3))

mkdir -p "$RESULT_DIR" logs
LOG_FILE="logs/geant_test_both_$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

trap 'echo "Inference failed near line $LINENO. See: $LOG_FILE" >&2' ERR

run_inference() {
    local sim_mf_mlu="$1"
    local label="$2"
    local values_file="$RESULT_DIR/hattrick_values_${PRED_TYPE}_sim_mlu_${sim_mf_mlu}.txt"
    local stats_file="$RESULT_DIR/hattrick_stats_${PRED_TYPE}_sim_mlu_${sim_mf_mlu}.txt"
    local runtime_file="$RESULT_DIR/hattrick_runtime.txt"
    local saved_runtime="$RESULT_DIR/hattrick_runtime_${label}.txt"

    echo
    echo "Starting GEANT ${label} inference"
    echo "Test range: [${TEST_START}, ${TEST_END}), snapshots: ${EXPECTED_SNAPSHOTS}"

    python run_hattrick.py \
        --topo geant \
        --mode test \
        --test_start_idx "$TEST_START" \
        --test_end_idx "$TEST_END" \
        --test_cluster "$TEST_CLUSTER" \
        --num_paths_per_pair "$NUM_PATHS" \
        --rau1 3 \
        --rau2 3 \
        --rau3 3 \
        --pred 1 \
        --dynamic 0 \
        --pred_type "$PRED_TYPE" \
        --dtype float32 \
        --violation 1 \
        --sim_mf_mlu "$sim_mf_mlu"

    if [[ ! -f "$values_file" || ! -f "$stats_file" || ! -f "$runtime_file" ]]; then
        echo "Expected output file is missing after ${label} inference." >&2
        exit 1
    fi

    cp -f "$runtime_file" "$saved_runtime"

    local value_lines runtime_lines
    value_lines="$(wc -l < "$values_file")"
    runtime_lines="$(wc -l < "$saved_runtime")"

    if [[ "$value_lines" -ne "$EXPECTED_VALUE_LINES" ]]; then
        echo "Unexpected values line count: ${value_lines}; expected ${EXPECTED_VALUE_LINES}." >&2
        exit 1
    fi
    if [[ "$runtime_lines" -ne "$EXPECTED_SNAPSHOTS" ]]; then
        echo "Unexpected runtime line count: ${runtime_lines}; expected ${EXPECTED_SNAPSHOTS}." >&2
        exit 1
    fi

    echo "Completed ${label} inference"
    echo "Values:  $values_file (${value_lines} lines)"
    echo "Stats:   $stats_file"
    echo "Runtime: $saved_runtime (${runtime_lines} lines)"
}

# sim_mf_mlu=0: evaluate normalized MLU on the real traffic matrices.
run_inference 0 mlu

# sim_mf_mlu=1: evaluate admitted flow / fulfill ratio with the simulator.
run_inference 1 flow

echo
echo "Both GEANT inference runs completed successfully."
echo "Log: $LOG_FILE"
