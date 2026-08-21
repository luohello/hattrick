#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source "${HATTRICK_VENV:-/mnt/data0/helo}/bin/activate"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONUNBUFFERED=1

TEST_START="${TEST_START:-8618}"
TEST_END="${TEST_END:-10772}"
RESULT_DIR="results/geant/8sp/0"
EXPECTED_SNAPSHOTS=$((TEST_END - TEST_START))
EXPECTED_VALUE_LINES=$((EXPECTED_SNAPSHOTS * 3))

common=(
    --topo geant --mode test
    --test_start_idx "$TEST_START" --test_end_idx "$TEST_END" --test_cluster 0
    --num_paths_per_pair 8 --rau1 3 --rau2 3 --rau3 3
    --pred 1 --pred_type esm --dynamic 0 --dtype float32 --violation 1
    --directional_edge_encoding 1
    --adaptive_rau_tol "${ADAPTIVE_RAU_TOL:-0.001}"
    --adaptive_rau_min_steps "${ADAPTIVE_RAU_MIN_STEPS:-1}"
    --uncertainty_scale "${UNCERTAINTY_SCALE:-1.0}"
    --uncertainty_ema "${UNCERTAINTY_EMA:-0.9}"
    --deterministic 1
)

for mode in 0 1; do
    python run_hattrick.py "${common[@]}" --sim_mf_mlu "$mode"
    values="$RESULT_DIR/hattrick_values_esm_sim_mlu_${mode}.txt"
    runtime="$RESULT_DIR/hattrick_runtime.txt"
    [[ "$(wc -l < "$values")" -eq "$EXPECTED_VALUE_LINES" ]]
    [[ "$(wc -l < "$runtime")" -eq "$EXPECTED_SNAPSHOTS" ]]
    label="mlu"
    [[ "$mode" -eq 1 ]] && label="flow"
    cp -f "$runtime" "$RESULT_DIR/hattrick_runtime_optimized_${label}.txt"
done
