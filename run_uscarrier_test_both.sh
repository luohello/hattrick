#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/run_uscarrier_common.sh"
uscarrier_preflight
uscarrier_ensure_esm

VARIANT="${VARIANT:-baseline}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
RESULT_DIR="results/uscarrier/${USCARRIER_NUM_PATHS}sp/${USCARRIER_CLUSTER}"
ARCHIVE_DIR="$RESULT_DIR/${VARIANT}_hattrick"
EXPECTED_SNAPSHOTS=$((USCARRIER_TEST_END - USCARRIER_TEST_START))
EXPECTED_VALUE_LINES=$((EXPECTED_SNAPSHOTS * 3))
mkdir -p "$RESULT_DIR" "$ARCHIVE_DIR"

case "$VARIANT" in
    baseline)
        VARIANT_ARGS=(--directional_edge_encoding 0 --adaptive_rau_tol 0 --uncertainty_scale 0 --cvar_weight 0)
        ;;
    optimized)
        VARIANT_ARGS=(
            --directional_edge_encoding 1
            --adaptive_rau_tol "${ADAPTIVE_RAU_TOL:-0.001}"
            --adaptive_rau_min_steps "${ADAPTIVE_RAU_MIN_STEPS:-1}"
            --uncertainty_scale "${UNCERTAINTY_SCALE:-1.0}"
            --uncertainty_ema "${UNCERTAINTY_EMA:-0.9}"
            --cvar_alpha "${CVAR_ALPHA:-0.1}" --cvar_weight "${CVAR_WEIGHT:-0.1}"
        )
        ;;
    *) echo "VARIANT must be baseline or optimized" >&2; exit 2 ;;
esac

run_inference() {
    local mode="$1" label="$2"
    local values="$RESULT_DIR/hattrick_values_esm_sim_mlu_${mode}.txt"
    local stats="$RESULT_DIR/hattrick_stats_esm_sim_mlu_${mode}.txt"
    local runtime="$RESULT_DIR/hattrick_runtime.txt"
    python run_hattrick.py \
        --topo uscarrier --mode test \
        --test_start_idx "$USCARRIER_TEST_START" --test_end_idx "$USCARRIER_TEST_END" \
        --test_cluster "$USCARRIER_CLUSTER" --num_paths_per_pair "$USCARRIER_NUM_PATHS" \
        --rau1 3 --rau2 3 --rau3 3 --pred 1 --dynamic 0 --pred_type esm \
        --dtype float32 --violation 1 --sim_mf_mlu "$mode" \
        "${VARIANT_ARGS[@]}"
    [[ "$(wc -l < "$values")" -eq "$EXPECTED_VALUE_LINES" ]] || return 1
    [[ "$(wc -l < "$runtime")" -eq "$EXPECTED_SNAPSHOTS" ]] || return 1
    cp -a "$values" "$ARCHIVE_DIR/"
    cp -a "$stats" "$ARCHIVE_DIR/"
    cp -a "$runtime" "$ARCHIVE_DIR/hattrick_runtime_${label}.txt"
}

echo "[$(date -Is)] Testing USCarrier ${VARIANT} Hattrick"
run_inference 0 mlu
run_inference 1 flow
cp -a "$RESULT_DIR/hattrick_runtime.txt" "$ARCHIVE_DIR/hattrick_runtime.txt"
echo "[$(date -Is)] USCarrier ${VARIANT} inference completed"
