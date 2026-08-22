#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/run_uscarrier_common.sh"
uscarrier_preflight
uscarrier_ensure_esm

RESULT_DIR="results/uscarrier/${USCARRIER_NUM_PATHS}sp/${USCARRIER_CLUSTER}"
STATE_DIR="$RESULT_DIR/.baseline_state"
SWAN_SPLIT_DIR="$(dirname "$SCRIPT_DIR")/scratch/split_ratios/uscarrier/${USCARRIER_NUM_PATHS}sp/swan/esm"
EXPECTED_SNAPSHOTS=$((USCARRIER_TEST_END - USCARRIER_TEST_START))
EXPECTED_SIMULATION_LINES=$((EXPECTED_SNAPSHOTS * 6))
mkdir -p "$RESULT_DIR" "$STATE_DIR"

line_count() {
    local path="$1"
    [[ -f "$path" ]] && wc -l < "$path" || echo 0
}

pickle_count() {
    local path="$1"
    [[ -d "$path" ]] && find "$path" -maxdepth 1 -type f -name '*.pkl' | wc -l || echo 0
}

stage_names=(
    01_bestmc_priority1 02_bestmc_priority2 03_bestmc_priority3
    04_swan_priority1 05_swan_priority2 06_swan_priority3
)
modes=(flexile flexile flexile swan swan swan)
priorities=(1 2 3 1 2 3)
objectives=("mf" "mf mf" "mf mf mf" "mf" "mf mf" "mf mf mf")
results=(esm_optimal_values_mf.txt esm_optimal_values_mf_mf.txt esm_optimal_values_mf_mf_mf.txt "" "" "")
runtimes=(flexile_runtime_1.txt flexile_runtime_2.txt flexile_runtime_3.txt swan_runtime_1.txt swan_runtime_2.txt swan_runtime_3.txt)
simulations=("" "" flexile_sim_results_esm_mf_mf_mf.txt "" "" swan_sim_results_esm_mf_mf_mf.txt)

stage_complete() {
    local index="$1"
    [[ "$(line_count "$RESULT_DIR/${runtimes[$index]}")" -eq "$EXPECTED_SNAPSHOTS" ]] || return 1
    if [[ -n "${results[$index]}" ]]; then
        [[ "$(line_count "$RESULT_DIR/${results[$index]}")" -eq "$EXPECTED_SNAPSHOTS" ]] || return 1
    fi
    if [[ "${modes[$index]}" == swan ]]; then
        [[ "$(pickle_count "$SWAN_SPLIT_DIR")" -eq "$EXPECTED_SNAPSHOTS" ]] || return 1
    fi
    if [[ -n "${simulations[$index]}" ]]; then
        [[ "$(line_count "$RESULT_DIR/${simulations[$index]}")" -eq "$EXPECTED_SIMULATION_LINES" ]] || return 1
    fi
}

echo "[$(date -Is)] USCarrier BEST-MC/SWAN baselines started"
for index in "${!stage_names[@]}"; do
    name="${stage_names[$index]}"
    marker="$STATE_DIR/$name.done"
    if stage_complete "$index"; then
        echo "[$(date -Is)] Skipping completed baseline stage: $name"
        continue
    fi
    rm -f "$marker"
    read -r -a objective_args <<< "${objectives[$index]}"
    python frameworks/gurobi_refactored.py \
        --num_paths_per_pair "$USCARRIER_NUM_PATHS" \
        --opt_start_idx "$USCARRIER_TEST_START" --opt_end_idx "$USCARRIER_TEST_END" \
        --topo uscarrier --framework gurobi --pred 1 --pred_type esm \
        --cluster "$USCARRIER_CLUSTER" --priority "${priorities[$index]}" \
        --objs "${objective_args[@]}" --gur_mode "${modes[$index]}" --tol 0.000001
    if ! stage_complete "$index"; then
        echo "Baseline stage $name finished but output validation failed" >&2
        exit 1
    fi
    printf 'completed=%s\n' "$(date -Is)" > "$marker"
done
echo "[$(date -Is)] USCarrier BEST-MC/SWAN baselines completed"
