#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="${HATTRICK_VENV:-/mnt/data0/helo}"
source "$VENV_DIR/bin/activate"

export GUROBI_LICENSE_FILE="${GUROBI_LICENSE_FILE:-/root/gurobi.lic}"
export PYTHONUNBUFFERED=1

NUM_PATHS=8
TEST_START=8618
TEST_END=10772
EXPECTED_SNAPSHOTS=$((TEST_END - TEST_START))
EXPECTED_SIMULATION_LINES=$((6 * EXPECTED_SNAPSHOTS))
RESULT_DIR="results/geant/${NUM_PATHS}sp/0"
STATE_DIR="$RESULT_DIR/.baseline_state"
SWAN_SPLIT_DIR="$(dirname "$SCRIPT_DIR")/scratch/split_ratios/geant/${NUM_PATHS}sp/swan/esm"
mkdir -p "$RESULT_DIR" "$STATE_DIR"

line_count() {
    local path="$1"
    if [[ -f "$path" ]]; then
        wc -l < "$path"
    else
        echo 0
    fi
}

pickle_count() {
    local path="$1"
    if [[ -d "$path" ]]; then
        find "$path" -maxdepth 1 -type f -name '*.pkl' | wc -l
    else
        echo 0
    fi
}

for priority in 1 2 3; do
    runtime="$RESULT_DIR/flexile_runtime_${priority}.txt"
    backup="$RESULT_DIR/gt_flexile_runtime_${priority}.txt"
    if [[ -f "$runtime" && ! -f "$backup" ]]; then
        cp "$runtime" "$backup"
    fi
done

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
    if [[ "${modes[$index]}" == "swan" ]]; then
        [[ "$(pickle_count "$SWAN_SPLIT_DIR")" -eq "$EXPECTED_SNAPSHOTS" ]] || return 1
    fi
    if [[ -n "${simulations[$index]}" ]]; then
        [[ "$(line_count "$RESULT_DIR/${simulations[$index]}")" -eq "$EXPECTED_SIMULATION_LINES" ]] || return 1
    fi
}

echo "[$(date -Is)] GEANT K=8 BEST_MC/SWAN baselines started"
echo "Test interval: [$TEST_START, $TEST_END)"

for index in "${!stage_names[@]}"; do
    name="${stage_names[$index]}"
    marker="$STATE_DIR/$name.done"
    if stage_complete "$index"; then
        echo "[$(date -Is)] Skipping completed baseline stage: $name"
        continue
    fi
    rm -f "$marker"

    read -r -a objective_args <<< "${objectives[$index]}"
    echo "[$(date -Is)] Starting baseline stage: $name"
    python frameworks/gurobi_refactored.py \
        --num_paths_per_pair "$NUM_PATHS" \
        --opt_start_idx "$TEST_START" \
        --opt_end_idx "$TEST_END" \
        --topo geant \
        --framework gurobi \
        --pred 1 \
        --pred_type esm \
        --cluster 0 \
        --priority "${priorities[$index]}" \
        --objs "${objective_args[@]}" \
        --gur_mode "${modes[$index]}" \
        --tol 0.000001

    if ! stage_complete "$index"; then
        echo "Baseline stage $name finished but output validation failed" >&2
        exit 1
    fi
    printf 'completed=%s\n' "$(date -Is)" > "$marker"
    echo "[$(date -Is)] Completed baseline stage: $name"
done

echo "[$(date -Is)] GEANT K=8 BEST_MC/SWAN baselines completed"
