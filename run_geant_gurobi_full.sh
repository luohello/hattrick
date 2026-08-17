#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="${HATTRICK_VENV:-/mnt/data0/helo}"
source "$VENV_DIR/bin/activate"

export GUROBI_LICENSE_FILE="${GUROBI_LICENSE_FILE:-/root/gurobi.lic}"
export PYTHONUNBUFFERED=1

NUM_PATHS=8
EXPECTED_SNAPSHOTS=10772
RESULT_DIR="results/geant/${NUM_PATHS}sp/0"
STATE_DIR="results/geant/${NUM_PATHS}sp/.full_solve_state"
mkdir -p "$RESULT_DIR" "$STATE_DIR"

if [[ ! -f "$GUROBI_LICENSE_FILE" ]]; then
    echo "Gurobi license not found: $GUROBI_LICENSE_FILE" >&2
    exit 1
fi

line_count() {
    local path="$1"
    if [[ -f "$path" ]]; then
        wc -l < "$path"
    else
        echo 0
    fi
}

stage_names=(01_mf 02_mf_mf 03_mf_mf_mf 04_mlu 05_mlu_mlu 06_mlu_mlu_mlu)
priorities=(1 2 3 1 2 3)
objectives=("mf" "mf mf" "mf mf mf" "mlu" "mlu mlu" "mlu mlu mlu")
outputs=(
    gt_optimal_values_mf.txt
    gt_optimal_values_mf_mf.txt
    gt_optimal_values_mf_mf_mf.txt
    gt_optimal_values_mlu.txt
    gt_optimal_values_mlu_mlu.txt
    gt_optimal_values_mlu_mlu_mlu.txt
)

echo "[$(date -Is)] Full GEANT K=8 oracle started"

for index in "${!stage_names[@]}"; do
    name="${stage_names[$index]}"
    output="${outputs[$index]}"
    marker="$STATE_DIR/$name.done"
    count="$(line_count "$RESULT_DIR/$output")"

    if [[ -f "$marker" && "$count" -eq "$EXPECTED_SNAPSHOTS" ]]; then
        echo "[$(date -Is)] Skipping completed oracle stage: $name"
        continue
    fi
    rm -f "$marker"

    read -r -a objective_args <<< "${objectives[$index]}"
    echo "[$(date -Is)] Starting oracle stage: $name"
    python frameworks/gurobi_refactored.py \
        --num_paths_per_pair "$NUM_PATHS" \
        --opt_start_idx 0 \
        --opt_end_idx "$EXPECTED_SNAPSHOTS" \
        --topo geant \
        --framework gurobi \
        --pred 0 \
        --pred_type esm \
        --cluster 0 \
        --priority "${priorities[$index]}" \
        --objs "${objective_args[@]}" \
        --gur_mode flexile \
        --tol 0.000001

    count="$(line_count "$RESULT_DIR/$output")"
    if [[ "$count" -ne "$EXPECTED_SNAPSHOTS" ]]; then
        echo "Oracle stage $name produced $count rows; expected $EXPECTED_SNAPSHOTS" >&2
        exit 1
    fi
    printf 'completed=%s\nrows=%s\n' "$(date -Is)" "$count" > "$marker"
    echo "[$(date -Is)] Completed oracle stage: $name"
done

echo "[$(date -Is)] Full GEANT K=8 oracle completed"
