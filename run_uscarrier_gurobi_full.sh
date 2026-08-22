#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/run_uscarrier_common.sh"
uscarrier_preflight

if [[ ! -s "$GUROBI_LICENSE_FILE" ]]; then
    echo "Gurobi license not found: $GUROBI_LICENSE_FILE" >&2
    exit 1
fi

RESULT_DIR="results/uscarrier/${USCARRIER_NUM_PATHS}sp/${USCARRIER_CLUSTER}"
STATE_DIR="results/uscarrier/${USCARRIER_NUM_PATHS}sp/.full_solve_state"
mkdir -p "$RESULT_DIR" "$STATE_DIR"

line_count() {
    local path="$1"
    [[ -f "$path" ]] && wc -l < "$path" || echo 0
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

echo "[$(date -Is)] Full USCarrier K=${USCARRIER_NUM_PATHS} oracle started"
for index in "${!stage_names[@]}"; do
    name="${stage_names[$index]}"
    output="${outputs[$index]}"
    marker="$STATE_DIR/$name.done"
    count="$(line_count "$RESULT_DIR/$output")"
    if [[ -f "$marker" && "$count" -eq "$USCARRIER_TOTAL" ]]; then
        echo "[$(date -Is)] Skipping completed oracle stage: $name"
        continue
    fi
    rm -f "$marker"

    start_idx=0
    resume_args=()
    if [[ "$count" -gt 0 && "$count" -lt "$USCARRIER_TOTAL" ]]; then
        start_idx="$count"
        resume_args=(--resume_opt)
        echo "[$(date -Is)] Resuming oracle stage $name from snapshot $start_idx"
    elif [[ "$count" -gt "$USCARRIER_TOTAL" ]]; then
        echo "Oracle stage $name has $count rows; expected at most $USCARRIER_TOTAL" >&2
        exit 1
    fi

    read -r -a objective_args <<< "${objectives[$index]}"
    python frameworks/gurobi_refactored.py \
        --num_paths_per_pair "$USCARRIER_NUM_PATHS" \
        --opt_start_idx "$start_idx" --opt_end_idx "$USCARRIER_TOTAL" \
        --topo uscarrier --framework gurobi --pred 0 --pred_type esm \
        --cluster "$USCARRIER_CLUSTER" --priority "${priorities[$index]}" \
        --objs "${objective_args[@]}" --gur_mode flexile --tol 0.000001 \
        "${resume_args[@]}"

    count="$(line_count "$RESULT_DIR/$output")"
    if [[ "$count" -ne "$USCARRIER_TOTAL" ]]; then
        echo "Oracle stage $name produced $count rows; expected $USCARRIER_TOTAL" >&2
        exit 1
    fi
    printf 'completed=%s\nrows=%s\n' "$(date -Is)" "$count" > "$marker"
done
echo "[$(date -Is)] Full USCarrier oracle completed"
