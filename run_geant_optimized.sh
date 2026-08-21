#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:-all}"
VENV_DIR="${HATTRICK_VENV:-/mnt/data0/helo}"
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "Virtual environment not found: $VENV_DIR" >&2
    exit 1
fi
source "$VENV_DIR/bin/activate"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONUNBUFFERED=1

mkdir -p logs output state
LOG_FILE="logs/geant_optimized_${MODE}_$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
trap 'echo "Optimized pipeline failed near line $LINENO. See: $LOG_FILE" >&2' ERR

COMMON_ARGS=(
    --topo geant
    --num_paths_per_pair 8
    --num_transformer_layers 3
    --num_gnn_layers 3
    --num_mlp1_hidden_layers 2
    --num_mlp2_hidden_layers 2
    --rau1 3 --rau2 3 --rau3 3
    --pred 1 --pred_type esm --dynamic 0
    --violation 1 --dtype float32
    --directional_edge_encoding 1
    --adaptive_rau_tol "${ADAPTIVE_RAU_TOL:-0.001}"
    --adaptive_rau_min_steps "${ADAPTIVE_RAU_MIN_STEPS:-1}"
    --uncertainty_scale "${UNCERTAINTY_SCALE:-1.0}"
    --uncertainty_ema "${UNCERTAINTY_EMA:-0.9}"
    --conditional_fulfill 0
    --fulfill_slo "${FULFILL_SLO:-0.995}"
    --cvar_alpha "${CVAR_ALPHA:-0.1}"
    --cvar_weight "${CVAR_WEIGHT:-0.1}"
    --checkpoint 2
    --detect_anomaly 0
    --deterministic 1
)

BASELINE_ROOT="${BASELINE_ROOT:-/mnt/data0/Hattrick}"

ensure_readonly_link() {
    local source_path="$1"
    local target_path="$2"
    if [[ -e "$target_path" || -L "$target_path" ]]; then
        return 0
    fi
    [[ -e "$source_path" ]] || return 0
    mkdir -p "$(dirname -- "$target_path")"
    ln -s "$source_path" "$target_path"
}

prepare_shared_inputs() {
    local name
    for name in geant_1 geant_1_esm geant_2 geant_2_esm geant_3 geant_3_esm; do
        ensure_readonly_link \
            "$BASELINE_ROOT/traffic_matrices/$name" \
            "$SCRIPT_DIR/traffic_matrices/$name"
    done

    for name in \
        filenames.txt \
        gt_optimal_values_mlu.txt gt_optimal_values_mlu_mlu.txt gt_optimal_values_mlu_mlu_mlu.txt \
        gt_optimal_values_mf.txt gt_optimal_values_mf_mf.txt gt_optimal_values_mf_mf_mf.txt \
        flexile_sim_results_esm_mf_mf_mf.txt swan_sim_results_esm_mf_mf_mf.txt \
        flexile_runtime_1.txt flexile_runtime_2.txt flexile_runtime_3.txt \
        swan_runtime_1.txt swan_runtime_2.txt swan_runtime_3.txt; do
        ensure_readonly_link \
            "$BASELINE_ROOT/results/geant/8sp/0/$name" \
            "$SCRIPT_DIR/results/geant/8sp/0/$name"
    done
}

validate_assets() {
    local required=(
        results/geant/8sp/0/filenames.txt
        results/geant/8sp/0/gt_optimal_values_mlu.txt
        results/geant/8sp/0/gt_optimal_values_mlu_mlu.txt
        results/geant/8sp/0/gt_optimal_values_mlu_mlu_mlu.txt
        results/geant/8sp/0/gt_optimal_values_mf.txt
        results/geant/8sp/0/gt_optimal_values_mf_mf.txt
        results/geant/8sp/0/gt_optimal_values_mf_mf_mf.txt
    )
    for path in "${required[@]}"; do
        [[ -s "$path" ]] || { echo "Missing optimized-run asset: $path" >&2; return 1; }
    done
}

run_checks() {
    prepare_shared_inputs
    python -m compileall -q run_hattrick.py frameworks utils tests
    python -m unittest tests.test_optimized_components -v
    validate_assets
    echo "Optimized implementation validation passed"
}

run_train() {
    python run_hattrick.py "${COMMON_ARGS[@]}" \
        --mode train \
        --epochs "${EPOCHS:-60}" \
        --batch_size "${BATCH_SIZE:-64}" \
        --validation_batch_size "${VALIDATION_BATCH_SIZE:-64}" \
        --train_clusters 0 \
        --train_start_indices "${TRAIN_START:-0}" \
        --train_end_indices "${TRAIN_END:-6464}" \
        --val_clusters 0 \
        --val_start_indices "${VAL_START:-6464}" \
        --val_end_indices "${VAL_END:-8618}" \
        --lr "${LEARNING_RATE:-0.0005}" \
        --initial_training "${INITIAL_TRAINING:-1}" \
        --meta_learning 0
}

run_test() {
    bash run_geant_optimized_test.sh
}

case "$MODE" in
    check)
        run_checks
        ;;
    train)
        run_checks
        run_train
        ;;
    smoke)
        run_checks
        EPOCHS=1 BATCH_SIZE=2 VALIDATION_BATCH_SIZE=1 \
            TRAIN_START=0 TRAIN_END=2 VAL_START=2 VAL_END=3 run_train
        ;;
    test)
        run_checks
        run_test
        ;;
    all)
        run_checks
        run_train
        run_test
        python summarize_geant_results.py \
            --num-paths 8 --test-start 8618 --test-end 10772 \
            --output-dir output/geant_k8_optimized
        ;;
    *)
        echo "Usage: $0 {check|smoke|train|test|all}" >&2
        exit 2
        ;;
esac

echo "[$(date -Is)] Optimized GEANT ${MODE} completed"
echo "Log: $LOG_FILE"
