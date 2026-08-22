#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/run_uscarrier_common.sh"
uscarrier_preflight
uscarrier_ensure_esm

VARIANT="${VARIANT:-baseline}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

case "$VARIANT" in
    baseline)
        VARIANT_ARGS=(
            --directional_edge_encoding 0 --adaptive_rau_tol 0
            --uncertainty_scale 0 --uncertainty_ema 0.9
            --conditional_fulfill 0 --cvar_alpha 0.1 --cvar_weight 0
        )
        ;;
    optimized)
        VARIANT_ARGS=(
            --directional_edge_encoding 1
            --adaptive_rau_tol "${ADAPTIVE_RAU_TOL:-0.001}"
            --adaptive_rau_min_steps "${ADAPTIVE_RAU_MIN_STEPS:-1}"
            --uncertainty_scale "${UNCERTAINTY_SCALE:-1.0}"
            --uncertainty_ema "${UNCERTAINTY_EMA:-0.9}"
            --conditional_fulfill 0 --fulfill_slo "${FULFILL_SLO:-0.995}"
            --cvar_alpha "${CVAR_ALPHA:-0.1}" --cvar_weight "${CVAR_WEIGHT:-0.1}"
        )
        ;;
    *)
        echo "VARIANT must be baseline or optimized" >&2
        exit 2
        ;;
esac

echo "[$(date -Is)] Training USCarrier ${VARIANT} Hattrick"
python run_hattrick.py \
    --topo uscarrier --mode train --epochs "${EPOCHS:-60}" \
    --batch_size "${BATCH_SIZE:-2}" --validation_batch_size "${VALIDATION_BATCH_SIZE:-1}" \
    --num_paths_per_pair "$USCARRIER_NUM_PATHS" \
    --num_transformer_layers 3 --num_gnn_layers 3 \
    --num_mlp1_hidden_layers 2 --num_mlp2_hidden_layers 2 \
    --rau1 3 --rau2 3 --rau3 3 \
    --train_clusters "$USCARRIER_CLUSTER" \
    --train_start_indices "$USCARRIER_TRAIN_START" --train_end_indices "$USCARRIER_TRAIN_END" \
    --val_clusters "$USCARRIER_CLUSTER" \
    --val_start_indices "$USCARRIER_VAL_START" --val_end_indices "$USCARRIER_VAL_END" \
    --pred 1 --pred_type esm --dynamic 0 --lr "${LEARNING_RATE:-0.0005}" \
    --initial_training "${INITIAL_TRAINING:-1}" --violation 1 --checkpoint 2 \
    --detect_anomaly 0 --deterministic 1 --dtype float32 --meta_learning 0 \
    "${VARIANT_ARGS[@]}"

MODEL="hattrick_uscarrier_${USCARRIER_NUM_PATHS}sp.pkl"
[[ -s "$MODEL" ]] || { echo "Training did not create $MODEL" >&2; exit 1; }
echo "[$(date -Is)] USCarrier ${VARIANT} training completed"
