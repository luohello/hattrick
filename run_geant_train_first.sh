#!/usr/bin/env bash
set -euo pipefail

cd /mnt/data0/Hattrick
source /mnt/data0/helo/bin/activate

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONUNBUFFERED=1

python run_hattrick.py \
  --topo geant \
  --mode train \
  --epochs 60 \
  --batch_size 64 \
  --validation_batch_size 64 \
  --num_paths_per_pair 8 \
  --num_transformer_layers 3 \
  --num_gnn_layers 3 \
  --num_mlp1_hidden_layers 2 \
  --num_mlp2_hidden_layers 2 \
  --rau1 3 \
  --rau2 3 \
  --rau3 3 \
  --train_clusters 0 \
  --train_start_indices 0 \
  --train_end_indices 6464 \
  --val_clusters 0 \
  --val_start_indices 6464 \
  --val_end_indices 8618 \
  --pred 1 \
  --dynamic 0 \
  --lr 0.0005 \
  --pred_type esm \
  --initial_training 1 \
  --violation 1 \
  --checkpoint 2 \
  --detect_anomaly 0 \
  --dtype float32 \
  --meta_learning 0
