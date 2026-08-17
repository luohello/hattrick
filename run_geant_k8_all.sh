#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs output
LOG_FILE="logs/geant_k8_all_$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

export GUROBI_LICENSE_FILE="${GUROBI_LICENSE_FILE:-/root/gurobi.lic}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONUNBUFFERED=1

echo "[$(date -Is)] GEANT K=8 pipeline started"

bash run_geant_gurobi_full.sh

if [[ ! -s hattrick_geant_8sp.pkl ]]; then
    bash run_geant_train_first.sh
else
    echo "[$(date -Is)] Existing Hattrick K=8 model found; skipping training"
fi

EXPECTED_VALUE_LINES=$(((10772 - 8618) * 3))
MLU_VALUES="results/geant/8sp/0/hattrick_values_esm_sim_mlu_0.txt"
FLOW_VALUES="results/geant/8sp/0/hattrick_values_esm_sim_mlu_1.txt"
mlu_lines=0
flow_lines=0
[[ -f "$MLU_VALUES" ]] && mlu_lines="$(wc -l < "$MLU_VALUES")"
[[ -f "$FLOW_VALUES" ]] && flow_lines="$(wc -l < "$FLOW_VALUES")"
if [[ "$mlu_lines" -ne "$EXPECTED_VALUE_LINES" || "$flow_lines" -ne "$EXPECTED_VALUE_LINES" ]]; then
    bash run_geant_test_both.sh
else
    echo "[$(date -Is)] Existing Hattrick K=8 test outputs verified; skipping inference"
fi

bash run_geant_baselines.sh

source "${HATTRICK_VENV:-/mnt/data0/helo}/bin/activate"
python summarize_geant_results.py \
    --num-paths 8 \
    --test-start 8618 \
    --test-end 10772 \
    --output-dir output/geant_k8

echo "[$(date -Is)] GEANT K=8 pipeline completed"
echo "Log: $LOG_FILE"
