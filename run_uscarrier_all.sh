#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/run_uscarrier_common.sh"
uscarrier_preflight
uscarrier_ensure_esm

PIPELINE_STATE="state/uscarrier_k${USCARRIER_NUM_PATHS}_pipeline"
mkdir -p "$PIPELINE_STATE" logs output
LOG_FILE="logs/uscarrier_k${USCARRIER_NUM_PATHS}_all_$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
trap 'echo "USCarrier pipeline failed near line $LINENO. See: $LOG_FILE" >&2' ERR

run_step() {
    local name="$1"
    shift
    local marker="$PIPELINE_STATE/$name.done"
    if [[ -f "$marker" ]]; then
        echo "[$(date -Is)] Skipping completed pipeline step: $name"
        return 0
    fi
    echo "[$(date -Is)] Starting pipeline step: $name"
    "$@"
    printf 'completed=%s\n' "$(date -Is)" > "$marker"
}

archive_variant() {
    local variant="$1"
    local model="hattrick_uscarrier_${USCARRIER_NUM_PATHS}sp.pkl"
    local model_archive="models/${variant}_uscarrier_k${USCARRIER_NUM_PATHS}"
    local optimizer_archive="optimizers/${variant}_uscarrier_k${USCARRIER_NUM_PATHS}"
    local hp_archive="hp_search/${variant}_uscarrier_k${USCARRIER_NUM_PATHS}"
    mkdir -p "$model_archive/training" "$optimizer_archive/training" "$hp_archive/training"
    cp -a "$model" "$model_archive/$model"
    [[ ! -d models/uscarrier ]] || cp -a models/uscarrier/. "$model_archive/training/"
    [[ ! -d optimizers/uscarrier ]] || cp -a optimizers/uscarrier/. "$optimizer_archive/training/"
    [[ ! -d hp_search/uscarrier ]] || cp -a hp_search/uscarrier/. "$hp_archive/training/"
}

run_step 01_oracle bash "$SCRIPT_DIR/run_uscarrier_gurobi_full.sh"
run_step 02_bestmc_swan bash "$SCRIPT_DIR/run_uscarrier_baselines.sh"
run_step 03_baseline_train env VARIANT=baseline bash "$SCRIPT_DIR/run_uscarrier_train.sh"
run_step 04_baseline_test env VARIANT=baseline bash "$SCRIPT_DIR/run_uscarrier_test_both.sh"
run_step 05_baseline_archive archive_variant baseline
run_step 06_optimized_train env VARIANT=optimized bash "$SCRIPT_DIR/run_uscarrier_train.sh"
run_step 07_optimized_test env VARIANT=optimized bash "$SCRIPT_DIR/run_uscarrier_test_both.sh"
run_step 08_optimized_archive archive_variant optimized

printf 'completed=%s\ncommit=%s\n' "$(date -Is)" "$(git rev-parse HEAD)" > "$PIPELINE_STATE/pipeline.done"
echo "[$(date -Is)] USCarrier baseline and optimized pipeline completed"
echo "Log: $LOG_FILE"
