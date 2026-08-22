#!/usr/bin/env bash

set -Eeuo pipefail

USCARRIER_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
USCARRIER_VENV="${HATTRICK_VENV:-/mnt/data0/helo}"
USCARRIER_NUM_PATHS="${USCARRIER_NUM_PATHS:-4}"
USCARRIER_CLUSTER="${USCARRIER_CLUSTER:-0}"
USCARRIER_MANIFEST="$USCARRIER_SCRIPT_DIR/manifest/uscarrier_manifest.txt"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export GUROBI_LICENSE_FILE="${GUROBI_LICENSE_FILE:-/root/gurobi.lic}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONUNBUFFERED=1

uscarrier_activate() {
    if [[ ! -f "$USCARRIER_VENV/bin/activate" ]]; then
        echo "Virtual environment not found: $USCARRIER_VENV" >&2
        return 1
    fi
    # shellcheck disable=SC1090
    source "$USCARRIER_VENV/bin/activate"
}

uscarrier_file_count() {
    local directory="$1"
    if [[ -d "$directory" ]]; then
        find "$directory" -maxdepth 1 -type f -name 't*.pkl' | wc -l
    else
        echo 0
    fi
}

uscarrier_initialize_ranges() {
    if [[ ! -s "$USCARRIER_MANIFEST" ]]; then
        echo "USCarrier manifest is missing: $USCARRIER_MANIFEST" >&2
        return 1
    fi
    USCARRIER_TOTAL="${USCARRIER_TOTAL:-$(wc -l < "$USCARRIER_MANIFEST")}"
    USCARRIER_TRAIN_START="${USCARRIER_TRAIN_START:-0}"
    USCARRIER_TRAIN_END="${USCARRIER_TRAIN_END:-$((USCARRIER_TOTAL * 3 / 5))}"
    USCARRIER_VAL_START="${USCARRIER_VAL_START:-$USCARRIER_TRAIN_END}"
    USCARRIER_VAL_END="${USCARRIER_VAL_END:-$((USCARRIER_TOTAL * 4 / 5))}"
    USCARRIER_TEST_START="${USCARRIER_TEST_START:-$USCARRIER_VAL_END}"
    USCARRIER_TEST_END="${USCARRIER_TEST_END:-$USCARRIER_TOTAL}"

    if (( USCARRIER_TRAIN_START < 0 \
        || USCARRIER_TRAIN_START >= USCARRIER_TRAIN_END \
        || USCARRIER_TRAIN_END != USCARRIER_VAL_START \
        || USCARRIER_VAL_START >= USCARRIER_VAL_END \
        || USCARRIER_VAL_END != USCARRIER_TEST_START \
        || USCARRIER_TEST_START >= USCARRIER_TEST_END \
        || USCARRIER_TEST_END > USCARRIER_TOTAL )); then
        echo "Invalid USCarrier split configuration" >&2
        return 1
    fi

    export USCARRIER_TOTAL USCARRIER_TRAIN_START USCARRIER_TRAIN_END
    export USCARRIER_VAL_START USCARRIER_VAL_END USCARRIER_TEST_START USCARRIER_TEST_END
}

uscarrier_validate_ground_truth() {
    local priority directory count
    for priority in 1 2 3; do
        directory="$USCARRIER_SCRIPT_DIR/traffic_matrices/uscarrier_${priority}"
        count="$(uscarrier_file_count "$directory")"
        if [[ "$count" -ne "$USCARRIER_TOTAL" ]]; then
            echo "USCarrier class ${priority} has ${count} TMs; expected ${USCARRIER_TOTAL}: $directory" >&2
            return 1
        fi
    done
}

uscarrier_ensure_esm() {
    local priority directory count generate=0
    for priority in 1 2 3; do
        directory="$USCARRIER_SCRIPT_DIR/traffic_matrices/uscarrier_${priority}_esm"
        count="$(uscarrier_file_count "$directory")"
        if [[ "$count" -ne "$USCARRIER_TOTAL" ]]; then
            generate=1
        fi
    done
    if [[ "$generate" -eq 1 ]]; then
        echo "[$(date -Is)] Generating USCarrier ESM predictions"
        (
            cd "$USCARRIER_SCRIPT_DIR/traffic_matrices"
            python esm_predictor.py uscarrier
        )
    fi
    for priority in 1 2 3; do
        directory="$USCARRIER_SCRIPT_DIR/traffic_matrices/uscarrier_${priority}_esm"
        count="$(uscarrier_file_count "$directory")"
        if [[ "$count" -ne "$USCARRIER_TOTAL" ]]; then
            echo "USCarrier ESM class ${priority} has ${count} TMs; expected ${USCARRIER_TOTAL}" >&2
            return 1
        fi
    done
}

uscarrier_preflight() {
    cd "$USCARRIER_SCRIPT_DIR"
    uscarrier_activate
    uscarrier_initialize_ranges
    uscarrier_validate_ground_truth
    if [[ ! -s topologies/uscarrier/t1.json || ! -s pairs/uscarrier/t1.pkl ]]; then
        echo "USCarrier topology or pairs file is missing" >&2
        return 1
    fi
    mkdir -p logs output state "results/uscarrier/${USCARRIER_NUM_PATHS}sp/${USCARRIER_CLUSTER}"
    echo "USCarrier K=${USCARRIER_NUM_PATHS}, total=${USCARRIER_TOTAL}"
    echo "train=[${USCARRIER_TRAIN_START},${USCARRIER_TRAIN_END})"
    echo "val=[${USCARRIER_VAL_START},${USCARRIER_VAL_END})"
    echo "test=[${USCARRIER_TEST_START},${USCARRIER_TEST_END})"
}

uscarrier_initialize_ranges
