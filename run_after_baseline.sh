#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BASELINE_LOG="${BASELINE_LOG:-/mnt/data0/Hattrick/logs/geant_k8_resume_20260821.log}"
BASELINE_PATTERN="[r]un_geant_k8_all.sh"
READY_MARKER="state/optimized_ready"
POLL_SECONDS="${POLL_SECONDS:-60}"

mkdir -p logs state

echo "[$(date -Is)] Waiting for the baseline GEANT K=8 pipeline"
echo "Baseline log: $BASELINE_LOG"

while true; do
    if [[ -f "$BASELINE_LOG" ]] && grep -Fq "GEANT K=8 pipeline completed" "$BASELINE_LOG"; then
        echo "[$(date -Is)] Baseline pipeline completed successfully"
        break
    fi

    if ! pgrep -f "$BASELINE_PATTERN" >/dev/null; then
        echo "[$(date -Is)] Baseline process stopped without a completion marker" >&2
        exit 1
    fi

    sleep "$POLL_SECONDS"
done

echo "[$(date -Is)] Waiting for the optimized implementation to pass validation"
while [[ ! -f "$READY_MARKER" ]]; do
    sleep "$POLL_SECONDS"
done

if [[ ! -x ./run_geant_optimized.sh ]]; then
    echo "Optimized entry point is not executable: $SCRIPT_DIR/run_geant_optimized.sh" >&2
    exit 1
fi

echo "[$(date -Is)] Starting the optimized GEANT experiment"
exec ./run_geant_optimized.sh all
