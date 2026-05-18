#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
RESULT_ROOT="${RESULT_ROOT:-results/natural_teacher_support32_sequence_rollout_0517}"
STAGE="${STAGE:-v133_conservative_budget_guard_readonly}"
SAVE_DIR="${RESULT_ROOT}/${STAGE}"
LOG="${SAVE_DIR}/pipeline.log"

mkdir -p "${SAVE_DIR}"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

{
  echo "=== ${STAGE} START $(timestamp) ==="
  echo "mode=v133 pure-python conservative calibration / budget-guard readonly diagnostic"
  echo "python=${PYTHON_BIN}"
} | tee -a "${LOG}"

"${PYTHON_BIN}" diagnose_candidate_budget_guard_v133.py \
  --output_dir "${SAVE_DIR}" \
  2>&1 | tee -a "${LOG}"

echo "=== ${STAGE} END $(timestamp) ===" | tee -a "${LOG}"
