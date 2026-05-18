#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
RESULT_ROOT="${RESULT_ROOT:-results/natural_teacher_support32_sequence_rollout_0517}"
STAGE="${STAGE:-v138_pair_retention_target_readonly}"
OUT_DIR="${RESULT_ROOT}/${STAGE}"
EVAL_JSON="${EVAL_JSON:-${RESULT_ROOT}/v137_full_true_pair_rank_budgeted_diagnostic_smoke20/eval_long_full_true_pair_rank_budgeted_diagnostic_20.json}"
LOG="${OUT_DIR}/pipeline.log"

mkdir -p "${OUT_DIR}"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

{
  echo "=== ${STAGE} START $(timestamp) ==="
  echo "mode=pure_python_grouped_loo_pair_retention_target_readonly"
  echo "eval_json=${EVAL_JSON}"
} | tee -a "${LOG}"

"${PYTHON_BIN}" diagnose_pair_score_retention_v138.py \
  --eval-json "${EVAL_JSON}" \
  --output-dir "${OUT_DIR}" \
  2>&1 | tee -a "${LOG}"

echo "=== ${STAGE} END $(timestamp) ===" | tee -a "${LOG}"
