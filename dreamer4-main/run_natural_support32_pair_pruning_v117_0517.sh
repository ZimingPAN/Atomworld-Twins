#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
RESULT_ROOT="${RESULT_ROOT:-results/natural_teacher_support32_sequence_rollout_0517}"
V115_DIR="${V115_DIR:-${RESULT_ROOT}/v115_pair_interaction_distill_readonly}"
STAGE="${STAGE:-v117_pair_pruning_calibration_readonly}"
SAVE_DIR="${RESULT_ROOT}/${STAGE}"
SCORE_FIELD="${SCORE_FIELD:-calibrated_interaction_score}"
LOG="${SAVE_DIR}/pipeline.log"

mkdir -p "${SAVE_DIR}"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

{
  echo "=== ${STAGE} START $(timestamp) ==="
  echo "mode=v117 pure-python pruning/count calibration diagnostic"
  echo "v115_dir=${V115_DIR}"
  echo "score_field=${SCORE_FIELD}"
} | tee -a "${LOG}"

"${PYTHON_BIN}" diagnose_vacancy_pair_pruning_v117.py \
  --pair-jsonl "${V115_DIR}/pair_distillation_samples_v115.jsonl" \
  --candidate-jsonl "${V115_DIR}/candidate_support_count_samples_v115.jsonl" \
  --output-dir "${SAVE_DIR}" \
  --score-field "${SCORE_FIELD}" \
  2>&1 | tee -a "${LOG}"

echo "=== ${STAGE} END $(timestamp) ===" | tee -a "${LOG}"
