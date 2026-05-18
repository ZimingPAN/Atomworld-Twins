#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
RESULT_ROOT="${RESULT_ROOT:-results/natural_teacher_support32_sequence_rollout_0517}"
V119_DIR="${V119_DIR:-${RESULT_ROOT}/v119_pair_joint_selector_grouped_readonly_smoke}"
V115_DIR="${V115_DIR:-${RESULT_ROOT}/v115_pair_interaction_distill_readonly}"
STAGE="${STAGE:-v120_candidate_pair_pruning_grouped_readonly}"
SAVE_DIR="${RESULT_ROOT}/${STAGE}"
RIDGE_L2="${RIDGE_L2:-1.0}"
LOG="${SAVE_DIR}/pipeline.log"

mkdir -p "${SAVE_DIR}"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

{
  echo "=== ${STAGE} START $(timestamp) ==="
  echo "mode=v120 pure-python candidate-quality + pair-pruning selector diagnostic"
  echo "v119_dir=${V119_DIR}"
  echo "v115_dir=${V115_DIR}"
  echo "ridge_l2=${RIDGE_L2}"
} | tee -a "${LOG}"

"${PYTHON_BIN}" diagnose_candidate_pruning_selector_v120.py \
  --v119-candidate-jsonl "${V119_DIR}/candidate_joint_targets_v119.jsonl" \
  --v115-candidate-jsonl "${V115_DIR}/candidate_support_count_samples_v115.jsonl" \
  --output-dir "${SAVE_DIR}" \
  --ridge-l2 "${RIDGE_L2}" \
  2>&1 | tee -a "${LOG}"

echo "=== ${STAGE} END $(timestamp) ===" | tee -a "${LOG}"
