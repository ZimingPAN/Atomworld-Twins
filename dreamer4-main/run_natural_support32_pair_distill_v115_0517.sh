#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
RESULT_ROOT="${RESULT_ROOT:-results/natural_teacher_support32_sequence_rollout_0517}"
STAGE="${STAGE:-v115_pair_interaction_distill_readonly}"
SAVE_DIR="${RESULT_ROOT}/${STAGE}"
V111_DIR="${V111_DIR:-${RESULT_ROOT}/v111_pair_factorized_readonly_smoke20}"
RIDGE_L2="${RIDGE_L2:-1.0}"
MAX_PAIRS_PER_CANDIDATE_JSONL="${MAX_PAIRS_PER_CANDIDATE_JSONL:-0}"
LOG="${SAVE_DIR}/pipeline.log"

mkdir -p "${SAVE_DIR}"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

{
  echo "=== ${STAGE} START $(timestamp) ==="
  echo "mode=v115 read-only calibrated interaction distillation samples"
  echo "v111_dir=${V111_DIR}"
  echo "ridge_l2=${RIDGE_L2}"
  echo "max_pairs_per_candidate_jsonl=${MAX_PAIRS_PER_CANDIDATE_JSONL}"
} | tee -a "${LOG}"

"${PYTHON_BIN}" diagnose_vacancy_pair_distill_v115.py \
  --eval-json "${V111_DIR}/eval_long_vacancy_factorized_20.json" \
  --name vacancy_factorized \
  --eval-json "${V111_DIR}/eval_long_energy_factorized_20.json" \
  --name energy_factorized \
  --output-dir "${SAVE_DIR}" \
  --ridge-l2 "${RIDGE_L2}" \
  --max-pairs-per-candidate-jsonl "${MAX_PAIRS_PER_CANDIDATE_JSONL}" \
  2>&1 | tee -a "${LOG}"

echo "=== ${STAGE} END $(timestamp) ===" | tee -a "${LOG}"
