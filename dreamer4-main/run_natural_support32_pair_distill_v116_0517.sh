#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
RESULT_ROOT="${RESULT_ROOT:-results/natural_teacher_support32_sequence_rollout_0517}"
V115_DIR="${V115_DIR:-${RESULT_ROOT}/v115_pair_interaction_distill_readonly}"
STAGE="${STAGE:-v116_pair_distill_pruning_smoke1}"
SAVE_DIR="${RESULT_ROOT}/${STAGE}"
EPOCHS="${EPOCHS:-18}"
COUNT_EPOCHS="${COUNT_EPOCHS:-120}"
BATCH_SIZE="${BATCH_SIZE:-8192}"
LR="${LR:-0.001}"
PAIR_LIMIT="${PAIR_LIMIT:-0}"
LOG="${SAVE_DIR}/pipeline.log"

mkdir -p "${SAVE_DIR}"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

{
  echo "=== ${STAGE} START $(timestamp) ==="
  echo "mode=v116 pair compatibility distillation + support-count pruning smoke"
  echo "v115_dir=${V115_DIR}"
  echo "epochs=${EPOCHS}"
  echo "count_epochs=${COUNT_EPOCHS}"
  echo "batch_size=${BATCH_SIZE}"
  echo "pair_limit=${PAIR_LIMIT}"
} | tee -a "${LOG}"

"${PYTHON_BIN}" train_vacancy_pair_distill_v116.py \
  --pair-jsonl "${V115_DIR}/pair_distillation_samples_v115.jsonl" \
  --candidate-jsonl "${V115_DIR}/candidate_support_count_samples_v115.jsonl" \
  --output-dir "${SAVE_DIR}" \
  --epochs "${EPOCHS}" \
  --count-epochs "${COUNT_EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --pair-limit "${PAIR_LIMIT}" \
  2>&1 | tee -a "${LOG}"

echo "=== ${STAGE} END $(timestamp) ===" | tee -a "${LOG}"
