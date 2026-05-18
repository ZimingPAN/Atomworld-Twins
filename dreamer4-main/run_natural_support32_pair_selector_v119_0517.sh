#!/usr/bin/env bash
set -euo pipefail

STAGE="${STAGE:-v119_pair_joint_selector_grouped_readonly}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PAIR_JSONL="${PAIR_JSONL:-results/natural_teacher_support32_sequence_rollout_0517/v115_pair_interaction_distill_readonly/pair_distillation_samples_v115.jsonl}"
CANDIDATE_JSONL="${CANDIDATE_JSONL:-results/natural_teacher_support32_sequence_rollout_0517/v115_pair_interaction_distill_readonly/candidate_support_count_samples_v115.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-results/natural_teacher_support32_sequence_rollout_0517/${STAGE}}"
PAIR_EPOCHS="${PAIR_EPOCHS:-3}"
PAIR_LR="${PAIR_LR:-0.0005}"
PAIR_RIDGE_L2="${PAIR_RIDGE_L2:-1.0}"
MAX_PAIR_TRAIN_ROWS_PER_FOLD="${MAX_PAIR_TRAIN_ROWS_PER_FOLD:-40000}"
BUDGET_EPOCHS="${BUDGET_EPOCHS:-220}"
BUDGET_LR="${BUDGET_LR:-0.002}"
SEED="${SEED:-0}"

mkdir -p "${OUTPUT_DIR}"
{
  echo "=== ${STAGE} START $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  echo "mode=v119 pure-python joint pair-score + PR-curve pruning selector diagnostic"
  echo "pair_jsonl=${PAIR_JSONL}"
  echo "candidate_jsonl=${CANDIDATE_JSONL}"
  echo "pair_epochs=${PAIR_EPOCHS}"
  echo "pair_solver=closed_form_ridge"
  echo "pair_ridge_l2=${PAIR_RIDGE_L2}"
  echo "max_pair_train_rows_per_fold=${MAX_PAIR_TRAIN_ROWS_PER_FOLD}"
  echo "budget_epochs=${BUDGET_EPOCHS}"
  "${PYTHON_BIN}" diagnose_vacancy_pair_selector_v119.py \
    --pair-jsonl "${PAIR_JSONL}" \
    --candidate-jsonl "${CANDIDATE_JSONL}" \
    --output-dir "${OUTPUT_DIR}" \
    --pair-epochs "${PAIR_EPOCHS}" \
    --pair-lr "${PAIR_LR}" \
    --pair-ridge-l2 "${PAIR_RIDGE_L2}" \
    --max-pair-train-rows-per-fold "${MAX_PAIR_TRAIN_ROWS_PER_FOLD}" \
    --budget-epochs "${BUDGET_EPOCHS}" \
    --budget-lr "${BUDGET_LR}" \
    --seed "${SEED}"
  echo "=== ${STAGE} END $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
} 2>&1 | tee "${OUTPUT_DIR}/pipeline.log"
