#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
STAGE="${STAGE:-v128_budget_retention_readonly}"
ROOT_DIR="results/natural_teacher_support32_sequence_rollout_0517"
SAVE_DIR="${ROOT_DIR}/${STAGE}"

mkdir -p "${SAVE_DIR}"

{
  echo "[v128] stage=${STAGE}"
  echo "[v128] python=${PYTHON_BIN}"
  "${PYTHON_BIN}" diagnose_candidate_budget_retention_v128.py \
    --output_dir "${SAVE_DIR}"
} 2>&1 | tee "${SAVE_DIR}/pipeline.log"
