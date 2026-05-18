#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
STAGE="${STAGE:-v129_recall_floor_selector_readonly}"
ROOT_DIR="results/natural_teacher_support32_sequence_rollout_0517"
SAVE_DIR="${ROOT_DIR}/${STAGE}"

mkdir -p "${SAVE_DIR}"

{
  echo "[v129] stage=${STAGE}"
  echo "[v129] python=${PYTHON_BIN}"
  "${PYTHON_BIN}" diagnose_candidate_recall_floor_v129.py \
    --output-dir "${SAVE_DIR}"
} 2>&1 | tee "${SAVE_DIR}/pipeline.log"
